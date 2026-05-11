# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1365.20
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
| PARTIAL | 29 |
| TARGET_HIT | 14 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 93 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 50
- **Target hits / Stop hits / Partials:** 14 / 50 / 29
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 16.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 17 | 37.0% | 5 | 29 | 12 | 0.12% | 5.6% |
| BUY @ 2nd Alert (retest1) | 46 | 17 | 37.0% | 5 | 29 | 12 | 0.12% | 5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 26 | 55.3% | 9 | 21 | 17 | 0.23% | 11.0% |
| SELL @ 2nd Alert (retest1) | 47 | 26 | 55.3% | 9 | 21 | 17 | 0.23% | 11.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 93 | 43 | 46.2% | 14 | 50 | 29 | 0.18% | 16.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 1708.70 | 1702.09 | 0.00 | ORB-long ORB[1690.50,1707.70] vol=2.3x ATR=4.92 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 1703.78 | 1701.80 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 11:00:00 | 1684.20 | 1690.89 | 0.00 | ORB-short ORB[1688.00,1699.70] vol=1.8x ATR=4.10 |
| Stop hit — per-position SL triggered | 2025-05-16 11:35:00 | 1688.30 | 1690.23 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 10:40:00 | 1661.80 | 1671.40 | 0.00 | ORB-short ORB[1662.60,1685.80] vol=1.6x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 14:05:00 | 1654.38 | 1666.19 | 0.00 | T1 1.5R @ 1654.38 |
| Target hit | 2025-05-19 15:20:00 | 1656.40 | 1663.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-05-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:00:00 | 1656.70 | 1644.95 | 0.00 | ORB-long ORB[1630.00,1654.70] vol=1.7x ATR=4.93 |
| Stop hit — per-position SL triggered | 2025-05-21 12:10:00 | 1651.77 | 1646.83 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 09:40:00 | 1651.00 | 1643.37 | 0.00 | ORB-long ORB[1634.10,1648.80] vol=3.6x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 09:55:00 | 1658.50 | 1648.33 | 0.00 | T1 1.5R @ 1658.50 |
| Stop hit — per-position SL triggered | 2025-05-22 11:00:00 | 1651.00 | 1657.68 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 1673.20 | 1662.86 | 0.00 | ORB-long ORB[1648.10,1667.30] vol=3.2x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 09:45:00 | 1680.51 | 1669.26 | 0.00 | T1 1.5R @ 1680.51 |
| Target hit | 2025-05-23 11:45:00 | 1680.00 | 1681.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2025-05-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:05:00 | 1682.80 | 1669.83 | 0.00 | ORB-long ORB[1658.00,1679.30] vol=1.8x ATR=5.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 10:45:00 | 1691.30 | 1676.38 | 0.00 | T1 1.5R @ 1691.30 |
| Target hit | 2025-05-26 15:20:00 | 1690.10 | 1683.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-05-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 09:40:00 | 1693.00 | 1688.00 | 0.00 | ORB-long ORB[1681.30,1691.00] vol=2.3x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-05-27 09:45:00 | 1688.76 | 1688.01 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:35:00 | 1710.00 | 1705.50 | 0.00 | ORB-long ORB[1698.00,1707.00] vol=4.7x ATR=4.17 |
| Stop hit — per-position SL triggered | 2025-05-28 09:45:00 | 1705.83 | 1705.65 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:50:00 | 1717.50 | 1703.41 | 0.00 | ORB-long ORB[1690.50,1709.90] vol=2.0x ATR=4.04 |
| Stop hit — per-position SL triggered | 2025-06-02 11:10:00 | 1713.46 | 1705.50 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:20:00 | 1800.50 | 1790.98 | 0.00 | ORB-long ORB[1772.00,1795.00] vol=2.1x ATR=6.11 |
| Stop hit — per-position SL triggered | 2025-06-09 10:30:00 | 1794.39 | 1791.46 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:20:00 | 1813.00 | 1801.10 | 0.00 | ORB-long ORB[1785.00,1806.00] vol=2.1x ATR=7.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 11:05:00 | 1824.29 | 1807.09 | 0.00 | T1 1.5R @ 1824.29 |
| Target hit | 2025-06-16 15:20:00 | 1830.00 | 1824.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-06-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 10:35:00 | 1853.00 | 1836.83 | 0.00 | ORB-long ORB[1815.50,1839.50] vol=3.1x ATR=6.43 |
| Stop hit — per-position SL triggered | 2025-06-17 10:40:00 | 1846.57 | 1837.48 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:00:00 | 1809.00 | 1823.40 | 0.00 | ORB-short ORB[1821.00,1840.00] vol=1.9x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:25:00 | 1799.14 | 1815.17 | 0.00 | T1 1.5R @ 1799.14 |
| Target hit | 2025-06-19 13:05:00 | 1801.00 | 1799.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2025-06-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:50:00 | 1813.50 | 1794.48 | 0.00 | ORB-long ORB[1785.00,1810.00] vol=1.9x ATR=6.65 |
| Stop hit — per-position SL triggered | 2025-06-20 10:55:00 | 1806.85 | 1795.56 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 09:30:00 | 1848.00 | 1857.10 | 0.00 | ORB-short ORB[1849.50,1869.50] vol=2.6x ATR=7.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 09:40:00 | 1837.39 | 1855.52 | 0.00 | T1 1.5R @ 1837.39 |
| Stop hit — per-position SL triggered | 2025-06-24 09:55:00 | 1848.00 | 1854.36 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-06-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:50:00 | 1883.50 | 1875.80 | 0.00 | ORB-long ORB[1863.00,1879.00] vol=1.6x ATR=6.02 |
| Stop hit — per-position SL triggered | 2025-06-25 10:20:00 | 1877.48 | 1877.52 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:30:00 | 1936.60 | 1925.10 | 0.00 | ORB-long ORB[1912.40,1930.00] vol=1.8x ATR=6.00 |
| Stop hit — per-position SL triggered | 2025-07-01 09:40:00 | 1930.60 | 1929.44 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:50:00 | 1959.30 | 1949.80 | 0.00 | ORB-long ORB[1942.50,1954.00] vol=1.7x ATR=5.17 |
| Stop hit — per-position SL triggered | 2025-07-07 10:05:00 | 1954.13 | 1952.02 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:30:00 | 1944.80 | 1951.18 | 0.00 | ORB-short ORB[1944.90,1961.10] vol=1.5x ATR=5.24 |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 1950.04 | 1950.03 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 11:05:00 | 1938.30 | 1948.43 | 0.00 | ORB-short ORB[1941.10,1954.00] vol=3.5x ATR=5.14 |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 1943.44 | 1948.02 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:40:00 | 1845.20 | 1852.20 | 0.00 | ORB-short ORB[1852.50,1869.50] vol=2.0x ATR=5.18 |
| Stop hit — per-position SL triggered | 2025-07-18 11:05:00 | 1850.38 | 1851.20 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:15:00 | 1857.90 | 1865.50 | 0.00 | ORB-short ORB[1870.30,1883.50] vol=5.1x ATR=6.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 13:05:00 | 1848.56 | 1861.71 | 0.00 | T1 1.5R @ 1848.56 |
| Stop hit — per-position SL triggered | 2025-07-22 14:00:00 | 1857.90 | 1860.56 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:40:00 | 1720.40 | 1716.70 | 0.00 | ORB-long ORB[1698.10,1718.90] vol=7.0x ATR=5.46 |
| Stop hit — per-position SL triggered | 2025-07-29 09:55:00 | 1714.94 | 1717.26 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:00:00 | 1720.50 | 1726.90 | 0.00 | ORB-short ORB[1735.00,1748.00] vol=3.2x ATR=5.26 |
| Stop hit — per-position SL triggered | 2025-08-05 10:05:00 | 1725.76 | 1726.52 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:50:00 | 1673.70 | 1684.48 | 0.00 | ORB-short ORB[1691.20,1710.40] vol=1.7x ATR=5.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:10:00 | 1665.04 | 1681.77 | 0.00 | T1 1.5R @ 1665.04 |
| Target hit | 2025-08-06 15:20:00 | 1637.50 | 1653.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-08-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:05:00 | 1645.50 | 1649.45 | 0.00 | ORB-short ORB[1649.10,1664.90] vol=1.8x ATR=3.67 |
| Stop hit — per-position SL triggered | 2025-08-19 11:45:00 | 1649.17 | 1649.29 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:35:00 | 1656.70 | 1645.68 | 0.00 | ORB-long ORB[1632.70,1652.00] vol=1.8x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:40:00 | 1663.98 | 1647.78 | 0.00 | T1 1.5R @ 1663.98 |
| Target hit | 2025-08-20 15:20:00 | 1707.80 | 1686.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:15:00 | 1736.50 | 1743.58 | 0.00 | ORB-short ORB[1737.90,1759.20] vol=2.0x ATR=5.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 11:05:00 | 1728.09 | 1740.32 | 0.00 | T1 1.5R @ 1728.09 |
| Target hit | 2025-09-03 14:15:00 | 1734.40 | 1733.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — SELL (started 2025-09-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 09:45:00 | 1711.00 | 1722.44 | 0.00 | ORB-short ORB[1725.00,1747.70] vol=1.6x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 09:50:00 | 1704.07 | 1719.69 | 0.00 | T1 1.5R @ 1704.07 |
| Target hit | 2025-09-04 11:50:00 | 1706.60 | 1705.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — SELL (started 2025-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:10:00 | 1657.30 | 1689.57 | 0.00 | ORB-short ORB[1696.40,1709.20] vol=4.2x ATR=7.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 1645.48 | 1678.29 | 0.00 | T1 1.5R @ 1645.48 |
| Stop hit — per-position SL triggered | 2025-09-05 10:20:00 | 1657.30 | 1674.27 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:40:00 | 1693.50 | 1681.37 | 0.00 | ORB-long ORB[1667.00,1686.20] vol=3.4x ATR=6.31 |
| Stop hit — per-position SL triggered | 2025-09-09 10:05:00 | 1687.19 | 1686.78 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:00:00 | 1768.00 | 1761.75 | 0.00 | ORB-long ORB[1755.00,1766.40] vol=1.9x ATR=3.74 |
| Stop hit — per-position SL triggered | 2025-09-16 10:05:00 | 1764.26 | 1762.20 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:30:00 | 1692.80 | 1706.70 | 0.00 | ORB-short ORB[1703.10,1723.20] vol=1.7x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:35:00 | 1685.38 | 1703.89 | 0.00 | T1 1.5R @ 1685.38 |
| Stop hit — per-position SL triggered | 2025-09-23 10:35:00 | 1692.80 | 1697.23 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 09:30:00 | 1560.70 | 1566.78 | 0.00 | ORB-short ORB[1563.10,1583.90] vol=5.2x ATR=6.99 |
| Stop hit — per-position SL triggered | 2025-09-30 09:40:00 | 1567.69 | 1568.45 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:15:00 | 1566.10 | 1570.35 | 0.00 | ORB-short ORB[1572.90,1592.70] vol=1.9x ATR=5.10 |
| Stop hit — per-position SL triggered | 2025-10-01 11:20:00 | 1571.20 | 1570.30 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:30:00 | 1610.40 | 1600.07 | 0.00 | ORB-long ORB[1588.50,1603.80] vol=1.5x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 09:55:00 | 1618.77 | 1604.59 | 0.00 | T1 1.5R @ 1618.77 |
| Stop hit — per-position SL triggered | 2025-10-03 10:20:00 | 1610.40 | 1605.84 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:20:00 | 1629.70 | 1621.25 | 0.00 | ORB-long ORB[1598.10,1621.90] vol=2.2x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:35:00 | 1637.55 | 1623.94 | 0.00 | T1 1.5R @ 1637.55 |
| Target hit | 2025-10-06 15:20:00 | 1660.90 | 1642.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2025-10-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:05:00 | 1702.60 | 1717.42 | 0.00 | ORB-short ORB[1716.30,1734.90] vol=1.7x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 1694.99 | 1716.31 | 0.00 | T1 1.5R @ 1694.99 |
| Stop hit — per-position SL triggered | 2025-10-14 12:35:00 | 1702.60 | 1707.50 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:40:00 | 1777.80 | 1769.04 | 0.00 | ORB-long ORB[1756.90,1776.90] vol=2.0x ATR=8.32 |
| Stop hit — per-position SL triggered | 2025-10-23 09:50:00 | 1769.48 | 1770.13 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:50:00 | 1817.90 | 1832.37 | 0.00 | ORB-short ORB[1829.40,1845.90] vol=1.6x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:00:00 | 1810.55 | 1830.52 | 0.00 | T1 1.5R @ 1810.55 |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 1817.90 | 1828.93 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:10:00 | 1783.90 | 1799.36 | 0.00 | ORB-short ORB[1809.80,1822.00] vol=2.9x ATR=5.66 |
| Stop hit — per-position SL triggered | 2025-10-29 10:25:00 | 1789.56 | 1796.57 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:30:00 | 1798.00 | 1804.38 | 0.00 | ORB-short ORB[1800.70,1810.90] vol=3.6x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-10-31 10:40:00 | 1802.08 | 1804.23 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:25:00 | 1763.10 | 1743.77 | 0.00 | ORB-long ORB[1723.30,1749.30] vol=2.8x ATR=7.37 |
| Stop hit — per-position SL triggered | 2025-11-10 13:40:00 | 1755.73 | 1754.42 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:25:00 | 1782.40 | 1770.34 | 0.00 | ORB-long ORB[1762.70,1778.60] vol=2.7x ATR=5.30 |
| Stop hit — per-position SL triggered | 2025-11-11 10:35:00 | 1777.10 | 1772.41 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:50:00 | 1826.00 | 1815.51 | 0.00 | ORB-long ORB[1801.00,1819.80] vol=1.6x ATR=4.64 |
| Stop hit — per-position SL triggered | 2025-11-12 11:05:00 | 1821.36 | 1816.91 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:55:00 | 1780.30 | 1786.13 | 0.00 | ORB-short ORB[1788.20,1805.00] vol=1.7x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 12:35:00 | 1773.46 | 1782.87 | 0.00 | T1 1.5R @ 1773.46 |
| Target hit | 2025-11-14 14:00:00 | 1777.50 | 1777.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 48 — BUY (started 2025-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:35:00 | 1852.00 | 1842.34 | 0.00 | ORB-long ORB[1830.20,1845.70] vol=1.7x ATR=5.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 09:50:00 | 1859.80 | 1847.48 | 0.00 | T1 1.5R @ 1859.80 |
| Stop hit — per-position SL triggered | 2025-11-26 10:30:00 | 1852.00 | 1848.94 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:55:00 | 1899.40 | 1884.31 | 0.00 | ORB-long ORB[1872.80,1887.20] vol=2.9x ATR=5.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:05:00 | 1907.10 | 1891.42 | 0.00 | T1 1.5R @ 1907.10 |
| Stop hit — per-position SL triggered | 2025-11-27 12:00:00 | 1899.40 | 1898.82 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:45:00 | 1980.10 | 1969.39 | 0.00 | ORB-long ORB[1958.60,1977.00] vol=2.3x ATR=6.37 |
| Stop hit — per-position SL triggered | 2025-12-05 14:40:00 | 1973.73 | 1977.49 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:55:00 | 1851.30 | 1874.39 | 0.00 | ORB-short ORB[1873.00,1889.00] vol=1.6x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-12-10 11:00:00 | 1856.34 | 1873.53 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:50:00 | 1872.00 | 1855.79 | 0.00 | ORB-long ORB[1845.00,1864.10] vol=5.5x ATR=5.41 |
| Stop hit — per-position SL triggered | 2025-12-15 10:55:00 | 1866.59 | 1857.04 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:40:00 | 1750.30 | 1761.27 | 0.00 | ORB-short ORB[1758.50,1781.00] vol=2.2x ATR=5.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 09:50:00 | 1741.34 | 1757.72 | 0.00 | T1 1.5R @ 1741.34 |
| Stop hit — per-position SL triggered | 2025-12-24 09:55:00 | 1750.30 | 1757.17 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:00:00 | 1655.50 | 1665.52 | 0.00 | ORB-short ORB[1662.20,1684.90] vol=2.0x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:10:00 | 1648.06 | 1663.53 | 0.00 | T1 1.5R @ 1648.06 |
| Target hit | 2025-12-30 13:50:00 | 1650.50 | 1650.12 | 0.00 | Trail-exit close>VWAP |

### Cycle 55 — SELL (started 2026-01-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 10:30:00 | 1639.40 | 1643.70 | 0.00 | ORB-short ORB[1644.20,1662.90] vol=2.6x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:55:00 | 1633.54 | 1641.93 | 0.00 | T1 1.5R @ 1633.54 |
| Stop hit — per-position SL triggered | 2026-01-02 12:30:00 | 1639.40 | 1637.46 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-01-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:45:00 | 1652.20 | 1643.95 | 0.00 | ORB-long ORB[1639.40,1650.00] vol=1.7x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:05:00 | 1659.00 | 1646.45 | 0.00 | T1 1.5R @ 1659.00 |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 1652.20 | 1647.73 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 1677.10 | 1691.64 | 0.00 | ORB-short ORB[1690.70,1707.70] vol=2.2x ATR=4.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:55:00 | 1670.11 | 1686.56 | 0.00 | T1 1.5R @ 1670.11 |
| Target hit | 2026-01-08 15:20:00 | 1645.10 | 1670.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2026-01-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:10:00 | 1690.70 | 1698.21 | 0.00 | ORB-short ORB[1697.90,1717.00] vol=2.8x ATR=6.46 |
| Stop hit — per-position SL triggered | 2026-01-13 10:20:00 | 1697.16 | 1697.86 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:25:00 | 1704.50 | 1710.82 | 0.00 | ORB-short ORB[1705.70,1729.20] vol=1.6x ATR=5.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 12:45:00 | 1695.73 | 1707.40 | 0.00 | T1 1.5R @ 1695.73 |
| Target hit | 2026-01-14 15:20:00 | 1685.60 | 1695.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 1662.20 | 1656.05 | 0.00 | ORB-long ORB[1644.60,1661.40] vol=2.7x ATR=5.40 |
| Stop hit — per-position SL triggered | 2026-01-30 13:30:00 | 1656.80 | 1660.48 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 1563.60 | 1557.29 | 0.00 | ORB-long ORB[1542.40,1561.00] vol=1.9x ATR=4.48 |
| Stop hit — per-position SL triggered | 2026-02-10 10:10:00 | 1559.12 | 1559.72 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-02-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:05:00 | 1531.30 | 1544.47 | 0.00 | ORB-short ORB[1550.30,1560.00] vol=2.0x ATR=4.68 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 1535.98 | 1543.22 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 1293.50 | 1289.35 | 0.00 | ORB-long ORB[1280.00,1292.40] vol=6.9x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:05:00 | 1298.23 | 1289.59 | 0.00 | T1 1.5R @ 1298.23 |
| Stop hit — per-position SL triggered | 2026-04-21 14:10:00 | 1293.50 | 1293.00 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:45:00 | 1217.70 | 1209.01 | 0.00 | ORB-long ORB[1201.50,1214.00] vol=1.7x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:50:00 | 1224.61 | 1214.07 | 0.00 | T1 1.5R @ 1224.61 |
| Stop hit — per-position SL triggered | 2026-04-29 09:55:00 | 1217.70 | 1214.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:30:00 | 1708.70 | 2025-05-15 09:35:00 | 1703.78 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-16 11:00:00 | 1684.20 | 2025-05-16 11:35:00 | 1688.30 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-19 10:40:00 | 1661.80 | 2025-05-19 14:05:00 | 1654.38 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-05-19 10:40:00 | 1661.80 | 2025-05-19 15:20:00 | 1656.40 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2025-05-21 11:00:00 | 1656.70 | 2025-05-21 12:10:00 | 1651.77 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-22 09:40:00 | 1651.00 | 2025-05-22 09:55:00 | 1658.50 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-22 09:40:00 | 1651.00 | 2025-05-22 11:00:00 | 1651.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-23 09:30:00 | 1673.20 | 2025-05-23 09:45:00 | 1680.51 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-05-23 09:30:00 | 1673.20 | 2025-05-23 11:45:00 | 1680.00 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2025-05-26 10:05:00 | 1682.80 | 2025-05-26 10:45:00 | 1691.30 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-05-26 10:05:00 | 1682.80 | 2025-05-26 15:20:00 | 1690.10 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-05-27 09:40:00 | 1693.00 | 2025-05-27 09:45:00 | 1688.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-28 09:35:00 | 1710.00 | 2025-05-28 09:45:00 | 1705.83 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-02 10:50:00 | 1717.50 | 2025-06-02 11:10:00 | 1713.46 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-09 10:20:00 | 1800.50 | 2025-06-09 10:30:00 | 1794.39 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-16 10:20:00 | 1813.00 | 2025-06-16 11:05:00 | 1824.29 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-06-16 10:20:00 | 1813.00 | 2025-06-16 15:20:00 | 1830.00 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2025-06-17 10:35:00 | 1853.00 | 2025-06-17 10:40:00 | 1846.57 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-06-19 10:00:00 | 1809.00 | 2025-06-19 10:25:00 | 1799.14 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-06-19 10:00:00 | 1809.00 | 2025-06-19 13:05:00 | 1801.00 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-20 10:50:00 | 1813.50 | 2025-06-20 10:55:00 | 1806.85 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-06-24 09:30:00 | 1848.00 | 2025-06-24 09:40:00 | 1837.39 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-06-24 09:30:00 | 1848.00 | 2025-06-24 09:55:00 | 1848.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-25 09:50:00 | 1883.50 | 2025-06-25 10:20:00 | 1877.48 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-01 09:30:00 | 1936.60 | 2025-07-01 09:40:00 | 1930.60 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-07 09:50:00 | 1959.30 | 2025-07-07 10:05:00 | 1954.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-08 10:30:00 | 1944.80 | 2025-07-08 11:15:00 | 1950.04 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-09 11:05:00 | 1938.30 | 2025-07-09 11:15:00 | 1943.44 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-18 10:40:00 | 1845.20 | 2025-07-18 11:05:00 | 1850.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-22 11:15:00 | 1857.90 | 2025-07-22 13:05:00 | 1848.56 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-22 11:15:00 | 1857.90 | 2025-07-22 14:00:00 | 1857.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-29 09:40:00 | 1720.40 | 2025-07-29 09:55:00 | 1714.94 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-05 10:00:00 | 1720.50 | 2025-08-05 10:05:00 | 1725.76 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-06 09:50:00 | 1673.70 | 2025-08-06 10:10:00 | 1665.04 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-08-06 09:50:00 | 1673.70 | 2025-08-06 15:20:00 | 1637.50 | TARGET_HIT | 0.50 | 2.16% |
| SELL | retest1 | 2025-08-19 11:05:00 | 1645.50 | 2025-08-19 11:45:00 | 1649.17 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-08-20 10:35:00 | 1656.70 | 2025-08-20 10:40:00 | 1663.98 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-08-20 10:35:00 | 1656.70 | 2025-08-20 15:20:00 | 1707.80 | TARGET_HIT | 0.50 | 3.08% |
| SELL | retest1 | 2025-09-03 10:15:00 | 1736.50 | 2025-09-03 11:05:00 | 1728.09 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-09-03 10:15:00 | 1736.50 | 2025-09-03 14:15:00 | 1734.40 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2025-09-04 09:45:00 | 1711.00 | 2025-09-04 09:50:00 | 1704.07 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-09-04 09:45:00 | 1711.00 | 2025-09-04 11:50:00 | 1706.60 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-09-05 10:10:00 | 1657.30 | 2025-09-05 10:15:00 | 1645.48 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2025-09-05 10:10:00 | 1657.30 | 2025-09-05 10:20:00 | 1657.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-09 09:40:00 | 1693.50 | 2025-09-09 10:05:00 | 1687.19 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-09-16 10:00:00 | 1768.00 | 2025-09-16 10:05:00 | 1764.26 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-23 09:30:00 | 1692.80 | 2025-09-23 09:35:00 | 1685.38 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-09-23 09:30:00 | 1692.80 | 2025-09-23 10:35:00 | 1692.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-30 09:30:00 | 1560.70 | 2025-09-30 09:40:00 | 1567.69 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-10-01 11:15:00 | 1566.10 | 2025-10-01 11:20:00 | 1571.20 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-03 09:30:00 | 1610.40 | 2025-10-03 09:55:00 | 1618.77 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-10-03 09:30:00 | 1610.40 | 2025-10-03 10:20:00 | 1610.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-06 10:20:00 | 1629.70 | 2025-10-06 10:35:00 | 1637.55 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-06 10:20:00 | 1629.70 | 2025-10-06 15:20:00 | 1660.90 | TARGET_HIT | 0.50 | 1.91% |
| SELL | retest1 | 2025-10-14 11:05:00 | 1702.60 | 2025-10-14 11:15:00 | 1694.99 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-10-14 11:05:00 | 1702.60 | 2025-10-14 12:35:00 | 1702.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-23 09:40:00 | 1777.80 | 2025-10-23 09:50:00 | 1769.48 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-10-28 10:50:00 | 1817.90 | 2025-10-28 11:00:00 | 1810.55 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-28 10:50:00 | 1817.90 | 2025-10-28 11:15:00 | 1817.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-29 10:10:00 | 1783.90 | 2025-10-29 10:25:00 | 1789.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-31 10:30:00 | 1798.00 | 2025-10-31 10:40:00 | 1802.08 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-10 10:25:00 | 1763.10 | 2025-11-10 13:40:00 | 1755.73 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-11-11 10:25:00 | 1782.40 | 2025-11-11 10:35:00 | 1777.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-12 10:50:00 | 1826.00 | 2025-11-12 11:05:00 | 1821.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-14 10:55:00 | 1780.30 | 2025-11-14 12:35:00 | 1773.46 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-11-14 10:55:00 | 1780.30 | 2025-11-14 14:00:00 | 1777.50 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-11-26 09:35:00 | 1852.00 | 2025-11-26 09:50:00 | 1859.80 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-11-26 09:35:00 | 1852.00 | 2025-11-26 10:30:00 | 1852.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-27 09:55:00 | 1899.40 | 2025-11-27 10:05:00 | 1907.10 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-11-27 09:55:00 | 1899.40 | 2025-11-27 12:00:00 | 1899.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:45:00 | 1980.10 | 2025-12-05 14:40:00 | 1973.73 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-10 10:55:00 | 1851.30 | 2025-12-10 11:00:00 | 1856.34 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-15 10:50:00 | 1872.00 | 2025-12-15 10:55:00 | 1866.59 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-24 09:40:00 | 1750.30 | 2025-12-24 09:50:00 | 1741.34 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-24 09:40:00 | 1750.30 | 2025-12-24 09:55:00 | 1750.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 11:00:00 | 1655.50 | 2025-12-30 11:10:00 | 1648.06 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-12-30 11:00:00 | 1655.50 | 2025-12-30 13:50:00 | 1650.50 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-01-02 10:30:00 | 1639.40 | 2026-01-02 10:55:00 | 1633.54 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-02 10:30:00 | 1639.40 | 2026-01-02 12:30:00 | 1639.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-06 10:45:00 | 1652.20 | 2026-01-06 11:05:00 | 1659.00 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-01-06 10:45:00 | 1652.20 | 2026-01-06 11:15:00 | 1652.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 11:00:00 | 1677.10 | 2026-01-08 11:55:00 | 1670.11 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-01-08 11:00:00 | 1677.10 | 2026-01-08 15:20:00 | 1645.10 | TARGET_HIT | 0.50 | 1.91% |
| SELL | retest1 | 2026-01-13 10:10:00 | 1690.70 | 2026-01-13 10:20:00 | 1697.16 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-01-14 10:25:00 | 1704.50 | 2026-01-14 12:45:00 | 1695.73 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-01-14 10:25:00 | 1704.50 | 2026-01-14 15:20:00 | 1685.60 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2026-01-30 09:30:00 | 1662.20 | 2026-01-30 13:30:00 | 1656.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-10 09:40:00 | 1563.60 | 2026-02-10 10:10:00 | 1559.12 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-11 10:05:00 | 1531.30 | 2026-02-11 10:15:00 | 1535.98 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-21 10:55:00 | 1293.50 | 2026-04-21 11:05:00 | 1298.23 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-04-21 10:55:00 | 1293.50 | 2026-04-21 14:10:00 | 1293.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:45:00 | 1217.70 | 2026-04-29 09:50:00 | 1224.61 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-29 09:45:00 | 1217.70 | 2026-04-29 09:55:00 | 1217.70 | STOP_HIT | 0.50 | 0.00% |
