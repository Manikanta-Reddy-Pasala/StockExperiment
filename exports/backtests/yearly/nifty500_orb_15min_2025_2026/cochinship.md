# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16888 bars)
- **Last close:** 1769.40
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
| ENTRY1 | 48 |
| ENTRY2 | 0 |
| PARTIAL | 17 |
| TARGET_HIT | 6 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 42
- **Target hits / Stop hits / Partials:** 6 / 42 / 17
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 3.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 10 | 34.5% | 3 | 19 | 7 | -0.01% | -0.2% |
| BUY @ 2nd Alert (retest1) | 29 | 10 | 34.5% | 3 | 19 | 7 | -0.01% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 36 | 13 | 36.1% | 3 | 23 | 10 | 0.12% | 4.1% |
| SELL @ 2nd Alert (retest1) | 36 | 13 | 36.1% | 3 | 23 | 10 | 0.12% | 4.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 65 | 23 | 35.4% | 6 | 42 | 17 | 0.06% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:15:00 | 2058.60 | 2076.72 | 0.00 | ORB-short ORB[2064.20,2089.50] vol=1.7x ATR=8.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 10:55:00 | 2045.62 | 2071.56 | 0.00 | T1 1.5R @ 2045.62 |
| Target hit | 2025-07-01 15:20:00 | 2050.00 | 2059.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 2017.50 | 2036.03 | 0.00 | ORB-short ORB[2032.00,2059.90] vol=1.9x ATR=6.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:45:00 | 2007.46 | 2031.76 | 0.00 | T1 1.5R @ 2007.46 |
| Stop hit — per-position SL triggered | 2025-07-02 09:50:00 | 2017.50 | 2030.51 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:40:00 | 2068.70 | 2048.78 | 0.00 | ORB-long ORB[2031.00,2060.50] vol=3.0x ATR=8.87 |
| Stop hit — per-position SL triggered | 2025-07-04 09:45:00 | 2059.83 | 2050.67 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-07-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 09:40:00 | 2037.00 | 2054.84 | 0.00 | ORB-short ORB[2048.00,2074.00] vol=2.5x ATR=8.80 |
| Stop hit — per-position SL triggered | 2025-07-07 09:55:00 | 2045.80 | 2053.63 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:35:00 | 2028.00 | 2047.67 | 0.00 | ORB-short ORB[2037.20,2065.60] vol=3.6x ATR=7.85 |
| Stop hit — per-position SL triggered | 2025-07-08 10:25:00 | 2035.85 | 2040.48 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:40:00 | 1914.60 | 1925.25 | 0.00 | ORB-short ORB[1924.10,1934.80] vol=2.1x ATR=5.31 |
| Stop hit — per-position SL triggered | 2025-07-16 09:55:00 | 1919.91 | 1924.02 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-07-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:10:00 | 1828.90 | 1842.30 | 0.00 | ORB-short ORB[1837.10,1855.00] vol=1.7x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:50:00 | 1820.72 | 1836.97 | 0.00 | T1 1.5R @ 1820.72 |
| Target hit | 2025-07-25 15:20:00 | 1803.80 | 1821.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-07-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:35:00 | 1757.10 | 1741.17 | 0.00 | ORB-long ORB[1722.90,1743.60] vol=2.1x ATR=9.15 |
| Stop hit — per-position SL triggered | 2025-07-29 09:40:00 | 1747.95 | 1742.20 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 1702.20 | 1716.23 | 0.00 | ORB-short ORB[1708.20,1731.90] vol=1.6x ATR=6.20 |
| Stop hit — per-position SL triggered | 2025-08-20 09:35:00 | 1708.40 | 1712.69 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:50:00 | 1754.90 | 1735.82 | 0.00 | ORB-long ORB[1714.00,1730.70] vol=6.5x ATR=6.85 |
| Stop hit — per-position SL triggered | 2025-08-21 10:10:00 | 1748.05 | 1748.70 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-09-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:45:00 | 1638.40 | 1623.59 | 0.00 | ORB-long ORB[1609.10,1627.80] vol=2.6x ATR=6.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:50:00 | 1647.65 | 1627.37 | 0.00 | T1 1.5R @ 1647.65 |
| Stop hit — per-position SL triggered | 2025-09-01 09:55:00 | 1638.40 | 1628.64 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-09-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:05:00 | 1701.00 | 1681.44 | 0.00 | ORB-long ORB[1664.10,1682.00] vol=1.6x ATR=8.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:30:00 | 1713.74 | 1690.87 | 0.00 | T1 1.5R @ 1713.74 |
| Stop hit — per-position SL triggered | 2025-09-02 10:40:00 | 1701.00 | 1691.95 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:50:00 | 1762.40 | 1748.27 | 0.00 | ORB-long ORB[1733.10,1755.00] vol=1.7x ATR=7.31 |
| Stop hit — per-position SL triggered | 2025-09-03 10:45:00 | 1755.09 | 1755.10 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-10-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:50:00 | 1818.30 | 1828.76 | 0.00 | ORB-short ORB[1825.00,1842.50] vol=1.9x ATR=12.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:20:00 | 1799.54 | 1820.26 | 0.00 | T1 1.5R @ 1799.54 |
| Stop hit — per-position SL triggered | 2025-10-08 13:20:00 | 1818.30 | 1811.94 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-10-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:45:00 | 1796.10 | 1786.97 | 0.00 | ORB-long ORB[1778.70,1793.80] vol=2.2x ATR=7.07 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 1789.03 | 1790.21 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:30:00 | 1810.40 | 1804.25 | 0.00 | ORB-long ORB[1793.00,1806.10] vol=2.4x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-10-16 09:35:00 | 1806.16 | 1804.44 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:35:00 | 1797.00 | 1785.41 | 0.00 | ORB-long ORB[1773.00,1791.00] vol=1.7x ATR=5.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:45:00 | 1804.93 | 1789.41 | 0.00 | T1 1.5R @ 1804.93 |
| Stop hit — per-position SL triggered | 2025-10-17 09:50:00 | 1797.00 | 1790.09 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-10-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:10:00 | 1838.60 | 1823.09 | 0.00 | ORB-long ORB[1809.70,1833.00] vol=2.3x ATR=8.22 |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 1830.38 | 1824.43 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-10-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 09:50:00 | 1815.70 | 1826.42 | 0.00 | ORB-short ORB[1825.00,1838.30] vol=1.8x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1820.49 | 1824.40 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:15:00 | 1769.50 | 1776.13 | 0.00 | ORB-short ORB[1777.00,1791.70] vol=2.0x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:30:00 | 1764.86 | 1775.69 | 0.00 | T1 1.5R @ 1764.86 |
| Stop hit — per-position SL triggered | 2025-11-04 15:00:00 | 1769.50 | 1772.00 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:35:00 | 1673.50 | 1683.21 | 0.00 | ORB-short ORB[1679.00,1699.00] vol=1.9x ATR=4.52 |
| Stop hit — per-position SL triggered | 2025-11-07 09:40:00 | 1678.02 | 1682.77 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-11-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:30:00 | 1762.30 | 1745.87 | 0.00 | ORB-long ORB[1725.80,1748.00] vol=3.2x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:35:00 | 1772.68 | 1759.02 | 0.00 | T1 1.5R @ 1772.68 |
| Target hit | 2025-11-10 09:55:00 | 1763.70 | 1764.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2025-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:30:00 | 1765.90 | 1758.75 | 0.00 | ORB-long ORB[1748.20,1765.60] vol=2.4x ATR=5.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 09:45:00 | 1773.80 | 1761.97 | 0.00 | T1 1.5R @ 1773.80 |
| Target hit | 2025-11-12 14:25:00 | 1786.00 | 1787.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2025-11-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:50:00 | 1704.80 | 1713.76 | 0.00 | ORB-short ORB[1712.60,1726.40] vol=1.8x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 11:00:00 | 1697.73 | 1708.29 | 0.00 | T1 1.5R @ 1697.73 |
| Stop hit — per-position SL triggered | 2025-11-21 12:55:00 | 1704.80 | 1706.75 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 1687.50 | 1680.54 | 0.00 | ORB-long ORB[1672.00,1684.00] vol=2.2x ATR=4.26 |
| Stop hit — per-position SL triggered | 2025-11-27 09:35:00 | 1683.24 | 1680.71 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:30:00 | 1655.60 | 1664.58 | 0.00 | ORB-short ORB[1661.00,1675.00] vol=2.1x ATR=3.73 |
| Stop hit — per-position SL triggered | 2025-12-02 09:35:00 | 1659.33 | 1663.92 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 1621.00 | 1634.74 | 0.00 | ORB-short ORB[1627.60,1647.90] vol=2.4x ATR=4.90 |
| Stop hit — per-position SL triggered | 2025-12-03 09:45:00 | 1625.90 | 1631.22 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:30:00 | 1633.00 | 1625.57 | 0.00 | ORB-long ORB[1614.00,1631.60] vol=1.5x ATR=4.89 |
| Stop hit — per-position SL triggered | 2025-12-04 09:50:00 | 1628.11 | 1626.74 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-12-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:55:00 | 1628.00 | 1644.25 | 0.00 | ORB-short ORB[1643.50,1660.00] vol=1.7x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-12-08 10:05:00 | 1633.04 | 1643.17 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:35:00 | 1595.40 | 1604.95 | 0.00 | ORB-short ORB[1602.40,1619.10] vol=2.0x ATR=5.76 |
| Stop hit — per-position SL triggered | 2025-12-11 10:05:00 | 1601.16 | 1601.83 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-12-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:00:00 | 1567.50 | 1575.51 | 0.00 | ORB-short ORB[1572.70,1587.60] vol=1.8x ATR=3.63 |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 1571.13 | 1574.53 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 09:45:00 | 1541.00 | 1549.26 | 0.00 | ORB-short ORB[1542.00,1565.00] vol=1.8x ATR=4.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 10:00:00 | 1534.18 | 1546.22 | 0.00 | T1 1.5R @ 1534.18 |
| Target hit | 2025-12-17 15:20:00 | 1499.90 | 1519.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 1526.30 | 1519.65 | 0.00 | ORB-long ORB[1510.00,1525.00] vol=1.6x ATR=5.51 |
| Stop hit — per-position SL triggered | 2025-12-19 09:40:00 | 1520.79 | 1520.28 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 1679.90 | 1667.89 | 0.00 | ORB-long ORB[1655.00,1674.60] vol=2.8x ATR=6.48 |
| Stop hit — per-position SL triggered | 2025-12-29 09:40:00 | 1673.42 | 1671.74 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-01-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 09:30:00 | 1619.00 | 1623.28 | 0.00 | ORB-short ORB[1620.00,1633.30] vol=2.5x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:45:00 | 1612.16 | 1619.34 | 0.00 | T1 1.5R @ 1612.16 |
| Stop hit — per-position SL triggered | 2026-01-06 11:10:00 | 1619.00 | 1616.98 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2026-01-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:30:00 | 1563.20 | 1570.49 | 0.00 | ORB-short ORB[1564.10,1582.50] vol=1.9x ATR=6.22 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 1569.42 | 1568.74 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2026-01-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 11:05:00 | 1542.70 | 1548.57 | 0.00 | ORB-short ORB[1545.00,1558.00] vol=1.6x ATR=4.62 |
| Stop hit — per-position SL triggered | 2026-01-14 11:25:00 | 1547.32 | 1548.08 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2026-01-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:35:00 | 1494.20 | 1487.60 | 0.00 | ORB-long ORB[1475.00,1490.00] vol=3.0x ATR=7.08 |
| Stop hit — per-position SL triggered | 2026-01-22 09:55:00 | 1487.12 | 1488.17 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 1653.20 | 1673.98 | 0.00 | ORB-short ORB[1665.00,1689.00] vol=3.6x ATR=9.79 |
| Stop hit — per-position SL triggered | 2026-02-01 11:20:00 | 1662.99 | 1671.16 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:30:00 | 1488.90 | 1480.11 | 0.00 | ORB-long ORB[1472.10,1481.80] vol=2.8x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-02-06 09:35:00 | 1484.40 | 1481.11 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 1509.00 | 1517.74 | 0.00 | ORB-short ORB[1512.50,1534.90] vol=2.3x ATR=5.09 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 1514.09 | 1515.44 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1481.60 | 1488.22 | 0.00 | ORB-short ORB[1482.80,1501.80] vol=1.5x ATR=5.62 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 1487.22 | 1486.92 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-02-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:10:00 | 1493.00 | 1488.01 | 0.00 | ORB-long ORB[1480.10,1492.70] vol=2.7x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:40:00 | 1499.92 | 1491.33 | 0.00 | T1 1.5R @ 1499.92 |
| Target hit | 2026-02-26 15:20:00 | 1497.00 | 1501.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:15:00 | 1479.00 | 1483.50 | 0.00 | ORB-short ORB[1482.10,1498.00] vol=2.0x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:50:00 | 1474.72 | 1482.18 | 0.00 | T1 1.5R @ 1474.72 |
| Stop hit — per-position SL triggered | 2026-02-27 13:55:00 | 1479.00 | 1480.92 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-03-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:00:00 | 1428.60 | 1413.62 | 0.00 | ORB-long ORB[1400.20,1419.00] vol=1.7x ATR=6.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:05:00 | 1438.59 | 1417.96 | 0.00 | T1 1.5R @ 1438.59 |
| Stop hit — per-position SL triggered | 2026-03-05 10:10:00 | 1428.60 | 1418.47 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 1489.30 | 1500.78 | 0.00 | ORB-short ORB[1490.50,1511.80] vol=3.0x ATR=6.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:45:00 | 1479.23 | 1494.65 | 0.00 | T1 1.5R @ 1479.23 |
| Stop hit — per-position SL triggered | 2026-04-16 11:00:00 | 1489.30 | 1488.32 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-04-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:55:00 | 1596.40 | 1587.93 | 0.00 | ORB-long ORB[1571.80,1592.20] vol=2.1x ATR=5.61 |
| Stop hit — per-position SL triggered | 2026-04-23 10:45:00 | 1590.79 | 1591.75 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:45:00 | 1749.30 | 1734.21 | 0.00 | ORB-long ORB[1721.90,1737.50] vol=2.4x ATR=6.83 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1742.47 | 1740.33 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-07-01 10:15:00 | 2058.60 | 2025-07-01 10:55:00 | 2045.62 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-07-01 10:15:00 | 2058.60 | 2025-07-01 15:20:00 | 2050.00 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-02 09:40:00 | 2017.50 | 2025-07-02 09:45:00 | 2007.46 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-02 09:40:00 | 2017.50 | 2025-07-02 09:50:00 | 2017.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-04 09:40:00 | 2068.70 | 2025-07-04 09:45:00 | 2059.83 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-07-07 09:40:00 | 2037.00 | 2025-07-07 09:55:00 | 2045.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-07-08 09:35:00 | 2028.00 | 2025-07-08 10:25:00 | 2035.85 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-07-16 09:40:00 | 1914.60 | 2025-07-16 09:55:00 | 1919.91 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-25 10:10:00 | 1828.90 | 2025-07-25 10:50:00 | 1820.72 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-07-25 10:10:00 | 1828.90 | 2025-07-25 15:20:00 | 1803.80 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2025-07-29 09:35:00 | 1757.10 | 2025-07-29 09:40:00 | 1747.95 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-08-20 09:30:00 | 1702.20 | 2025-08-20 09:35:00 | 1708.40 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-08-21 09:50:00 | 1754.90 | 2025-08-21 10:10:00 | 1748.05 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-01 09:45:00 | 1638.40 | 2025-09-01 09:50:00 | 1647.65 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-09-01 09:45:00 | 1638.40 | 2025-09-01 09:55:00 | 1638.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 10:05:00 | 1701.00 | 2025-09-02 10:30:00 | 1713.74 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2025-09-02 10:05:00 | 1701.00 | 2025-09-02 10:40:00 | 1701.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 09:50:00 | 1762.40 | 2025-09-03 10:45:00 | 1755.09 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-10-08 09:50:00 | 1818.30 | 2025-10-08 10:20:00 | 1799.54 | PARTIAL | 0.50 | 1.03% |
| SELL | retest1 | 2025-10-08 09:50:00 | 1818.30 | 2025-10-08 13:20:00 | 1818.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 09:45:00 | 1796.10 | 2025-10-10 10:15:00 | 1789.03 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-10-16 09:30:00 | 1810.40 | 2025-10-16 09:35:00 | 1806.16 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-17 09:35:00 | 1797.00 | 2025-10-17 09:45:00 | 1804.93 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-17 09:35:00 | 1797.00 | 2025-10-17 09:50:00 | 1797.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-24 10:10:00 | 1838.60 | 2025-10-24 10:15:00 | 1830.38 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-10-28 09:50:00 | 1815.70 | 2025-10-28 10:15:00 | 1820.49 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-04 11:15:00 | 1769.50 | 2025-11-04 11:30:00 | 1764.86 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-11-04 11:15:00 | 1769.50 | 2025-11-04 15:00:00 | 1769.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-07 09:35:00 | 1673.50 | 2025-11-07 09:40:00 | 1678.02 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-10 09:30:00 | 1762.30 | 2025-11-10 09:35:00 | 1772.68 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-11-10 09:30:00 | 1762.30 | 2025-11-10 09:55:00 | 1763.70 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2025-11-12 09:30:00 | 1765.90 | 2025-11-12 09:45:00 | 1773.80 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-12 09:30:00 | 1765.90 | 2025-11-12 14:25:00 | 1786.00 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-11-21 09:50:00 | 1704.80 | 2025-11-21 11:00:00 | 1697.73 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-21 09:50:00 | 1704.80 | 2025-11-21 12:55:00 | 1704.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-27 09:30:00 | 1687.50 | 2025-11-27 09:35:00 | 1683.24 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-02 09:30:00 | 1655.60 | 2025-12-02 09:35:00 | 1659.33 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-03 09:30:00 | 1621.00 | 2025-12-03 09:45:00 | 1625.90 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-04 09:30:00 | 1633.00 | 2025-12-04 09:50:00 | 1628.11 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-08 09:55:00 | 1628.00 | 2025-12-08 10:05:00 | 1633.04 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-11 09:35:00 | 1595.40 | 2025-12-11 10:05:00 | 1601.16 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-16 10:00:00 | 1567.50 | 2025-12-16 10:15:00 | 1571.13 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-17 09:45:00 | 1541.00 | 2025-12-17 10:00:00 | 1534.18 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-12-17 09:45:00 | 1541.00 | 2025-12-17 15:20:00 | 1499.90 | TARGET_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2025-12-19 09:30:00 | 1526.30 | 2025-12-19 09:40:00 | 1520.79 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-12-29 09:30:00 | 1679.90 | 2025-12-29 09:40:00 | 1673.42 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-01-06 09:30:00 | 1619.00 | 2026-01-06 09:45:00 | 1612.16 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-01-06 09:30:00 | 1619.00 | 2026-01-06 11:10:00 | 1619.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-13 09:30:00 | 1563.20 | 2026-01-13 09:45:00 | 1569.42 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-01-14 11:05:00 | 1542.70 | 2026-01-14 11:25:00 | 1547.32 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-22 09:35:00 | 1494.20 | 2026-01-22 09:55:00 | 1487.12 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-02-01 11:10:00 | 1653.20 | 2026-02-01 11:20:00 | 1662.99 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2026-02-06 09:30:00 | 1488.90 | 2026-02-06 09:35:00 | 1484.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-11 09:30:00 | 1509.00 | 2026-02-11 09:45:00 | 1514.09 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-24 09:30:00 | 1481.60 | 2026-02-24 09:45:00 | 1487.22 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-26 10:10:00 | 1493.00 | 2026-02-26 10:40:00 | 1499.92 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-26 10:10:00 | 1493.00 | 2026-02-26 15:20:00 | 1497.00 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-27 11:15:00 | 1479.00 | 2026-02-27 11:50:00 | 1474.72 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-27 11:15:00 | 1479.00 | 2026-02-27 13:55:00 | 1479.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:00:00 | 1428.60 | 2026-03-05 10:05:00 | 1438.59 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-05 10:00:00 | 1428.60 | 2026-03-05 10:10:00 | 1428.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:30:00 | 1489.30 | 2026-04-16 09:45:00 | 1479.23 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-16 09:30:00 | 1489.30 | 2026-04-16 11:00:00 | 1489.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:55:00 | 1596.40 | 2026-04-23 10:45:00 | 1590.79 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-06 09:45:00 | 1749.30 | 2026-05-06 10:15:00 | 1742.47 | STOP_HIT | 1.00 | -0.39% |
