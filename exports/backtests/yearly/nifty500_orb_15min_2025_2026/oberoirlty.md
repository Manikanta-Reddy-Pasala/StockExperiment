# Oberoi Realty Ltd. (OBEROIRLTY)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1710.00
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
| ENTRY1 | 86 |
| ENTRY2 | 0 |
| PARTIAL | 42 |
| TARGET_HIT | 15 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 71
- **Target hits / Stop hits / Partials:** 15 / 71 / 42
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 21.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 25 | 43.1% | 7 | 33 | 18 | 0.12% | 7.0% |
| BUY @ 2nd Alert (retest1) | 58 | 25 | 43.1% | 7 | 33 | 18 | 0.12% | 7.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 70 | 32 | 45.7% | 8 | 38 | 24 | 0.20% | 14.2% |
| SELL @ 2nd Alert (retest1) | 70 | 32 | 45.7% | 8 | 38 | 24 | 0.20% | 14.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 128 | 57 | 44.5% | 15 | 71 | 42 | 0.17% | 21.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:55:00 | 1648.10 | 1640.64 | 0.00 | ORB-long ORB[1626.30,1642.90] vol=1.6x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 10:00:00 | 1655.99 | 1645.45 | 0.00 | T1 1.5R @ 1655.99 |
| Target hit | 2025-05-16 15:20:00 | 1675.80 | 1661.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 1698.70 | 1688.13 | 0.00 | ORB-long ORB[1670.80,1693.30] vol=1.6x ATR=5.64 |
| Stop hit — per-position SL triggered | 2025-05-19 09:35:00 | 1693.06 | 1689.09 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 1754.40 | 1747.72 | 0.00 | ORB-long ORB[1740.00,1752.70] vol=2.5x ATR=5.29 |
| Stop hit — per-position SL triggered | 2025-05-26 09:50:00 | 1749.11 | 1752.81 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:00:00 | 1760.50 | 1747.83 | 0.00 | ORB-long ORB[1727.90,1750.60] vol=4.3x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1755.95 | 1748.78 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 09:55:00 | 1746.10 | 1750.09 | 0.00 | ORB-short ORB[1746.80,1760.80] vol=1.8x ATR=4.57 |
| Stop hit — per-position SL triggered | 2025-05-28 10:00:00 | 1750.67 | 1750.43 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 1793.00 | 1786.46 | 0.00 | ORB-long ORB[1772.40,1792.00] vol=2.3x ATR=5.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 09:55:00 | 1801.41 | 1791.46 | 0.00 | T1 1.5R @ 1801.41 |
| Stop hit — per-position SL triggered | 2025-06-03 10:20:00 | 1793.00 | 1794.60 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 1787.40 | 1812.57 | 0.00 | ORB-short ORB[1800.00,1826.00] vol=2.2x ATR=8.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:35:00 | 1774.04 | 1807.55 | 0.00 | T1 1.5R @ 1774.04 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1787.40 | 1790.56 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:45:00 | 1796.70 | 1785.72 | 0.00 | ORB-long ORB[1776.20,1794.70] vol=1.8x ATR=5.63 |
| Stop hit — per-position SL triggered | 2025-06-05 10:55:00 | 1791.07 | 1787.36 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:00:00 | 1929.40 | 1913.22 | 0.00 | ORB-long ORB[1901.20,1919.80] vol=2.0x ATR=6.43 |
| Stop hit — per-position SL triggered | 2025-06-11 10:10:00 | 1922.97 | 1915.37 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 1890.30 | 1897.94 | 0.00 | ORB-short ORB[1891.70,1905.10] vol=3.0x ATR=5.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:45:00 | 1881.45 | 1894.85 | 0.00 | T1 1.5R @ 1881.45 |
| Stop hit — per-position SL triggered | 2025-06-16 09:50:00 | 1890.30 | 1894.04 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 1970.90 | 1957.74 | 0.00 | ORB-long ORB[1938.60,1964.60] vol=3.1x ATR=6.64 |
| Stop hit — per-position SL triggered | 2025-06-17 09:35:00 | 1964.26 | 1958.85 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:45:00 | 1907.00 | 1899.54 | 0.00 | ORB-long ORB[1882.00,1902.90] vol=2.1x ATR=6.56 |
| Stop hit — per-position SL triggered | 2025-06-20 09:50:00 | 1900.44 | 1899.60 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:00:00 | 1956.30 | 1946.14 | 0.00 | ORB-long ORB[1933.50,1955.00] vol=3.2x ATR=6.62 |
| Stop hit — per-position SL triggered | 2025-06-24 10:10:00 | 1949.68 | 1947.86 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:55:00 | 1899.40 | 1910.14 | 0.00 | ORB-short ORB[1901.20,1919.80] vol=1.6x ATR=5.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 12:40:00 | 1890.85 | 1902.76 | 0.00 | T1 1.5R @ 1890.85 |
| Target hit | 2025-07-02 14:55:00 | 1896.20 | 1895.94 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2025-07-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 09:45:00 | 1899.00 | 1906.73 | 0.00 | ORB-short ORB[1900.90,1914.70] vol=1.5x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:00:00 | 1892.15 | 1899.72 | 0.00 | T1 1.5R @ 1892.15 |
| Target hit | 2025-07-03 15:20:00 | 1871.10 | 1889.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-07-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 11:00:00 | 1865.00 | 1875.90 | 0.00 | ORB-short ORB[1867.20,1882.40] vol=1.9x ATR=4.69 |
| Stop hit — per-position SL triggered | 2025-07-04 11:05:00 | 1869.69 | 1875.41 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:30:00 | 1823.90 | 1831.64 | 0.00 | ORB-short ORB[1825.20,1845.00] vol=2.6x ATR=4.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:00:00 | 1816.80 | 1827.20 | 0.00 | T1 1.5R @ 1816.80 |
| Stop hit — per-position SL triggered | 2025-07-08 10:35:00 | 1823.90 | 1824.49 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 10:00:00 | 1843.70 | 1842.44 | 0.00 | ORB-long ORB[1822.80,1842.00] vol=2.0x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 10:10:00 | 1850.81 | 1844.00 | 0.00 | T1 1.5R @ 1850.81 |
| Target hit | 2025-07-10 10:45:00 | 1844.90 | 1847.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2025-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:45:00 | 1830.30 | 1838.91 | 0.00 | ORB-short ORB[1833.80,1854.00] vol=2.1x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:05:00 | 1825.36 | 1836.72 | 0.00 | T1 1.5R @ 1825.36 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 1830.30 | 1836.64 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 10:20:00 | 1820.40 | 1830.27 | 0.00 | ORB-short ORB[1825.00,1846.30] vol=3.0x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 10:30:00 | 1813.35 | 1829.46 | 0.00 | T1 1.5R @ 1813.35 |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 1820.40 | 1819.33 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1837.80 | 1846.69 | 0.00 | ORB-short ORB[1839.00,1857.00] vol=1.8x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:30:00 | 1829.64 | 1844.87 | 0.00 | T1 1.5R @ 1829.64 |
| Stop hit — per-position SL triggered | 2025-07-18 10:35:00 | 1837.80 | 1844.49 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:55:00 | 1744.00 | 1761.29 | 0.00 | ORB-short ORB[1765.20,1787.00] vol=3.2x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:05:00 | 1737.06 | 1758.01 | 0.00 | T1 1.5R @ 1737.06 |
| Stop hit — per-position SL triggered | 2025-07-24 14:10:00 | 1744.00 | 1747.38 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:10:00 | 1726.00 | 1733.63 | 0.00 | ORB-short ORB[1727.40,1744.80] vol=2.1x ATR=5.15 |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 1731.15 | 1733.30 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:55:00 | 1609.80 | 1619.92 | 0.00 | ORB-short ORB[1617.10,1637.00] vol=2.4x ATR=5.65 |
| Stop hit — per-position SL triggered | 2025-08-01 10:00:00 | 1615.45 | 1618.76 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:00:00 | 1585.50 | 1603.15 | 0.00 | ORB-short ORB[1606.90,1617.90] vol=1.7x ATR=4.77 |
| Stop hit — per-position SL triggered | 2025-08-06 11:25:00 | 1590.27 | 1600.73 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:25:00 | 1593.60 | 1598.56 | 0.00 | ORB-short ORB[1600.20,1616.70] vol=1.7x ATR=6.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 10:35:00 | 1583.99 | 1596.92 | 0.00 | T1 1.5R @ 1583.99 |
| Stop hit — per-position SL triggered | 2025-08-08 11:10:00 | 1593.60 | 1595.16 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:00:00 | 1591.50 | 1582.00 | 0.00 | ORB-long ORB[1564.30,1580.50] vol=1.8x ATR=3.97 |
| Stop hit — per-position SL triggered | 2025-08-11 11:45:00 | 1587.53 | 1583.36 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:45:00 | 1648.00 | 1637.64 | 0.00 | ORB-long ORB[1624.90,1638.40] vol=1.6x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:00:00 | 1654.89 | 1641.69 | 0.00 | T1 1.5R @ 1654.89 |
| Stop hit — per-position SL triggered | 2025-08-18 10:45:00 | 1648.00 | 1645.05 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:30:00 | 1676.50 | 1670.08 | 0.00 | ORB-long ORB[1655.00,1671.60] vol=1.7x ATR=4.66 |
| Stop hit — per-position SL triggered | 2025-08-21 09:55:00 | 1671.84 | 1673.71 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 1646.10 | 1656.98 | 0.00 | ORB-short ORB[1654.20,1669.40] vol=1.7x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 09:35:00 | 1640.20 | 1652.76 | 0.00 | T1 1.5R @ 1640.20 |
| Stop hit — per-position SL triggered | 2025-08-22 10:25:00 | 1646.10 | 1645.19 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 1645.10 | 1653.57 | 0.00 | ORB-short ORB[1653.80,1674.40] vol=1.6x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:45:00 | 1639.40 | 1652.12 | 0.00 | T1 1.5R @ 1639.40 |
| Stop hit — per-position SL triggered | 2025-08-26 09:55:00 | 1645.10 | 1651.65 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 10:00:00 | 1610.20 | 1619.58 | 0.00 | ORB-short ORB[1618.00,1633.00] vol=3.6x ATR=4.69 |
| Stop hit — per-position SL triggered | 2025-08-29 10:20:00 | 1614.89 | 1619.13 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:50:00 | 1627.50 | 1620.62 | 0.00 | ORB-long ORB[1604.60,1617.00] vol=1.6x ATR=4.73 |
| Stop hit — per-position SL triggered | 2025-09-01 11:55:00 | 1622.77 | 1623.97 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:45:00 | 1666.70 | 1655.59 | 0.00 | ORB-long ORB[1638.00,1646.30] vol=4.7x ATR=4.07 |
| Stop hit — per-position SL triggered | 2025-09-02 12:20:00 | 1662.63 | 1661.14 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 10:55:00 | 1633.10 | 1639.03 | 0.00 | ORB-short ORB[1635.10,1646.00] vol=8.1x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-09-08 11:10:00 | 1637.29 | 1638.86 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 11:10:00 | 1605.30 | 1614.28 | 0.00 | ORB-short ORB[1610.20,1626.90] vol=1.9x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-09-11 11:25:00 | 1608.96 | 1609.94 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:35:00 | 1671.00 | 1666.39 | 0.00 | ORB-long ORB[1653.90,1668.10] vol=1.9x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 10:10:00 | 1677.06 | 1669.76 | 0.00 | T1 1.5R @ 1677.06 |
| Stop hit — per-position SL triggered | 2025-09-18 11:45:00 | 1671.00 | 1672.27 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:50:00 | 1673.80 | 1667.31 | 0.00 | ORB-long ORB[1647.30,1663.60] vol=1.6x ATR=5.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 10:50:00 | 1682.02 | 1671.48 | 0.00 | T1 1.5R @ 1682.02 |
| Target hit | 2025-09-22 14:00:00 | 1685.40 | 1685.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2025-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:35:00 | 1652.00 | 1652.35 | 0.00 | ORB-short ORB[1653.70,1669.70] vol=10.6x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:40:00 | 1645.85 | 1652.11 | 0.00 | T1 1.5R @ 1645.85 |
| Target hit | 2025-09-24 15:20:00 | 1618.00 | 1626.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:15:00 | 1585.50 | 1587.03 | 0.00 | ORB-short ORB[1585.60,1606.70] vol=2.9x ATR=4.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 10:35:00 | 1578.89 | 1586.39 | 0.00 | T1 1.5R @ 1578.89 |
| Stop hit — per-position SL triggered | 2025-09-30 10:50:00 | 1585.50 | 1586.11 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:35:00 | 1641.10 | 1633.69 | 0.00 | ORB-long ORB[1622.60,1636.30] vol=2.7x ATR=4.04 |
| Stop hit — per-position SL triggered | 2025-10-07 09:40:00 | 1637.06 | 1634.21 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:15:00 | 1610.30 | 1620.31 | 0.00 | ORB-short ORB[1617.30,1633.80] vol=2.0x ATR=5.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:20:00 | 1601.98 | 1616.52 | 0.00 | T1 1.5R @ 1601.98 |
| Stop hit — per-position SL triggered | 2025-10-08 10:25:00 | 1610.30 | 1616.19 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:00:00 | 1609.00 | 1607.50 | 0.00 | ORB-long ORB[1585.00,1608.50] vol=4.6x ATR=5.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 12:00:00 | 1617.40 | 1608.32 | 0.00 | T1 1.5R @ 1617.40 |
| Stop hit — per-position SL triggered | 2025-10-10 14:10:00 | 1609.00 | 1609.01 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 1580.00 | 1586.24 | 0.00 | ORB-short ORB[1586.10,1604.00] vol=3.5x ATR=5.47 |
| Stop hit — per-position SL triggered | 2025-10-13 09:35:00 | 1585.47 | 1586.05 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 1572.10 | 1583.10 | 0.00 | ORB-short ORB[1586.00,1599.00] vol=1.8x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 12:00:00 | 1567.23 | 1580.00 | 0.00 | T1 1.5R @ 1567.23 |
| Stop hit — per-position SL triggered | 2025-10-14 14:30:00 | 1572.10 | 1573.84 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:15:00 | 1606.00 | 1590.78 | 0.00 | ORB-long ORB[1567.70,1587.00] vol=2.1x ATR=4.90 |
| Stop hit — per-position SL triggered | 2025-10-15 10:25:00 | 1601.10 | 1594.38 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:10:00 | 1707.00 | 1694.61 | 0.00 | ORB-long ORB[1681.00,1698.70] vol=2.7x ATR=6.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:25:00 | 1716.94 | 1701.50 | 0.00 | T1 1.5R @ 1716.94 |
| Stop hit — per-position SL triggered | 2025-10-17 11:50:00 | 1707.00 | 1707.25 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 11:10:00 | 1690.20 | 1695.10 | 0.00 | ORB-short ORB[1690.80,1711.90] vol=2.6x ATR=4.83 |
| Stop hit — per-position SL triggered | 2025-10-20 11:35:00 | 1695.03 | 1694.86 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 1699.30 | 1691.69 | 0.00 | ORB-long ORB[1678.10,1694.70] vol=1.6x ATR=5.33 |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 1693.97 | 1696.61 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 10:45:00 | 1752.20 | 1734.07 | 0.00 | ORB-long ORB[1721.50,1738.00] vol=1.7x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:05:00 | 1758.94 | 1738.95 | 0.00 | T1 1.5R @ 1758.94 |
| Stop hit — per-position SL triggered | 2025-10-30 13:10:00 | 1752.20 | 1744.63 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 1783.90 | 1774.20 | 0.00 | ORB-long ORB[1759.00,1782.40] vol=2.0x ATR=5.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:00:00 | 1792.46 | 1780.04 | 0.00 | T1 1.5R @ 1792.46 |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 1783.90 | 1781.21 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:20:00 | 1772.20 | 1779.15 | 0.00 | ORB-short ORB[1774.70,1796.90] vol=2.4x ATR=5.37 |
| Stop hit — per-position SL triggered | 2025-11-06 10:35:00 | 1777.57 | 1778.34 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:35:00 | 1742.50 | 1757.22 | 0.00 | ORB-short ORB[1766.00,1783.20] vol=1.7x ATR=5.17 |
| Stop hit — per-position SL triggered | 2025-11-12 10:40:00 | 1747.67 | 1756.43 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:35:00 | 1753.90 | 1747.62 | 0.00 | ORB-long ORB[1735.20,1751.70] vol=1.5x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:45:00 | 1760.98 | 1750.51 | 0.00 | T1 1.5R @ 1760.98 |
| Stop hit — per-position SL triggered | 2025-11-13 10:05:00 | 1753.90 | 1752.48 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-11-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:25:00 | 1736.10 | 1750.77 | 0.00 | ORB-short ORB[1741.70,1758.50] vol=1.5x ATR=5.05 |
| Stop hit — per-position SL triggered | 2025-11-14 10:50:00 | 1741.15 | 1747.49 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:45:00 | 1718.00 | 1724.83 | 0.00 | ORB-short ORB[1728.60,1753.00] vol=3.4x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:00:00 | 1711.06 | 1722.80 | 0.00 | T1 1.5R @ 1711.06 |
| Stop hit — per-position SL triggered | 2025-11-18 10:05:00 | 1718.00 | 1722.63 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:10:00 | 1680.00 | 1690.15 | 0.00 | ORB-short ORB[1692.00,1715.00] vol=1.8x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 11:20:00 | 1674.26 | 1686.31 | 0.00 | T1 1.5R @ 1674.26 |
| Target hit | 2025-11-21 15:20:00 | 1655.60 | 1672.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2025-11-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:25:00 | 1663.30 | 1658.48 | 0.00 | ORB-long ORB[1648.80,1662.90] vol=2.2x ATR=5.21 |
| Stop hit — per-position SL triggered | 2025-11-24 10:30:00 | 1658.09 | 1658.51 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:00:00 | 1639.60 | 1647.01 | 0.00 | ORB-short ORB[1646.30,1656.20] vol=2.3x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-12-01 11:20:00 | 1642.41 | 1646.72 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:55:00 | 1653.10 | 1644.53 | 0.00 | ORB-long ORB[1630.60,1646.90] vol=2.5x ATR=4.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 11:50:00 | 1659.40 | 1648.86 | 0.00 | T1 1.5R @ 1659.40 |
| Target hit | 2025-12-04 15:20:00 | 1664.80 | 1658.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-12-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:45:00 | 1643.60 | 1652.69 | 0.00 | ORB-short ORB[1652.80,1671.00] vol=1.6x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:15:00 | 1635.53 | 1646.76 | 0.00 | T1 1.5R @ 1635.53 |
| Target hit | 2025-12-08 15:20:00 | 1610.80 | 1624.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-12-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:05:00 | 1629.60 | 1622.48 | 0.00 | ORB-long ORB[1611.00,1626.00] vol=2.3x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:25:00 | 1638.97 | 1624.96 | 0.00 | T1 1.5R @ 1638.97 |
| Target hit | 2025-12-09 12:35:00 | 1632.50 | 1632.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — BUY (started 2025-12-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:00:00 | 1640.80 | 1633.93 | 0.00 | ORB-long ORB[1613.10,1629.40] vol=2.0x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:20:00 | 1647.35 | 1635.16 | 0.00 | T1 1.5R @ 1647.35 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1640.80 | 1640.98 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:00:00 | 1641.40 | 1645.36 | 0.00 | ORB-short ORB[1644.50,1656.40] vol=3.9x ATR=3.10 |
| Stop hit — per-position SL triggered | 2025-12-15 11:35:00 | 1644.50 | 1644.61 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:35:00 | 1619.80 | 1617.70 | 0.00 | ORB-long ORB[1607.60,1618.70] vol=12.2x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 10:40:00 | 1626.39 | 1621.60 | 0.00 | T1 1.5R @ 1626.39 |
| Target hit | 2025-12-18 15:20:00 | 1655.70 | 1643.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2025-12-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 09:50:00 | 1663.70 | 1665.44 | 0.00 | ORB-short ORB[1664.00,1676.10] vol=5.7x ATR=4.46 |
| Stop hit — per-position SL triggered | 2025-12-23 09:55:00 | 1668.16 | 1666.24 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:40:00 | 1687.70 | 1682.53 | 0.00 | ORB-long ORB[1663.60,1685.10] vol=1.5x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:25:00 | 1693.32 | 1686.30 | 0.00 | T1 1.5R @ 1693.32 |
| Target hit | 2025-12-26 12:20:00 | 1690.00 | 1690.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 1671.10 | 1658.76 | 0.00 | ORB-long ORB[1640.00,1663.70] vol=2.9x ATR=4.20 |
| Stop hit — per-position SL triggered | 2025-12-31 11:05:00 | 1666.90 | 1659.38 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-01-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:10:00 | 1755.20 | 1741.37 | 0.00 | ORB-long ORB[1726.30,1745.00] vol=1.6x ATR=3.72 |
| Stop hit — per-position SL triggered | 2026-01-05 11:20:00 | 1751.48 | 1742.63 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:00:00 | 1734.30 | 1740.24 | 0.00 | ORB-short ORB[1735.50,1752.20] vol=2.6x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:10:00 | 1728.52 | 1739.82 | 0.00 | T1 1.5R @ 1728.52 |
| Stop hit — per-position SL triggered | 2026-01-06 11:40:00 | 1734.30 | 1738.30 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1655.00 | 1658.55 | 0.00 | ORB-short ORB[1659.20,1681.80] vol=1.5x ATR=7.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 10:45:00 | 1644.40 | 1656.59 | 0.00 | T1 1.5R @ 1644.40 |
| Target hit | 2026-01-13 14:40:00 | 1647.50 | 1645.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — BUY (started 2026-01-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:30:00 | 1516.70 | 1507.76 | 0.00 | ORB-long ORB[1495.10,1515.10] vol=1.8x ATR=6.15 |
| Stop hit — per-position SL triggered | 2026-01-22 09:45:00 | 1510.55 | 1511.22 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 1534.60 | 1541.42 | 0.00 | ORB-short ORB[1545.10,1557.40] vol=2.9x ATR=3.45 |
| Stop hit — per-position SL triggered | 2026-02-18 11:00:00 | 1538.05 | 1539.33 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 1525.00 | 1528.54 | 0.00 | ORB-short ORB[1528.00,1540.00] vol=4.6x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 1518.86 | 1527.12 | 0.00 | T1 1.5R @ 1518.86 |
| Target hit | 2026-02-19 15:20:00 | 1504.00 | 1517.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 1510.70 | 1517.56 | 0.00 | ORB-short ORB[1519.30,1532.10] vol=2.0x ATR=4.55 |
| Stop hit — per-position SL triggered | 2026-02-23 11:40:00 | 1515.25 | 1516.20 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1524.40 | 1515.36 | 0.00 | ORB-long ORB[1510.00,1519.00] vol=1.7x ATR=4.05 |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 1520.35 | 1517.82 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 1527.20 | 1529.14 | 0.00 | ORB-short ORB[1528.90,1542.50] vol=2.7x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:50:00 | 1520.87 | 1528.03 | 0.00 | T1 1.5R @ 1520.87 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 1527.20 | 1527.67 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 1498.50 | 1505.85 | 0.00 | ORB-short ORB[1500.50,1512.70] vol=2.6x ATR=4.28 |
| Stop hit — per-position SL triggered | 2026-03-11 10:30:00 | 1502.78 | 1505.19 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 1448.90 | 1453.08 | 0.00 | ORB-short ORB[1458.10,1470.40] vol=6.7x ATR=5.52 |
| Stop hit — per-position SL triggered | 2026-03-13 11:20:00 | 1454.42 | 1452.60 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 1458.00 | 1443.81 | 0.00 | ORB-long ORB[1427.30,1444.20] vol=1.8x ATR=7.47 |
| Stop hit — per-position SL triggered | 2026-03-17 09:40:00 | 1450.53 | 1444.99 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 1477.70 | 1470.14 | 0.00 | ORB-long ORB[1449.00,1466.80] vol=3.0x ATR=6.12 |
| Stop hit — per-position SL triggered | 2026-03-20 09:45:00 | 1471.58 | 1470.40 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:50:00 | 1688.70 | 1671.53 | 0.00 | ORB-long ORB[1652.50,1676.00] vol=1.6x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:05:00 | 1698.28 | 1673.95 | 0.00 | T1 1.5R @ 1698.28 |
| Stop hit — per-position SL triggered | 2026-04-10 13:25:00 | 1688.70 | 1687.55 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:05:00 | 1699.00 | 1704.45 | 0.00 | ORB-short ORB[1702.10,1720.00] vol=2.0x ATR=6.97 |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 1705.97 | 1701.51 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 1744.00 | 1731.93 | 0.00 | ORB-long ORB[1712.40,1737.30] vol=2.3x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:50:00 | 1753.22 | 1735.16 | 0.00 | T1 1.5R @ 1753.22 |
| Stop hit — per-position SL triggered | 2026-04-21 10:55:00 | 1744.00 | 1735.41 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 1712.30 | 1722.20 | 0.00 | ORB-short ORB[1713.20,1736.90] vol=1.8x ATR=4.59 |
| Stop hit — per-position SL triggered | 2026-04-23 11:25:00 | 1716.89 | 1722.04 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:00:00 | 1709.80 | 1716.12 | 0.00 | ORB-short ORB[1711.80,1735.50] vol=2.6x ATR=6.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:20:00 | 1700.25 | 1714.79 | 0.00 | T1 1.5R @ 1700.25 |
| Target hit | 2026-04-24 15:20:00 | 1691.20 | 1697.67 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 09:55:00 | 1648.10 | 2025-05-16 10:00:00 | 1655.99 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-16 09:55:00 | 1648.10 | 2025-05-16 15:20:00 | 1675.80 | TARGET_HIT | 0.50 | 1.68% |
| BUY | retest1 | 2025-05-19 09:30:00 | 1698.70 | 2025-05-19 09:35:00 | 1693.06 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-26 09:30:00 | 1754.40 | 2025-05-26 09:50:00 | 1749.11 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-27 11:00:00 | 1760.50 | 2025-05-27 11:15:00 | 1755.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-05-28 09:55:00 | 1746.10 | 2025-05-28 10:00:00 | 1750.67 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-03 09:30:00 | 1793.00 | 2025-06-03 09:55:00 | 1801.41 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-03 09:30:00 | 1793.00 | 2025-06-03 10:20:00 | 1793.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 09:30:00 | 1787.40 | 2025-06-04 09:35:00 | 1774.04 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2025-06-04 09:30:00 | 1787.40 | 2025-06-04 12:15:00 | 1787.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 10:45:00 | 1796.70 | 2025-06-05 10:55:00 | 1791.07 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-11 10:00:00 | 1929.40 | 2025-06-11 10:10:00 | 1922.97 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-06-16 09:30:00 | 1890.30 | 2025-06-16 09:45:00 | 1881.45 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-06-16 09:30:00 | 1890.30 | 2025-06-16 09:50:00 | 1890.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-17 09:30:00 | 1970.90 | 2025-06-17 09:35:00 | 1964.26 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-20 09:45:00 | 1907.00 | 2025-06-20 09:50:00 | 1900.44 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-24 10:00:00 | 1956.30 | 2025-06-24 10:10:00 | 1949.68 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-07-02 09:55:00 | 1899.40 | 2025-07-02 12:40:00 | 1890.85 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-07-02 09:55:00 | 1899.40 | 2025-07-02 14:55:00 | 1896.20 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-07-03 09:45:00 | 1899.00 | 2025-07-03 10:00:00 | 1892.15 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-07-03 09:45:00 | 1899.00 | 2025-07-03 15:20:00 | 1871.10 | TARGET_HIT | 0.50 | 1.47% |
| SELL | retest1 | 2025-07-04 11:00:00 | 1865.00 | 2025-07-04 11:05:00 | 1869.69 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-08 09:30:00 | 1823.90 | 2025-07-08 10:00:00 | 1816.80 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-08 09:30:00 | 1823.90 | 2025-07-08 10:35:00 | 1823.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-10 10:00:00 | 1843.70 | 2025-07-10 10:10:00 | 1850.81 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-10 10:00:00 | 1843.70 | 2025-07-10 10:45:00 | 1844.90 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2025-07-11 10:45:00 | 1830.30 | 2025-07-11 11:05:00 | 1825.36 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-07-11 10:45:00 | 1830.30 | 2025-07-11 11:10:00 | 1830.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-14 10:20:00 | 1820.40 | 2025-07-14 10:30:00 | 1813.35 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-14 10:20:00 | 1820.40 | 2025-07-14 12:15:00 | 1820.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:15:00 | 1837.80 | 2025-07-18 10:30:00 | 1829.64 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-07-18 10:15:00 | 1837.80 | 2025-07-18 10:35:00 | 1837.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 10:55:00 | 1744.00 | 2025-07-24 11:05:00 | 1737.06 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-24 10:55:00 | 1744.00 | 2025-07-24 14:10:00 | 1744.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 10:10:00 | 1726.00 | 2025-07-25 10:15:00 | 1731.15 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-01 09:55:00 | 1609.80 | 2025-08-01 10:00:00 | 1615.45 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-06 11:00:00 | 1585.50 | 2025-08-06 11:25:00 | 1590.27 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-08 10:25:00 | 1593.60 | 2025-08-08 10:35:00 | 1583.99 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-08-08 10:25:00 | 1593.60 | 2025-08-08 11:10:00 | 1593.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-11 11:00:00 | 1591.50 | 2025-08-11 11:45:00 | 1587.53 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-18 09:45:00 | 1648.00 | 2025-08-18 10:00:00 | 1654.89 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-08-18 09:45:00 | 1648.00 | 2025-08-18 10:45:00 | 1648.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-21 09:30:00 | 1676.50 | 2025-08-21 09:55:00 | 1671.84 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-22 09:30:00 | 1646.10 | 2025-08-22 09:35:00 | 1640.20 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-22 09:30:00 | 1646.10 | 2025-08-22 10:25:00 | 1646.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 09:35:00 | 1645.10 | 2025-08-26 09:45:00 | 1639.40 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-08-26 09:35:00 | 1645.10 | 2025-08-26 09:55:00 | 1645.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-29 10:00:00 | 1610.20 | 2025-08-29 10:20:00 | 1614.89 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-01 10:50:00 | 1627.50 | 2025-09-01 11:55:00 | 1622.77 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-02 10:45:00 | 1666.70 | 2025-09-02 12:20:00 | 1662.63 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-08 10:55:00 | 1633.10 | 2025-09-08 11:10:00 | 1637.29 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-11 11:10:00 | 1605.30 | 2025-09-11 11:25:00 | 1608.96 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-18 09:35:00 | 1671.00 | 2025-09-18 10:10:00 | 1677.06 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-18 09:35:00 | 1671.00 | 2025-09-18 11:45:00 | 1671.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 09:50:00 | 1673.80 | 2025-09-22 10:50:00 | 1682.02 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-09-22 09:50:00 | 1673.80 | 2025-09-22 14:00:00 | 1685.40 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-09-24 09:35:00 | 1652.00 | 2025-09-24 09:40:00 | 1645.85 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-24 09:35:00 | 1652.00 | 2025-09-24 15:20:00 | 1618.00 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2025-09-30 10:15:00 | 1585.50 | 2025-09-30 10:35:00 | 1578.89 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-09-30 10:15:00 | 1585.50 | 2025-09-30 10:50:00 | 1585.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-07 09:35:00 | 1641.10 | 2025-10-07 09:40:00 | 1637.06 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-08 10:15:00 | 1610.30 | 2025-10-08 10:20:00 | 1601.98 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-10-08 10:15:00 | 1610.30 | 2025-10-08 10:25:00 | 1610.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 10:00:00 | 1609.00 | 2025-10-10 12:00:00 | 1617.40 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-10-10 10:00:00 | 1609.00 | 2025-10-10 14:10:00 | 1609.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-13 09:30:00 | 1580.00 | 2025-10-13 09:35:00 | 1585.47 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-10-14 11:15:00 | 1572.10 | 2025-10-14 12:00:00 | 1567.23 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-14 11:15:00 | 1572.10 | 2025-10-14 14:30:00 | 1572.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 10:15:00 | 1606.00 | 2025-10-15 10:25:00 | 1601.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-17 10:10:00 | 1707.00 | 2025-10-17 10:25:00 | 1716.94 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-10-17 10:10:00 | 1707.00 | 2025-10-17 11:50:00 | 1707.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-20 11:10:00 | 1690.20 | 2025-10-20 11:35:00 | 1695.03 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-24 09:30:00 | 1699.30 | 2025-10-24 10:15:00 | 1693.97 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-30 10:45:00 | 1752.20 | 2025-10-30 11:05:00 | 1758.94 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-30 10:45:00 | 1752.20 | 2025-10-30 13:10:00 | 1752.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-31 09:30:00 | 1783.90 | 2025-10-31 10:00:00 | 1792.46 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-31 09:30:00 | 1783.90 | 2025-10-31 10:15:00 | 1783.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 10:20:00 | 1772.20 | 2025-11-06 10:35:00 | 1777.57 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-12 10:35:00 | 1742.50 | 2025-11-12 10:40:00 | 1747.67 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-13 09:35:00 | 1753.90 | 2025-11-13 09:45:00 | 1760.98 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-11-13 09:35:00 | 1753.90 | 2025-11-13 10:05:00 | 1753.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-14 10:25:00 | 1736.10 | 2025-11-14 10:50:00 | 1741.15 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-18 09:45:00 | 1718.00 | 2025-11-18 10:00:00 | 1711.06 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-18 09:45:00 | 1718.00 | 2025-11-18 10:05:00 | 1718.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 11:10:00 | 1680.00 | 2025-11-21 11:20:00 | 1674.26 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-21 11:10:00 | 1680.00 | 2025-11-21 15:20:00 | 1655.60 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2025-11-24 10:25:00 | 1663.30 | 2025-11-24 10:30:00 | 1658.09 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-01 11:00:00 | 1639.60 | 2025-12-01 11:20:00 | 1642.41 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-04 10:55:00 | 1653.10 | 2025-12-04 11:50:00 | 1659.40 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-12-04 10:55:00 | 1653.10 | 2025-12-04 15:20:00 | 1664.80 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2025-12-08 10:45:00 | 1643.60 | 2025-12-08 11:15:00 | 1635.53 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-12-08 10:45:00 | 1643.60 | 2025-12-08 15:20:00 | 1610.80 | TARGET_HIT | 0.50 | 2.00% |
| BUY | retest1 | 2025-12-09 10:05:00 | 1629.60 | 2025-12-09 10:25:00 | 1638.97 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-12-09 10:05:00 | 1629.60 | 2025-12-09 12:35:00 | 1632.50 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2025-12-11 11:00:00 | 1640.80 | 2025-12-11 11:20:00 | 1647.35 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-12-11 11:00:00 | 1640.80 | 2025-12-11 13:15:00 | 1640.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-15 11:00:00 | 1641.40 | 2025-12-15 11:35:00 | 1644.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-18 10:35:00 | 1619.80 | 2025-12-18 10:40:00 | 1626.39 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-12-18 10:35:00 | 1619.80 | 2025-12-18 15:20:00 | 1655.70 | TARGET_HIT | 0.50 | 2.22% |
| SELL | retest1 | 2025-12-23 09:50:00 | 1663.70 | 2025-12-23 09:55:00 | 1668.16 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-26 10:40:00 | 1687.70 | 2025-12-26 11:25:00 | 1693.32 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-12-26 10:40:00 | 1687.70 | 2025-12-26 12:20:00 | 1690.00 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2025-12-31 11:00:00 | 1671.10 | 2025-12-31 11:05:00 | 1666.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-05 11:10:00 | 1755.20 | 2026-01-05 11:20:00 | 1751.48 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-06 11:00:00 | 1734.30 | 2026-01-06 11:10:00 | 1728.52 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-06 11:00:00 | 1734.30 | 2026-01-06 11:40:00 | 1734.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-13 10:30:00 | 1655.00 | 2026-01-13 10:45:00 | 1644.40 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-01-13 10:30:00 | 1655.00 | 2026-01-13 14:40:00 | 1647.50 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2026-01-22 09:30:00 | 1516.70 | 2026-01-22 09:45:00 | 1510.55 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-18 10:50:00 | 1534.60 | 2026-02-18 11:00:00 | 1538.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1525.00 | 2026-02-19 11:15:00 | 1518.86 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1525.00 | 2026-02-19 15:20:00 | 1504.00 | TARGET_HIT | 0.50 | 1.38% |
| SELL | retest1 | 2026-02-23 10:55:00 | 1510.70 | 2026-02-23 11:40:00 | 1515.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-24 09:45:00 | 1524.40 | 2026-02-24 10:15:00 | 1520.35 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-27 10:40:00 | 1527.20 | 2026-02-27 10:50:00 | 1520.87 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-27 10:40:00 | 1527.20 | 2026-02-27 11:15:00 | 1527.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:25:00 | 1498.50 | 2026-03-11 10:30:00 | 1502.78 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-13 10:50:00 | 1448.90 | 2026-03-13 11:20:00 | 1454.42 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-17 09:35:00 | 1458.00 | 2026-03-17 09:40:00 | 1450.53 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-03-20 09:35:00 | 1477.70 | 2026-03-20 09:45:00 | 1471.58 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-10 10:50:00 | 1688.70 | 2026-04-10 11:05:00 | 1698.28 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-10 10:50:00 | 1688.70 | 2026-04-10 13:25:00 | 1688.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-15 10:05:00 | 1699.00 | 2026-04-15 11:15:00 | 1705.97 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-21 10:40:00 | 1744.00 | 2026-04-21 10:50:00 | 1753.22 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-21 10:40:00 | 1744.00 | 2026-04-21 10:55:00 | 1744.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:15:00 | 1712.30 | 2026-04-23 11:25:00 | 1716.89 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 10:00:00 | 1709.80 | 2026-04-24 10:20:00 | 1700.25 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-24 10:00:00 | 1709.80 | 2026-04-24 15:20:00 | 1691.20 | TARGET_HIT | 0.50 | 1.09% |
