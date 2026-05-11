# Narayana Hrudayalaya Ltd. (NH)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1820.00
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
| ENTRY1 | 77 |
| ENTRY2 | 0 |
| PARTIAL | 40 |
| TARGET_HIT | 20 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 117 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 57
- **Target hits / Stop hits / Partials:** 20 / 57 / 40
- **Avg / median % per leg:** 0.20% / 0.23%
- **Sum % (uncompounded):** 23.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 32 | 52.5% | 11 | 29 | 21 | 0.19% | 11.5% |
| BUY @ 2nd Alert (retest1) | 61 | 32 | 52.5% | 11 | 29 | 21 | 0.19% | 11.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 28 | 50.0% | 9 | 28 | 19 | 0.21% | 12.0% |
| SELL @ 2nd Alert (retest1) | 56 | 28 | 50.0% | 9 | 28 | 19 | 0.21% | 12.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 117 | 60 | 51.3% | 20 | 57 | 40 | 0.20% | 23.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:45:00 | 1801.70 | 1784.79 | 0.00 | ORB-long ORB[1765.10,1791.80] vol=2.4x ATR=7.29 |
| Stop hit — per-position SL triggered | 2025-05-15 09:50:00 | 1794.41 | 1787.07 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 09:50:00 | 1782.70 | 1802.60 | 0.00 | ORB-short ORB[1799.70,1823.30] vol=1.8x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 11:25:00 | 1772.27 | 1790.67 | 0.00 | T1 1.5R @ 1772.27 |
| Stop hit — per-position SL triggered | 2025-05-16 12:30:00 | 1782.70 | 1788.29 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:40:00 | 1745.10 | 1729.69 | 0.00 | ORB-long ORB[1715.60,1730.00] vol=5.2x ATR=5.67 |
| Stop hit — per-position SL triggered | 2025-05-29 10:55:00 | 1739.43 | 1737.94 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:30:00 | 1774.40 | 1765.16 | 0.00 | ORB-long ORB[1756.00,1769.00] vol=1.9x ATR=5.03 |
| Stop hit — per-position SL triggered | 2025-06-04 10:50:00 | 1769.37 | 1766.14 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 09:35:00 | 1818.50 | 1832.19 | 0.00 | ORB-short ORB[1821.60,1845.00] vol=1.8x ATR=6.87 |
| Stop hit — per-position SL triggered | 2025-06-11 10:20:00 | 1825.37 | 1829.52 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 10:10:00 | 1856.90 | 1846.42 | 0.00 | ORB-long ORB[1835.50,1849.00] vol=3.6x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 10:15:00 | 1865.27 | 1850.33 | 0.00 | T1 1.5R @ 1865.27 |
| Stop hit — per-position SL triggered | 2025-06-12 10:20:00 | 1856.90 | 1850.43 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-07-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:45:00 | 1998.10 | 1983.66 | 0.00 | ORB-long ORB[1970.10,1993.00] vol=2.6x ATR=6.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:50:00 | 2008.10 | 1990.34 | 0.00 | T1 1.5R @ 2008.10 |
| Target hit | 2025-07-11 10:05:00 | 2004.20 | 2005.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2025-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 09:30:00 | 2000.90 | 2012.10 | 0.00 | ORB-short ORB[2007.30,2032.00] vol=2.0x ATR=6.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:55:00 | 1991.45 | 2005.20 | 0.00 | T1 1.5R @ 1991.45 |
| Target hit | 2025-07-15 15:20:00 | 1983.40 | 1990.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-07-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:40:00 | 1968.10 | 1975.88 | 0.00 | ORB-short ORB[1972.00,1990.80] vol=2.1x ATR=4.14 |
| Stop hit — per-position SL triggered | 2025-07-16 11:40:00 | 1972.24 | 1973.23 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1948.50 | 1958.30 | 0.00 | ORB-short ORB[1955.00,1969.80] vol=1.6x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:30:00 | 1940.64 | 1955.77 | 0.00 | T1 1.5R @ 1940.64 |
| Stop hit — per-position SL triggered | 2025-07-18 11:00:00 | 1948.50 | 1953.45 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 11:05:00 | 1957.40 | 1944.03 | 0.00 | ORB-long ORB[1927.00,1955.70] vol=2.0x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 11:30:00 | 1965.41 | 1946.08 | 0.00 | T1 1.5R @ 1965.41 |
| Stop hit — per-position SL triggered | 2025-07-21 12:20:00 | 1957.40 | 1949.11 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 10:05:00 | 1963.90 | 1948.68 | 0.00 | ORB-long ORB[1924.40,1949.90] vol=2.0x ATR=8.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:20:00 | 1976.94 | 1955.12 | 0.00 | T1 1.5R @ 1976.94 |
| Stop hit — per-position SL triggered | 2025-07-22 10:30:00 | 1963.90 | 1956.73 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-25 11:00:00 | 1986.50 | 1976.35 | 0.00 | ORB-long ORB[1962.70,1983.90] vol=3.0x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:10:00 | 1994.24 | 1978.46 | 0.00 | T1 1.5R @ 1994.24 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 1986.50 | 1980.21 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:20:00 | 1970.60 | 1976.60 | 0.00 | ORB-short ORB[1976.40,2000.10] vol=4.8x ATR=5.83 |
| Stop hit — per-position SL triggered | 2025-07-29 12:05:00 | 1976.43 | 1975.68 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-08-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:55:00 | 1814.10 | 1833.51 | 0.00 | ORB-short ORB[1832.70,1850.30] vol=1.6x ATR=9.85 |
| Stop hit — per-position SL triggered | 2025-08-05 10:10:00 | 1823.95 | 1831.08 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:55:00 | 1765.00 | 1801.96 | 0.00 | ORB-short ORB[1806.30,1830.00] vol=2.7x ATR=7.10 |
| Stop hit — per-position SL triggered | 2025-08-06 10:00:00 | 1772.10 | 1794.97 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:35:00 | 1796.00 | 1784.30 | 0.00 | ORB-long ORB[1773.80,1793.00] vol=2.6x ATR=7.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 12:10:00 | 1807.13 | 1790.47 | 0.00 | T1 1.5R @ 1807.13 |
| Target hit | 2025-08-12 15:20:00 | 1808.00 | 1804.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-08-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:35:00 | 1829.00 | 1817.08 | 0.00 | ORB-long ORB[1803.10,1822.00] vol=1.7x ATR=6.00 |
| Stop hit — per-position SL triggered | 2025-08-18 09:45:00 | 1823.00 | 1819.87 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 1826.20 | 1832.24 | 0.00 | ORB-short ORB[1829.80,1848.90] vol=2.3x ATR=5.48 |
| Stop hit — per-position SL triggered | 2025-08-22 09:50:00 | 1831.68 | 1831.15 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 1814.60 | 1828.53 | 0.00 | ORB-short ORB[1824.70,1849.00] vol=2.8x ATR=5.62 |
| Stop hit — per-position SL triggered | 2025-08-26 09:55:00 | 1820.22 | 1824.19 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:45:00 | 1769.00 | 1756.72 | 0.00 | ORB-long ORB[1741.20,1758.90] vol=1.5x ATR=5.96 |
| Stop hit — per-position SL triggered | 2025-08-29 11:50:00 | 1763.04 | 1759.38 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:40:00 | 1763.60 | 1756.21 | 0.00 | ORB-long ORB[1740.60,1762.50] vol=1.6x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:40:00 | 1774.02 | 1762.71 | 0.00 | T1 1.5R @ 1774.02 |
| Target hit | 2025-09-01 15:20:00 | 1782.00 | 1775.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-09-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 10:00:00 | 1803.70 | 1811.52 | 0.00 | ORB-short ORB[1809.20,1825.00] vol=1.9x ATR=6.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 10:40:00 | 1793.35 | 1808.71 | 0.00 | T1 1.5R @ 1793.35 |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 1803.70 | 1804.84 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 11:15:00 | 1753.20 | 1761.29 | 0.00 | ORB-short ORB[1762.10,1784.10] vol=1.6x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:40:00 | 1747.37 | 1759.39 | 0.00 | T1 1.5R @ 1747.37 |
| Target hit | 2025-09-11 15:20:00 | 1742.60 | 1745.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-09-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:00:00 | 1776.00 | 1769.56 | 0.00 | ORB-long ORB[1752.10,1774.90] vol=2.2x ATR=5.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 10:10:00 | 1784.00 | 1772.14 | 0.00 | T1 1.5R @ 1784.00 |
| Stop hit — per-position SL triggered | 2025-09-15 11:10:00 | 1776.00 | 1775.44 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:40:00 | 1811.10 | 1799.26 | 0.00 | ORB-long ORB[1783.40,1802.10] vol=2.1x ATR=4.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 09:45:00 | 1818.55 | 1803.11 | 0.00 | T1 1.5R @ 1818.55 |
| Stop hit — per-position SL triggered | 2025-09-17 09:50:00 | 1811.10 | 1804.85 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 10:50:00 | 1799.60 | 1801.88 | 0.00 | ORB-short ORB[1801.10,1814.00] vol=1.7x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:15:00 | 1792.35 | 1801.25 | 0.00 | T1 1.5R @ 1792.35 |
| Target hit | 2025-09-18 15:20:00 | 1790.30 | 1794.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-09-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 10:00:00 | 1779.90 | 1788.45 | 0.00 | ORB-short ORB[1780.50,1798.20] vol=1.5x ATR=5.86 |
| Stop hit — per-position SL triggered | 2025-09-19 10:20:00 | 1785.76 | 1785.77 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:10:00 | 1734.10 | 1741.52 | 0.00 | ORB-short ORB[1736.10,1758.80] vol=2.5x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 11:35:00 | 1727.51 | 1740.33 | 0.00 | T1 1.5R @ 1727.51 |
| Stop hit — per-position SL triggered | 2025-09-23 12:40:00 | 1734.10 | 1738.33 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:40:00 | 1722.90 | 1727.84 | 0.00 | ORB-short ORB[1723.70,1742.50] vol=1.9x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:55:00 | 1716.52 | 1725.66 | 0.00 | T1 1.5R @ 1716.52 |
| Stop hit — per-position SL triggered | 2025-09-24 11:10:00 | 1722.90 | 1722.45 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 09:30:00 | 1717.20 | 1720.97 | 0.00 | ORB-short ORB[1717.80,1730.00] vol=1.9x ATR=3.25 |
| Stop hit — per-position SL triggered | 2025-09-25 09:40:00 | 1720.45 | 1720.93 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:30:00 | 1770.00 | 1774.55 | 0.00 | ORB-short ORB[1777.40,1790.30] vol=2.0x ATR=4.89 |
| Stop hit — per-position SL triggered | 2025-10-08 10:40:00 | 1774.89 | 1774.24 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:15:00 | 1748.80 | 1762.53 | 0.00 | ORB-short ORB[1758.70,1778.00] vol=4.6x ATR=4.49 |
| Stop hit — per-position SL triggered | 2025-10-13 11:55:00 | 1753.29 | 1759.64 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:00:00 | 1759.10 | 1754.10 | 0.00 | ORB-long ORB[1744.20,1758.90] vol=2.1x ATR=4.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:10:00 | 1766.09 | 1755.45 | 0.00 | T1 1.5R @ 1766.09 |
| Stop hit — per-position SL triggered | 2025-10-15 11:30:00 | 1759.10 | 1756.27 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:35:00 | 1767.10 | 1762.10 | 0.00 | ORB-long ORB[1751.70,1765.00] vol=1.6x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 09:40:00 | 1773.21 | 1763.20 | 0.00 | T1 1.5R @ 1773.21 |
| Stop hit — per-position SL triggered | 2025-10-16 09:50:00 | 1767.10 | 1764.17 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 1763.40 | 1768.91 | 0.00 | ORB-short ORB[1767.70,1775.00] vol=2.3x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:05:00 | 1756.08 | 1763.79 | 0.00 | T1 1.5R @ 1756.08 |
| Stop hit — per-position SL triggered | 2025-10-17 11:25:00 | 1763.40 | 1763.63 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:45:00 | 1775.00 | 1765.53 | 0.00 | ORB-long ORB[1756.00,1768.40] vol=1.6x ATR=5.36 |
| Stop hit — per-position SL triggered | 2025-10-20 10:25:00 | 1769.64 | 1770.00 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 10:00:00 | 1762.40 | 1776.76 | 0.00 | ORB-short ORB[1771.00,1786.00] vol=2.5x ATR=5.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:45:00 | 1753.42 | 1767.07 | 0.00 | T1 1.5R @ 1753.42 |
| Stop hit — per-position SL triggered | 2025-10-23 11:45:00 | 1762.40 | 1762.39 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 09:55:00 | 1748.60 | 1754.37 | 0.00 | ORB-short ORB[1753.00,1768.00] vol=1.5x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:00:00 | 1742.83 | 1752.62 | 0.00 | T1 1.5R @ 1742.83 |
| Stop hit — per-position SL triggered | 2025-10-24 10:35:00 | 1748.60 | 1749.48 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:35:00 | 1742.80 | 1736.83 | 0.00 | ORB-long ORB[1728.00,1735.80] vol=1.7x ATR=4.29 |
| Stop hit — per-position SL triggered | 2025-10-27 10:00:00 | 1738.51 | 1739.04 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 10:20:00 | 1776.20 | 1767.56 | 0.00 | ORB-long ORB[1750.10,1768.00] vol=1.8x ATR=5.30 |
| Stop hit — per-position SL triggered | 2025-10-28 12:20:00 | 1770.90 | 1772.46 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:50:00 | 1825.90 | 1806.32 | 0.00 | ORB-long ORB[1783.00,1809.80] vol=3.2x ATR=8.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:15:00 | 1838.73 | 1820.37 | 0.00 | T1 1.5R @ 1838.73 |
| Target hit | 2025-11-04 11:20:00 | 1837.00 | 1839.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — SELL (started 2025-11-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 11:00:00 | 1821.80 | 1824.96 | 0.00 | ORB-short ORB[1825.40,1843.70] vol=1.6x ATR=6.22 |
| Stop hit — per-position SL triggered | 2025-11-06 11:30:00 | 1828.02 | 1824.81 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:10:00 | 1824.10 | 1806.98 | 0.00 | ORB-long ORB[1795.60,1817.90] vol=2.3x ATR=7.23 |
| Stop hit — per-position SL triggered | 2025-11-10 10:45:00 | 1816.87 | 1814.52 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:30:00 | 1758.60 | 1772.74 | 0.00 | ORB-short ORB[1765.60,1787.30] vol=1.5x ATR=7.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 09:35:00 | 1747.97 | 1767.40 | 0.00 | T1 1.5R @ 1747.97 |
| Target hit | 2025-11-12 11:00:00 | 1755.20 | 1750.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — BUY (started 2025-11-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:45:00 | 1777.10 | 1768.51 | 0.00 | ORB-long ORB[1756.90,1768.40] vol=1.7x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:45:00 | 1783.09 | 1778.36 | 0.00 | T1 1.5R @ 1783.09 |
| Target hit | 2025-11-13 12:25:00 | 1781.10 | 1781.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2025-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 09:35:00 | 2026.00 | 2019.71 | 0.00 | ORB-long ORB[1995.00,2025.00] vol=2.7x ATR=10.01 |
| Stop hit — per-position SL triggered | 2025-11-20 09:40:00 | 2015.99 | 2019.83 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-11-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:05:00 | 2022.10 | 2038.33 | 0.00 | ORB-short ORB[2034.00,2061.90] vol=1.5x ATR=9.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:10:00 | 2007.89 | 2035.09 | 0.00 | T1 1.5R @ 2007.89 |
| Target hit | 2025-11-24 15:20:00 | 1975.30 | 2006.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2025-11-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:45:00 | 1980.90 | 1968.71 | 0.00 | ORB-long ORB[1946.30,1974.00] vol=2.2x ATR=6.03 |
| Stop hit — per-position SL triggered | 2025-11-26 11:05:00 | 1974.87 | 1971.13 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:45:00 | 1935.90 | 1954.52 | 0.00 | ORB-short ORB[1955.40,1975.00] vol=2.2x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:10:00 | 1925.48 | 1951.08 | 0.00 | T1 1.5R @ 1925.48 |
| Target hit | 2025-12-01 15:20:00 | 1911.70 | 1931.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2025-12-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 10:45:00 | 1924.90 | 1913.32 | 0.00 | ORB-long ORB[1893.00,1916.00] vol=2.5x ATR=6.35 |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 1918.55 | 1915.84 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 09:35:00 | 1939.00 | 1928.49 | 0.00 | ORB-long ORB[1910.40,1933.50] vol=2.4x ATR=8.92 |
| Stop hit — per-position SL triggered | 2025-12-08 09:55:00 | 1930.08 | 1929.76 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:55:00 | 1891.50 | 1911.97 | 0.00 | ORB-short ORB[1904.60,1929.00] vol=2.1x ATR=7.21 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1898.71 | 1903.40 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:30:00 | 1875.50 | 1864.09 | 0.00 | ORB-long ORB[1845.40,1873.00] vol=2.0x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 11:05:00 | 1885.97 | 1867.56 | 0.00 | T1 1.5R @ 1885.97 |
| Target hit | 2025-12-15 15:20:00 | 1900.90 | 1883.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-12-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 10:45:00 | 1900.80 | 1889.94 | 0.00 | ORB-long ORB[1883.40,1898.40] vol=2.0x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:00:00 | 1909.31 | 1893.47 | 0.00 | T1 1.5R @ 1909.31 |
| Target hit | 2025-12-16 12:30:00 | 1908.30 | 1908.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — SELL (started 2025-12-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:10:00 | 1872.00 | 1881.78 | 0.00 | ORB-short ORB[1875.00,1893.90] vol=1.7x ATR=6.26 |
| Stop hit — per-position SL triggered | 2025-12-17 10:55:00 | 1878.26 | 1877.83 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:00:00 | 1865.00 | 1853.41 | 0.00 | ORB-long ORB[1839.70,1859.40] vol=1.8x ATR=7.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:15:00 | 1876.21 | 1860.02 | 0.00 | T1 1.5R @ 1876.21 |
| Target hit | 2025-12-19 11:15:00 | 1872.30 | 1874.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — SELL (started 2025-12-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:25:00 | 1843.50 | 1854.03 | 0.00 | ORB-short ORB[1851.00,1866.00] vol=2.8x ATR=5.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:30:00 | 1834.91 | 1851.16 | 0.00 | T1 1.5R @ 1834.91 |
| Target hit | 2025-12-30 13:55:00 | 1827.00 | 1822.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — BUY (started 2026-01-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:40:00 | 1907.70 | 1877.38 | 0.00 | ORB-long ORB[1855.00,1880.00] vol=4.2x ATR=9.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 12:30:00 | 1921.77 | 1895.07 | 0.00 | T1 1.5R @ 1921.77 |
| Target hit | 2026-01-01 15:20:00 | 1937.00 | 1906.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2026-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:30:00 | 1929.50 | 1924.14 | 0.00 | ORB-long ORB[1914.00,1929.40] vol=1.6x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:35:00 | 1936.37 | 1927.27 | 0.00 | T1 1.5R @ 1936.37 |
| Stop hit — per-position SL triggered | 2026-01-08 10:20:00 | 1929.50 | 1931.06 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-01-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 11:05:00 | 1819.00 | 1835.11 | 0.00 | ORB-short ORB[1836.00,1862.00] vol=1.8x ATR=6.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:40:00 | 1809.79 | 1831.30 | 0.00 | T1 1.5R @ 1809.79 |
| Stop hit — per-position SL triggered | 2026-01-20 13:20:00 | 1819.00 | 1826.30 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-01-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 10:25:00 | 1796.20 | 1788.68 | 0.00 | ORB-long ORB[1769.40,1792.00] vol=2.4x ATR=6.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:55:00 | 1806.50 | 1790.27 | 0.00 | T1 1.5R @ 1806.50 |
| Stop hit — per-position SL triggered | 2026-01-23 11:40:00 | 1796.20 | 1795.71 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 09:40:00 | 1771.90 | 1747.02 | 0.00 | ORB-long ORB[1724.00,1744.30] vol=1.8x ATR=7.79 |
| Stop hit — per-position SL triggered | 2026-01-28 09:55:00 | 1764.11 | 1753.39 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 1755.90 | 1743.87 | 0.00 | ORB-long ORB[1730.00,1747.00] vol=2.0x ATR=6.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:50:00 | 1765.46 | 1749.93 | 0.00 | T1 1.5R @ 1765.46 |
| Target hit | 2026-01-30 13:15:00 | 1761.00 | 1763.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 65 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 1768.10 | 1756.34 | 0.00 | ORB-long ORB[1750.10,1765.60] vol=3.8x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:30:00 | 1776.12 | 1777.60 | 0.00 | T1 1.5R @ 1776.12 |
| Target hit | 2026-02-01 12:00:00 | 1787.30 | 1787.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — SELL (started 2026-02-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:45:00 | 1731.20 | 1745.13 | 0.00 | ORB-short ORB[1741.00,1764.90] vol=1.6x ATR=5.26 |
| Stop hit — per-position SL triggered | 2026-02-05 14:30:00 | 1736.46 | 1733.71 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1853.20 | 1843.72 | 0.00 | ORB-long ORB[1831.40,1850.00] vol=4.1x ATR=6.32 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 1846.88 | 1845.91 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 1686.60 | 1691.19 | 0.00 | ORB-short ORB[1687.50,1710.00] vol=1.5x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:00:00 | 1678.11 | 1688.45 | 0.00 | T1 1.5R @ 1678.11 |
| Target hit | 2026-03-13 12:45:00 | 1672.10 | 1669.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — BUY (started 2026-03-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 09:30:00 | 1656.30 | 1643.31 | 0.00 | ORB-long ORB[1634.00,1654.00] vol=1.6x ATR=7.71 |
| Stop hit — per-position SL triggered | 2026-03-24 09:40:00 | 1648.59 | 1644.37 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 1663.90 | 1680.37 | 0.00 | ORB-short ORB[1674.00,1698.60] vol=2.1x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:15:00 | 1654.49 | 1674.95 | 0.00 | T1 1.5R @ 1654.49 |
| Target hit | 2026-03-27 14:10:00 | 1641.80 | 1638.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — SELL (started 2026-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:55:00 | 1611.50 | 1629.81 | 0.00 | ORB-short ORB[1629.00,1652.00] vol=3.6x ATR=6.51 |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 1618.01 | 1628.16 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 1812.40 | 1794.07 | 0.00 | ORB-long ORB[1773.90,1800.40] vol=2.8x ATR=8.05 |
| Stop hit — per-position SL triggered | 2026-04-15 09:55:00 | 1804.35 | 1797.20 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-04-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:50:00 | 1819.70 | 1809.58 | 0.00 | ORB-long ORB[1798.50,1817.20] vol=1.5x ATR=4.63 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 1815.07 | 1809.85 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 1793.10 | 1801.62 | 0.00 | ORB-short ORB[1798.20,1815.90] vol=2.2x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 12:25:00 | 1786.33 | 1798.38 | 0.00 | T1 1.5R @ 1786.33 |
| Stop hit — per-position SL triggered | 2026-04-23 12:35:00 | 1793.10 | 1797.39 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-04-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:40:00 | 1780.70 | 1790.22 | 0.00 | ORB-short ORB[1792.10,1810.50] vol=2.0x ATR=5.02 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 1785.72 | 1788.94 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-04-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:00:00 | 1749.90 | 1761.49 | 0.00 | ORB-short ORB[1750.30,1767.00] vol=3.4x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-04-30 11:25:00 | 1755.55 | 1759.69 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-05-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:50:00 | 1806.50 | 1786.66 | 0.00 | ORB-long ORB[1768.00,1793.00] vol=1.7x ATR=8.01 |
| Stop hit — per-position SL triggered | 2026-05-04 10:30:00 | 1798.49 | 1792.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:45:00 | 1801.70 | 2025-05-15 09:50:00 | 1794.41 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-05-16 09:50:00 | 1782.70 | 2025-05-16 11:25:00 | 1772.27 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-05-16 09:50:00 | 1782.70 | 2025-05-16 12:30:00 | 1782.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-29 09:40:00 | 1745.10 | 2025-05-29 10:55:00 | 1739.43 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-04 10:30:00 | 1774.40 | 2025-06-04 10:50:00 | 1769.37 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-11 09:35:00 | 1818.50 | 2025-06-11 10:20:00 | 1825.37 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-12 10:10:00 | 1856.90 | 2025-06-12 10:15:00 | 1865.27 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-06-12 10:10:00 | 1856.90 | 2025-06-12 10:20:00 | 1856.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-11 09:45:00 | 1998.10 | 2025-07-11 09:50:00 | 2008.10 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-07-11 09:45:00 | 1998.10 | 2025-07-11 10:05:00 | 2004.20 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-15 09:30:00 | 2000.90 | 2025-07-15 09:55:00 | 1991.45 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-07-15 09:30:00 | 2000.90 | 2025-07-15 15:20:00 | 1983.40 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2025-07-16 10:40:00 | 1968.10 | 2025-07-16 11:40:00 | 1972.24 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-18 10:15:00 | 1948.50 | 2025-07-18 10:30:00 | 1940.64 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-18 10:15:00 | 1948.50 | 2025-07-18 11:00:00 | 1948.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 11:05:00 | 1957.40 | 2025-07-21 11:30:00 | 1965.41 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-07-21 11:05:00 | 1957.40 | 2025-07-21 12:20:00 | 1957.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-22 10:05:00 | 1963.90 | 2025-07-22 10:20:00 | 1976.94 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-07-22 10:05:00 | 1963.90 | 2025-07-22 10:30:00 | 1963.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-25 11:00:00 | 1986.50 | 2025-07-25 11:10:00 | 1994.24 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-25 11:00:00 | 1986.50 | 2025-07-25 11:15:00 | 1986.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-29 10:20:00 | 1970.60 | 2025-07-29 12:05:00 | 1976.43 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-05 09:55:00 | 1814.10 | 2025-08-05 10:10:00 | 1823.95 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-08-06 09:55:00 | 1765.00 | 2025-08-06 10:00:00 | 1772.10 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-08-12 10:35:00 | 1796.00 | 2025-08-12 12:10:00 | 1807.13 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-08-12 10:35:00 | 1796.00 | 2025-08-12 15:20:00 | 1808.00 | TARGET_HIT | 0.50 | 0.67% |
| BUY | retest1 | 2025-08-18 09:35:00 | 1829.00 | 2025-08-18 09:45:00 | 1823.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-08-22 09:30:00 | 1826.20 | 2025-08-22 09:50:00 | 1831.68 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-26 09:35:00 | 1814.60 | 2025-08-26 09:55:00 | 1820.22 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-29 10:45:00 | 1769.00 | 2025-08-29 11:50:00 | 1763.04 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-01 09:40:00 | 1763.60 | 2025-09-01 11:40:00 | 1774.02 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-09-01 09:40:00 | 1763.60 | 2025-09-01 15:20:00 | 1782.00 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2025-09-04 10:00:00 | 1803.70 | 2025-09-04 10:40:00 | 1793.35 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-09-04 10:00:00 | 1803.70 | 2025-09-04 11:15:00 | 1803.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-11 11:15:00 | 1753.20 | 2025-09-11 11:40:00 | 1747.37 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-09-11 11:15:00 | 1753.20 | 2025-09-11 15:20:00 | 1742.60 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-09-15 10:00:00 | 1776.00 | 2025-09-15 10:10:00 | 1784.00 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-15 10:00:00 | 1776.00 | 2025-09-15 11:10:00 | 1776.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 09:40:00 | 1811.10 | 2025-09-17 09:45:00 | 1818.55 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-17 09:40:00 | 1811.10 | 2025-09-17 09:50:00 | 1811.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-18 10:50:00 | 1799.60 | 2025-09-18 11:15:00 | 1792.35 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-09-18 10:50:00 | 1799.60 | 2025-09-18 15:20:00 | 1790.30 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2025-09-19 10:00:00 | 1779.90 | 2025-09-19 10:20:00 | 1785.76 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-23 11:10:00 | 1734.10 | 2025-09-23 11:35:00 | 1727.51 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-09-23 11:10:00 | 1734.10 | 2025-09-23 12:40:00 | 1734.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 10:40:00 | 1722.90 | 2025-09-24 10:55:00 | 1716.52 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-24 10:40:00 | 1722.90 | 2025-09-24 11:10:00 | 1722.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-25 09:30:00 | 1717.20 | 2025-09-25 09:40:00 | 1720.45 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-08 10:30:00 | 1770.00 | 2025-10-08 10:40:00 | 1774.89 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-13 11:15:00 | 1748.80 | 2025-10-13 11:55:00 | 1753.29 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-15 11:00:00 | 1759.10 | 2025-10-15 11:10:00 | 1766.09 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-15 11:00:00 | 1759.10 | 2025-10-15 11:30:00 | 1759.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-16 09:35:00 | 1767.10 | 2025-10-16 09:40:00 | 1773.21 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-16 09:35:00 | 1767.10 | 2025-10-16 09:50:00 | 1767.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 09:30:00 | 1763.40 | 2025-10-17 11:05:00 | 1756.08 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-17 09:30:00 | 1763.40 | 2025-10-17 11:25:00 | 1763.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 09:45:00 | 1775.00 | 2025-10-20 10:25:00 | 1769.64 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-23 10:00:00 | 1762.40 | 2025-10-23 10:45:00 | 1753.42 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-10-23 10:00:00 | 1762.40 | 2025-10-23 11:45:00 | 1762.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-24 09:55:00 | 1748.60 | 2025-10-24 10:00:00 | 1742.83 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-24 09:55:00 | 1748.60 | 2025-10-24 10:35:00 | 1748.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 09:35:00 | 1742.80 | 2025-10-27 10:00:00 | 1738.51 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-28 10:20:00 | 1776.20 | 2025-10-28 12:20:00 | 1770.90 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-04 09:50:00 | 1825.90 | 2025-11-04 10:15:00 | 1838.73 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-11-04 09:50:00 | 1825.90 | 2025-11-04 11:20:00 | 1837.00 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2025-11-06 11:00:00 | 1821.80 | 2025-11-06 11:30:00 | 1828.02 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-11-10 10:10:00 | 1824.10 | 2025-11-10 10:45:00 | 1816.87 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-11-12 09:30:00 | 1758.60 | 2025-11-12 09:35:00 | 1747.97 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-11-12 09:30:00 | 1758.60 | 2025-11-12 11:00:00 | 1755.20 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-11-13 10:45:00 | 1777.10 | 2025-11-13 11:45:00 | 1783.09 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-11-13 10:45:00 | 1777.10 | 2025-11-13 12:25:00 | 1781.10 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2025-11-20 09:35:00 | 2026.00 | 2025-11-20 09:40:00 | 2015.99 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-11-24 10:05:00 | 2022.10 | 2025-11-24 10:10:00 | 2007.89 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2025-11-24 10:05:00 | 2022.10 | 2025-11-24 15:20:00 | 1975.30 | TARGET_HIT | 0.50 | 2.31% |
| BUY | retest1 | 2025-11-26 10:45:00 | 1980.90 | 2025-11-26 11:05:00 | 1974.87 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-01 10:45:00 | 1935.90 | 2025-12-01 11:10:00 | 1925.48 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-12-01 10:45:00 | 1935.90 | 2025-12-01 15:20:00 | 1911.70 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2025-12-03 10:45:00 | 1924.90 | 2025-12-03 11:15:00 | 1918.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-08 09:35:00 | 1939.00 | 2025-12-08 09:55:00 | 1930.08 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-12-09 10:55:00 | 1891.50 | 2025-12-09 12:15:00 | 1898.71 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-12-15 10:30:00 | 1875.50 | 2025-12-15 11:05:00 | 1885.97 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-12-15 10:30:00 | 1875.50 | 2025-12-15 15:20:00 | 1900.90 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2025-12-16 10:45:00 | 1900.80 | 2025-12-16 11:00:00 | 1909.31 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-16 10:45:00 | 1900.80 | 2025-12-16 12:30:00 | 1908.30 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-17 10:10:00 | 1872.00 | 2025-12-17 10:55:00 | 1878.26 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-19 10:00:00 | 1865.00 | 2025-12-19 10:15:00 | 1876.21 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-12-19 10:00:00 | 1865.00 | 2025-12-19 11:15:00 | 1872.30 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-30 10:25:00 | 1843.50 | 2025-12-30 10:30:00 | 1834.91 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-12-30 10:25:00 | 1843.50 | 2025-12-30 13:55:00 | 1827.00 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2026-01-01 10:40:00 | 1907.70 | 2026-01-01 12:30:00 | 1921.77 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-01-01 10:40:00 | 1907.70 | 2026-01-01 15:20:00 | 1937.00 | TARGET_HIT | 0.50 | 1.54% |
| BUY | retest1 | 2026-01-08 09:30:00 | 1929.50 | 2026-01-08 09:35:00 | 1936.37 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-01-08 09:30:00 | 1929.50 | 2026-01-08 10:20:00 | 1929.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-20 11:05:00 | 1819.00 | 2026-01-20 11:40:00 | 1809.79 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-01-20 11:05:00 | 1819.00 | 2026-01-20 13:20:00 | 1819.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-23 10:25:00 | 1796.20 | 2026-01-23 10:55:00 | 1806.50 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-01-23 10:25:00 | 1796.20 | 2026-01-23 11:40:00 | 1796.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-28 09:40:00 | 1771.90 | 2026-01-28 09:55:00 | 1764.11 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-01-30 09:30:00 | 1755.90 | 2026-01-30 09:50:00 | 1765.46 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-01-30 09:30:00 | 1755.90 | 2026-01-30 13:15:00 | 1761.00 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-01 11:10:00 | 1768.10 | 2026-02-01 11:30:00 | 1776.12 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-01 11:10:00 | 1768.10 | 2026-02-01 12:00:00 | 1787.30 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2026-02-05 09:45:00 | 1731.20 | 2026-02-05 14:30:00 | 1736.46 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-19 09:30:00 | 1853.20 | 2026-02-19 09:35:00 | 1846.88 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1686.60 | 2026-03-13 10:00:00 | 1678.11 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1686.60 | 2026-03-13 12:45:00 | 1672.10 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2026-03-24 09:30:00 | 1656.30 | 2026-03-24 09:40:00 | 1648.59 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-03-27 11:05:00 | 1663.90 | 2026-03-27 11:15:00 | 1654.49 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-27 11:05:00 | 1663.90 | 2026-03-27 14:10:00 | 1641.80 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2026-04-01 10:55:00 | 1611.50 | 2026-04-01 11:15:00 | 1618.01 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1812.40 | 2026-04-15 09:55:00 | 1804.35 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-21 10:50:00 | 1819.70 | 2026-04-21 11:00:00 | 1815.07 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-23 11:15:00 | 1793.10 | 2026-04-23 12:25:00 | 1786.33 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-23 11:15:00 | 1793.10 | 2026-04-23 12:35:00 | 1793.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 10:40:00 | 1780.70 | 2026-04-28 11:20:00 | 1785.72 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-30 11:00:00 | 1749.90 | 2026-04-30 11:25:00 | 1755.55 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-04 09:50:00 | 1806.50 | 2026-05-04 10:30:00 | 1798.49 | STOP_HIT | 1.00 | -0.44% |
