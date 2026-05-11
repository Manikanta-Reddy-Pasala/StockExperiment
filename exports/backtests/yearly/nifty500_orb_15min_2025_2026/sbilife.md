# SBI Life Insurance Company Ltd. (SBILIFE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-10-07 15:25:00 (7800 bars)
- **Last close:** 1786.70
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
| ENTRY1 | 35 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 30
- **Target hits / Stop hits / Partials:** 5 / 30 / 11
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 2.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 7 | 36.8% | 2 | 12 | 5 | 0.05% | 0.9% |
| BUY @ 2nd Alert (retest1) | 19 | 7 | 36.8% | 2 | 12 | 5 | 0.05% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 9 | 33.3% | 3 | 18 | 6 | 0.05% | 1.4% |
| SELL @ 2nd Alert (retest1) | 27 | 9 | 33.3% | 3 | 18 | 6 | 0.05% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 46 | 16 | 34.8% | 5 | 30 | 11 | 0.05% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 11:10:00 | 1747.50 | 1743.68 | 0.00 | ORB-long ORB[1720.00,1741.70] vol=1.6x ATR=6.92 |
| Target hit | 2025-05-12 15:20:00 | 1751.10 | 1747.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2025-05-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-13 10:50:00 | 1738.10 | 1744.55 | 0.00 | ORB-short ORB[1742.10,1756.00] vol=2.9x ATR=4.30 |
| Stop hit — per-position SL triggered | 2025-05-13 11:55:00 | 1742.40 | 1742.14 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:35:00 | 1742.60 | 1746.45 | 0.00 | ORB-short ORB[1745.00,1758.90] vol=1.8x ATR=3.92 |
| Stop hit — per-position SL triggered | 2025-05-15 09:50:00 | 1746.52 | 1743.80 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:05:00 | 1768.80 | 1763.28 | 0.00 | ORB-long ORB[1751.20,1761.80] vol=1.5x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 11:20:00 | 1773.77 | 1767.12 | 0.00 | T1 1.5R @ 1773.77 |
| Stop hit — per-position SL triggered | 2025-05-21 11:45:00 | 1768.80 | 1769.47 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:30:00 | 1793.00 | 1797.98 | 0.00 | ORB-short ORB[1798.00,1810.10] vol=3.2x ATR=4.09 |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 1797.09 | 1797.54 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:35:00 | 1803.50 | 1807.65 | 0.00 | ORB-short ORB[1805.70,1816.40] vol=2.2x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-05-29 09:40:00 | 1806.67 | 1807.65 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:45:00 | 1767.50 | 1771.29 | 0.00 | ORB-short ORB[1768.90,1777.50] vol=1.7x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-06-04 11:15:00 | 1770.84 | 1770.31 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 11:05:00 | 1795.10 | 1779.37 | 0.00 | ORB-long ORB[1765.70,1789.60] vol=2.0x ATR=4.97 |
| Stop hit — per-position SL triggered | 2025-06-09 11:10:00 | 1790.13 | 1780.21 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 11:05:00 | 1774.40 | 1781.66 | 0.00 | ORB-short ORB[1776.00,1796.60] vol=2.7x ATR=3.30 |
| Stop hit — per-position SL triggered | 2025-06-10 13:30:00 | 1777.70 | 1777.77 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:50:00 | 1793.00 | 1784.15 | 0.00 | ORB-long ORB[1776.90,1791.10] vol=1.8x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-06-11 10:55:00 | 1790.11 | 1784.69 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 10:00:00 | 1746.60 | 1739.67 | 0.00 | ORB-long ORB[1720.00,1745.90] vol=2.5x ATR=6.48 |
| Stop hit — per-position SL triggered | 2025-06-13 10:05:00 | 1740.12 | 1739.77 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:50:00 | 1790.60 | 1776.63 | 0.00 | ORB-long ORB[1751.90,1774.90] vol=2.5x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 11:10:00 | 1797.44 | 1781.21 | 0.00 | T1 1.5R @ 1797.44 |
| Stop hit — per-position SL triggered | 2025-06-16 11:45:00 | 1790.60 | 1784.56 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:05:00 | 1815.80 | 1810.32 | 0.00 | ORB-long ORB[1800.20,1815.30] vol=2.5x ATR=3.41 |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1812.39 | 1810.69 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 09:40:00 | 1827.60 | 1837.82 | 0.00 | ORB-short ORB[1834.80,1861.00] vol=1.6x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:55:00 | 1819.94 | 1833.77 | 0.00 | T1 1.5R @ 1819.94 |
| Stop hit — per-position SL triggered | 2025-06-27 10:30:00 | 1827.60 | 1829.30 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 11:00:00 | 1830.30 | 1839.81 | 0.00 | ORB-short ORB[1841.80,1861.80] vol=1.7x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 11:40:00 | 1824.68 | 1836.96 | 0.00 | T1 1.5R @ 1824.68 |
| Target hit | 2025-07-03 15:20:00 | 1807.90 | 1821.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 1810.60 | 1805.75 | 0.00 | ORB-long ORB[1797.30,1808.40] vol=2.3x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:10:00 | 1814.46 | 1806.85 | 0.00 | T1 1.5R @ 1814.46 |
| Stop hit — per-position SL triggered | 2025-07-08 11:55:00 | 1810.60 | 1807.83 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:00:00 | 1803.40 | 1812.37 | 0.00 | ORB-short ORB[1811.00,1821.70] vol=1.8x ATR=3.18 |
| Stop hit — per-position SL triggered | 2025-07-10 10:10:00 | 1806.58 | 1811.50 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 10:40:00 | 1835.10 | 1840.39 | 0.00 | ORB-short ORB[1836.40,1847.20] vol=3.0x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-07-14 10:55:00 | 1837.97 | 1839.60 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:35:00 | 1807.70 | 1814.89 | 0.00 | ORB-short ORB[1812.60,1829.00] vol=1.7x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-07-17 10:00:00 | 1811.54 | 1812.89 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:50:00 | 1773.70 | 1782.75 | 0.00 | ORB-short ORB[1789.10,1802.80] vol=4.6x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:00:00 | 1766.94 | 1781.14 | 0.00 | T1 1.5R @ 1766.94 |
| Target hit | 2025-07-18 11:50:00 | 1772.40 | 1771.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — SELL (started 2025-07-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:25:00 | 1795.00 | 1800.12 | 0.00 | ORB-short ORB[1802.10,1812.10] vol=1.6x ATR=3.68 |
| Stop hit — per-position SL triggered | 2025-07-23 11:05:00 | 1798.68 | 1799.38 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:15:00 | 1794.50 | 1797.47 | 0.00 | ORB-short ORB[1800.90,1809.90] vol=1.5x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:55:00 | 1790.46 | 1796.01 | 0.00 | T1 1.5R @ 1790.46 |
| Stop hit — per-position SL triggered | 2025-07-24 13:00:00 | 1794.50 | 1792.48 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 11:00:00 | 1829.90 | 1835.14 | 0.00 | ORB-short ORB[1833.00,1857.40] vol=8.5x ATR=4.91 |
| Stop hit — per-position SL triggered | 2025-07-31 11:10:00 | 1834.81 | 1835.07 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:30:00 | 1787.90 | 1790.33 | 0.00 | ORB-short ORB[1790.80,1801.90] vol=2.0x ATR=4.54 |
| Stop hit — per-position SL triggered | 2025-08-04 10:40:00 | 1792.44 | 1790.54 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 11:00:00 | 1835.90 | 1829.62 | 0.00 | ORB-long ORB[1818.70,1833.30] vol=2.1x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 12:10:00 | 1841.06 | 1833.23 | 0.00 | T1 1.5R @ 1841.06 |
| Target hit | 2025-08-05 15:20:00 | 1857.60 | 1848.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:45:00 | 1891.00 | 1883.30 | 0.00 | ORB-long ORB[1870.00,1890.00] vol=2.3x ATR=5.32 |
| Stop hit — per-position SL triggered | 2025-08-21 09:55:00 | 1885.68 | 1884.27 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:40:00 | 1802.00 | 1806.01 | 0.00 | ORB-short ORB[1804.10,1818.90] vol=2.4x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-09-03 11:05:00 | 1805.35 | 1805.21 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 11:10:00 | 1798.50 | 1789.78 | 0.00 | ORB-long ORB[1778.30,1796.20] vol=4.4x ATR=3.29 |
| Stop hit — per-position SL triggered | 2025-09-09 12:55:00 | 1795.21 | 1790.93 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:50:00 | 1813.50 | 1817.80 | 0.00 | ORB-short ORB[1815.00,1833.00] vol=1.6x ATR=3.28 |
| Stop hit — per-position SL triggered | 2025-09-15 09:55:00 | 1816.78 | 1817.75 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 11:00:00 | 1827.40 | 1807.99 | 0.00 | ORB-long ORB[1795.30,1813.30] vol=2.2x ATR=4.01 |
| Stop hit — per-position SL triggered | 2025-09-18 11:05:00 | 1823.39 | 1812.96 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1838.40 | 1832.56 | 0.00 | ORB-long ORB[1821.80,1837.80] vol=3.7x ATR=4.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 10:45:00 | 1845.22 | 1833.43 | 0.00 | T1 1.5R @ 1845.22 |
| Stop hit — per-position SL triggered | 2025-09-19 11:40:00 | 1838.40 | 1834.75 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:30:00 | 1878.40 | 1865.68 | 0.00 | ORB-long ORB[1854.00,1869.80] vol=1.7x ATR=5.68 |
| Stop hit — per-position SL triggered | 2025-09-22 09:40:00 | 1872.72 | 1867.58 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:15:00 | 1840.20 | 1850.48 | 0.00 | ORB-short ORB[1843.30,1865.70] vol=1.7x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 13:00:00 | 1834.19 | 1845.98 | 0.00 | T1 1.5R @ 1834.19 |
| Target hit | 2025-09-23 15:20:00 | 1818.10 | 1833.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-09-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:45:00 | 1780.20 | 1787.43 | 0.00 | ORB-short ORB[1780.80,1795.70] vol=3.0x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-09-30 10:50:00 | 1783.24 | 1786.99 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:45:00 | 1779.70 | 1788.76 | 0.00 | ORB-short ORB[1788.50,1803.20] vol=2.1x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 11:05:00 | 1773.80 | 1787.01 | 0.00 | T1 1.5R @ 1773.80 |
| Stop hit — per-position SL triggered | 2025-10-03 14:35:00 | 1779.70 | 1778.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 11:10:00 | 1747.50 | 2025-05-12 15:20:00 | 1751.10 | TARGET_HIT | 1.00 | 0.21% |
| SELL | retest1 | 2025-05-13 10:50:00 | 1738.10 | 2025-05-13 11:55:00 | 1742.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-15 09:35:00 | 1742.60 | 2025-05-15 09:50:00 | 1746.52 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-05-21 11:05:00 | 1768.80 | 2025-05-21 11:20:00 | 1773.77 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-05-21 11:05:00 | 1768.80 | 2025-05-21 11:45:00 | 1768.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-26 10:30:00 | 1793.00 | 2025-05-26 11:15:00 | 1797.09 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-05-29 09:35:00 | 1803.50 | 2025-05-29 09:40:00 | 1806.67 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-06-04 10:45:00 | 1767.50 | 2025-06-04 11:15:00 | 1770.84 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-09 11:05:00 | 1795.10 | 2025-06-09 11:10:00 | 1790.13 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-10 11:05:00 | 1774.40 | 2025-06-10 13:30:00 | 1777.70 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-11 10:50:00 | 1793.00 | 2025-06-11 10:55:00 | 1790.11 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-06-13 10:00:00 | 1746.60 | 2025-06-13 10:05:00 | 1740.12 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-16 10:50:00 | 1790.60 | 2025-06-16 11:10:00 | 1797.44 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-06-16 10:50:00 | 1790.60 | 2025-06-16 11:45:00 | 1790.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-23 11:05:00 | 1815.80 | 2025-06-23 11:15:00 | 1812.39 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-06-27 09:40:00 | 1827.60 | 2025-06-27 09:55:00 | 1819.94 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-06-27 09:40:00 | 1827.60 | 2025-06-27 10:30:00 | 1827.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-03 11:00:00 | 1830.30 | 2025-07-03 11:40:00 | 1824.68 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-03 11:00:00 | 1830.30 | 2025-07-03 15:20:00 | 1807.90 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2025-07-08 11:05:00 | 1810.60 | 2025-07-08 11:10:00 | 1814.46 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-07-08 11:05:00 | 1810.60 | 2025-07-08 11:55:00 | 1810.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 10:00:00 | 1803.40 | 2025-07-10 10:10:00 | 1806.58 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-14 10:40:00 | 1835.10 | 2025-07-14 10:55:00 | 1837.97 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-17 09:35:00 | 1807.70 | 2025-07-17 10:00:00 | 1811.54 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-18 09:50:00 | 1773.70 | 2025-07-18 10:00:00 | 1766.94 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-18 09:50:00 | 1773.70 | 2025-07-18 11:50:00 | 1772.40 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2025-07-23 10:25:00 | 1795.00 | 2025-07-23 11:05:00 | 1798.68 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-24 11:15:00 | 1794.50 | 2025-07-24 11:55:00 | 1790.46 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-07-24 11:15:00 | 1794.50 | 2025-07-24 13:00:00 | 1794.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-31 11:00:00 | 1829.90 | 2025-07-31 11:10:00 | 1834.81 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-04 10:30:00 | 1787.90 | 2025-08-04 10:40:00 | 1792.44 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-05 11:00:00 | 1835.90 | 2025-08-05 12:10:00 | 1841.06 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-08-05 11:00:00 | 1835.90 | 2025-08-05 15:20:00 | 1857.60 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2025-08-21 09:45:00 | 1891.00 | 2025-08-21 09:55:00 | 1885.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-03 10:40:00 | 1802.00 | 2025-09-03 11:05:00 | 1805.35 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-09 11:10:00 | 1798.50 | 2025-09-09 12:55:00 | 1795.21 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-15 09:50:00 | 1813.50 | 2025-09-15 09:55:00 | 1816.78 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-18 11:00:00 | 1827.40 | 2025-09-18 11:05:00 | 1823.39 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-19 10:15:00 | 1838.40 | 2025-09-19 10:45:00 | 1845.22 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-09-19 10:15:00 | 1838.40 | 2025-09-19 11:40:00 | 1838.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 09:30:00 | 1878.40 | 2025-09-22 09:40:00 | 1872.72 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-23 11:15:00 | 1840.20 | 2025-09-23 13:00:00 | 1834.19 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-09-23 11:15:00 | 1840.20 | 2025-09-23 15:20:00 | 1818.10 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2025-09-30 10:45:00 | 1780.20 | 2025-09-30 10:50:00 | 1783.24 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-10-03 10:45:00 | 1779.70 | 2025-10-03 11:05:00 | 1773.80 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-03 10:45:00 | 1779.70 | 2025-10-03 14:35:00 | 1779.70 | STOP_HIT | 0.50 | 0.00% |
