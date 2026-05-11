# Bharti Airtel Ltd. (BHARTIARTL)

## Backtest Summary

- **Window:** 2026-03-09 09:15:00 → 2026-05-08 15:25:00 (3000 bars)
- **Last close:** 1834.70
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 5
- **Avg / median % per leg:** 0.06% / -0.19%
- **Sum % (uncompounded):** 1.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.17% | -1.5% |
| BUY @ 2nd Alert (retest1) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.17% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.21% | 3.0% |
| SELL @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.21% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 8 | 34.8% | 3 | 15 | 5 | 0.06% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:35:00 | 1859.00 | 1859.36 | 0.00 | ORB-short ORB[1859.70,1881.00] vol=2.3x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:15:00 | 1852.87 | 1858.91 | 0.00 | T1 1.5R @ 1852.87 |
| Target hit | 2026-03-10 12:45:00 | 1855.70 | 1855.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2026-03-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:00:00 | 1834.80 | 1835.07 | 0.00 | ORB-short ORB[1837.20,1847.70] vol=3.9x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:05:00 | 1828.61 | 1834.96 | 0.00 | T1 1.5R @ 1828.61 |
| Target hit | 2026-03-11 15:20:00 | 1801.00 | 1819.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-03-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:50:00 | 1826.00 | 1812.36 | 0.00 | ORB-long ORB[1791.10,1814.00] vol=1.6x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-03-17 11:00:00 | 1821.50 | 1813.20 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:55:00 | 1853.40 | 1843.34 | 0.00 | ORB-long ORB[1826.60,1852.80] vol=2.8x ATR=4.35 |
| Stop hit — per-position SL triggered | 2026-03-19 11:10:00 | 1849.05 | 1844.13 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:15:00 | 1804.40 | 1805.04 | 0.00 | ORB-short ORB[1808.40,1830.00] vol=1.6x ATR=4.04 |
| Stop hit — per-position SL triggered | 2026-03-23 11:25:00 | 1808.44 | 1805.18 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:00:00 | 1784.60 | 1798.98 | 0.00 | ORB-short ORB[1809.00,1826.80] vol=2.3x ATR=4.75 |
| Stop hit — per-position SL triggered | 2026-03-24 11:10:00 | 1789.35 | 1796.72 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:55:00 | 1798.30 | 1807.31 | 0.00 | ORB-short ORB[1800.00,1824.50] vol=1.6x ATR=5.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 11:50:00 | 1789.50 | 1804.41 | 0.00 | T1 1.5R @ 1789.50 |
| Stop hit — per-position SL triggered | 2026-03-30 12:00:00 | 1798.30 | 1803.74 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:00:00 | 1799.90 | 1806.70 | 0.00 | ORB-short ORB[1802.00,1817.30] vol=1.8x ATR=4.17 |
| Stop hit — per-position SL triggered | 2026-04-01 11:25:00 | 1804.07 | 1805.73 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 11:05:00 | 1810.10 | 1801.15 | 0.00 | ORB-long ORB[1772.00,1795.30] vol=2.8x ATR=5.60 |
| Stop hit — per-position SL triggered | 2026-04-07 11:45:00 | 1804.50 | 1804.02 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 11:00:00 | 1850.60 | 1856.44 | 0.00 | ORB-short ORB[1852.00,1869.90] vol=1.6x ATR=4.44 |
| Stop hit — per-position SL triggered | 2026-04-09 12:05:00 | 1855.04 | 1854.79 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 1871.60 | 1877.99 | 0.00 | ORB-short ORB[1876.20,1890.90] vol=1.8x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:25:00 | 1867.34 | 1877.39 | 0.00 | T1 1.5R @ 1867.34 |
| Target hit | 2026-04-15 15:20:00 | 1854.20 | 1863.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 1840.00 | 1849.34 | 0.00 | ORB-short ORB[1852.30,1864.90] vol=1.9x ATR=3.96 |
| Stop hit — per-position SL triggered | 2026-04-16 10:50:00 | 1843.96 | 1844.72 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:00:00 | 1866.00 | 1857.08 | 0.00 | ORB-long ORB[1830.00,1855.00] vol=2.1x ATR=3.51 |
| Stop hit — per-position SL triggered | 2026-04-21 11:35:00 | 1862.49 | 1860.53 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1836.50 | 1827.93 | 0.00 | ORB-long ORB[1812.10,1829.00] vol=2.5x ATR=3.88 |
| Stop hit — per-position SL triggered | 2026-04-23 12:25:00 | 1832.62 | 1831.98 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:05:00 | 1818.10 | 1821.78 | 0.00 | ORB-short ORB[1823.70,1850.70] vol=1.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 1821.27 | 1821.66 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:50:00 | 1838.60 | 1825.45 | 0.00 | ORB-long ORB[1813.80,1827.60] vol=1.5x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:00:00 | 1843.21 | 1827.42 | 0.00 | T1 1.5R @ 1843.21 |
| Stop hit — per-position SL triggered | 2026-04-28 11:25:00 | 1838.60 | 1831.73 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 1839.70 | 1833.22 | 0.00 | ORB-long ORB[1818.90,1839.20] vol=1.6x ATR=5.45 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 1834.25 | 1833.60 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1833.30 | 1826.18 | 0.00 | ORB-long ORB[1816.40,1828.00] vol=3.0x ATR=5.12 |
| Stop hit — per-position SL triggered | 2026-05-06 10:20:00 | 1828.18 | 1826.28 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-03-10 10:35:00 | 1859.00 | 2026-03-10 11:15:00 | 1852.87 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-03-10 10:35:00 | 1859.00 | 2026-03-10 12:45:00 | 1855.70 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2026-03-11 10:00:00 | 1834.80 | 2026-03-11 10:05:00 | 1828.61 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-11 10:00:00 | 1834.80 | 2026-03-11 15:20:00 | 1801.00 | TARGET_HIT | 0.50 | 1.84% |
| BUY | retest1 | 2026-03-17 10:50:00 | 1826.00 | 2026-03-17 11:00:00 | 1821.50 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-19 10:55:00 | 1853.40 | 2026-03-19 11:10:00 | 1849.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-23 11:15:00 | 1804.40 | 2026-03-23 11:25:00 | 1808.44 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-24 11:00:00 | 1784.60 | 2026-03-24 11:10:00 | 1789.35 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-30 10:55:00 | 1798.30 | 2026-03-30 11:50:00 | 1789.50 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-30 10:55:00 | 1798.30 | 2026-03-30 12:00:00 | 1798.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-01 11:00:00 | 1799.90 | 2026-04-01 11:25:00 | 1804.07 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-07 11:05:00 | 1810.10 | 2026-04-07 11:45:00 | 1804.50 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-09 11:00:00 | 1850.60 | 2026-04-09 12:05:00 | 1855.04 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-15 11:15:00 | 1871.60 | 2026-04-15 11:25:00 | 1867.34 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2026-04-15 11:15:00 | 1871.60 | 2026-04-15 15:20:00 | 1854.20 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2026-04-16 09:50:00 | 1840.00 | 2026-04-16 10:50:00 | 1843.96 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-21 11:00:00 | 1866.00 | 2026-04-21 11:35:00 | 1862.49 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-04-23 10:15:00 | 1836.50 | 2026-04-23 12:25:00 | 1832.62 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-04-24 11:05:00 | 1818.10 | 2026-04-24 11:20:00 | 1821.27 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-04-28 10:50:00 | 1838.60 | 2026-04-28 11:00:00 | 1843.21 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2026-04-28 10:50:00 | 1838.60 | 2026-04-28 11:25:00 | 1838.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:30:00 | 1839.70 | 2026-05-05 09:40:00 | 1834.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-06 10:15:00 | 1833.30 | 2026-05-06 10:20:00 | 1828.18 | STOP_HIT | 1.00 | -0.28% |
