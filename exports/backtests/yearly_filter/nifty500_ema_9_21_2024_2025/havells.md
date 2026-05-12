# Havells India Ltd. (HAVELLS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1253.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 154 |
| ALERT1 | 121 |
| ALERT2 | 120 |
| ALERT2_SKIP | 61 |
| ALERT3 | 325 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 126 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 120 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 100
- **Target hits / Stop hits / Partials:** 6 / 120 / 10
- **Avg / median % per leg:** 0.44% / -0.73%
- **Sum % (uncompounded):** 60.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 9 | 14.1% | 4 | 60 | 0 | -0.05% | -3.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 64 | 9 | 14.1% | 4 | 60 | 0 | -0.05% | -3.1% |
| SELL (all) | 72 | 27 | 37.5% | 2 | 60 | 10 | 0.88% | 63.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 72 | 27 | 37.5% | 2 | 60 | 10 | 0.88% | 63.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 136 | 36 | 26.5% | 6 | 120 | 10 | 0.44% | 60.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 13:15:00 | 1877.25 | 1886.12 | 1886.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 1856.50 | 1880.20 | 1884.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 1880.00 | 1872.77 | 1876.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 1880.00 | 1872.77 | 1876.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1880.00 | 1872.77 | 1876.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 1880.00 | 1872.77 | 1876.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 1908.80 | 1879.97 | 1879.87 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 1839.95 | 1875.17 | 1879.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 1769.80 | 1854.10 | 1869.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1811.75 | 1796.54 | 1824.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 1811.75 | 1796.54 | 1824.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1856.10 | 1804.12 | 1815.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 1851.10 | 1804.12 | 1815.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1849.65 | 1813.23 | 1818.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 1859.75 | 1813.23 | 1818.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 1839.75 | 1823.25 | 1822.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1844.50 | 1832.18 | 1826.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 1850.00 | 1854.97 | 1847.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 1856.75 | 1854.97 | 1847.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1842.70 | 1852.52 | 1847.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:00:00 | 1842.70 | 1852.52 | 1847.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 1839.10 | 1849.83 | 1846.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:15:00 | 1829.05 | 1849.83 | 1846.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 1829.30 | 1845.73 | 1844.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:00:00 | 1829.30 | 1845.73 | 1844.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 12:15:00 | 1833.25 | 1843.23 | 1843.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 09:15:00 | 1823.80 | 1836.13 | 1840.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 09:15:00 | 1842.00 | 1832.91 | 1835.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 09:15:00 | 1842.00 | 1832.91 | 1835.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1842.00 | 1832.91 | 1835.61 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 15:15:00 | 1838.25 | 1836.71 | 1836.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 10:15:00 | 1860.75 | 1841.78 | 1839.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 12:15:00 | 1841.10 | 1844.18 | 1840.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 12:15:00 | 1841.10 | 1844.18 | 1840.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 1841.10 | 1844.18 | 1840.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 12:45:00 | 1841.35 | 1844.18 | 1840.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 1839.85 | 1843.32 | 1840.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:00:00 | 1839.85 | 1843.32 | 1840.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 1839.30 | 1842.51 | 1840.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 1839.30 | 1842.51 | 1840.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 1838.55 | 1841.72 | 1840.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 1853.30 | 1841.72 | 1840.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1836.40 | 1840.62 | 1840.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:00:00 | 1836.40 | 1840.62 | 1840.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 11:15:00 | 1833.55 | 1839.21 | 1839.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 12:15:00 | 1829.90 | 1837.35 | 1838.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 11:15:00 | 1829.70 | 1828.21 | 1832.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-19 12:00:00 | 1829.70 | 1828.21 | 1832.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 1823.95 | 1827.35 | 1831.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 13:45:00 | 1819.80 | 1825.76 | 1830.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 14:30:00 | 1819.35 | 1824.02 | 1829.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:15:00 | 1818.50 | 1821.11 | 1825.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1832.95 | 1822.38 | 1824.69 | SL hit (close>static) qty=1.00 sl=1832.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 1878.10 | 1833.52 | 1829.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 11:15:00 | 1909.50 | 1848.72 | 1836.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 14:15:00 | 1916.90 | 1924.44 | 1904.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 15:00:00 | 1916.90 | 1924.44 | 1904.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1936.10 | 1925.29 | 1908.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:15:00 | 1944.50 | 1925.29 | 1908.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 1862.45 | 1909.33 | 1909.09 | SL hit (close<static) qty=1.00 sl=1901.20 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 10:15:00 | 1873.80 | 1902.23 | 1905.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 1851.40 | 1880.36 | 1894.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 1846.40 | 1840.74 | 1860.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 1846.40 | 1840.74 | 1860.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1846.40 | 1840.74 | 1860.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 12:00:00 | 1820.60 | 1827.10 | 1840.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 12:30:00 | 1818.25 | 1824.22 | 1837.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 12:15:00 | 1859.00 | 1840.49 | 1839.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 1859.00 | 1840.49 | 1839.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 1878.45 | 1851.64 | 1844.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 15:15:00 | 1888.80 | 1890.94 | 1880.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 09:15:00 | 1887.55 | 1890.94 | 1880.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1894.55 | 1891.66 | 1881.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 1885.10 | 1891.66 | 1881.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1871.45 | 1886.55 | 1881.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 1871.45 | 1886.55 | 1881.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 1869.90 | 1883.22 | 1880.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:30:00 | 1870.20 | 1883.22 | 1880.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1919.60 | 1890.21 | 1884.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 10:15:00 | 1922.45 | 1890.21 | 1884.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 12:00:00 | 1923.10 | 1902.10 | 1890.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 15:15:00 | 1925.00 | 1909.15 | 1897.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:30:00 | 1921.80 | 1916.29 | 1905.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 1922.00 | 1925.23 | 1916.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:45:00 | 1920.50 | 1925.23 | 1916.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 1917.65 | 1922.38 | 1917.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 1915.10 | 1919.12 | 1916.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 1910.90 | 1917.48 | 1915.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-12 14:15:00 | 1907.70 | 1913.44 | 1914.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 14:15:00 | 1907.70 | 1913.44 | 1914.20 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 1922.00 | 1914.60 | 1914.56 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 10:15:00 | 1910.90 | 1913.86 | 1914.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 11:15:00 | 1890.90 | 1909.27 | 1912.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 1904.25 | 1898.55 | 1904.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 1904.25 | 1898.55 | 1904.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1904.25 | 1898.55 | 1904.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 1905.30 | 1898.55 | 1904.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1894.45 | 1897.73 | 1903.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 11:15:00 | 1887.00 | 1897.73 | 1903.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 12:30:00 | 1882.45 | 1893.82 | 1900.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:15:00 | 1792.65 | 1841.89 | 1864.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:15:00 | 1788.33 | 1841.89 | 1864.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-23 12:15:00 | 1698.30 | 1751.38 | 1779.69 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 1795.00 | 1782.71 | 1781.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 10:15:00 | 1796.60 | 1786.12 | 1783.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 13:15:00 | 1829.50 | 1831.05 | 1814.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 13:45:00 | 1827.70 | 1831.05 | 1814.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1824.50 | 1836.44 | 1829.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 1824.50 | 1836.44 | 1829.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1818.75 | 1832.90 | 1828.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:00:00 | 1818.75 | 1832.90 | 1828.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 13:15:00 | 1820.30 | 1826.44 | 1826.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 14:15:00 | 1815.20 | 1824.20 | 1825.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 10:15:00 | 1836.30 | 1824.76 | 1825.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 10:15:00 | 1836.30 | 1824.76 | 1825.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1836.30 | 1824.76 | 1825.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 1838.05 | 1824.76 | 1825.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 11:15:00 | 1843.00 | 1828.41 | 1826.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 12:15:00 | 1847.60 | 1832.25 | 1828.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 11:15:00 | 1835.50 | 1843.28 | 1837.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 11:15:00 | 1835.50 | 1843.28 | 1837.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 1835.50 | 1843.28 | 1837.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 1835.50 | 1843.28 | 1837.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1841.55 | 1842.93 | 1837.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:45:00 | 1837.05 | 1842.93 | 1837.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 1831.75 | 1840.70 | 1836.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 1832.00 | 1840.70 | 1836.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1840.20 | 1840.60 | 1837.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 1830.90 | 1840.60 | 1837.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1838.90 | 1840.26 | 1837.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 1825.20 | 1840.26 | 1837.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1821.35 | 1836.48 | 1835.94 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 1825.00 | 1834.18 | 1834.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 15:15:00 | 1816.50 | 1826.61 | 1830.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1799.90 | 1795.67 | 1808.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1799.90 | 1795.67 | 1808.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1799.90 | 1795.67 | 1808.47 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 1818.00 | 1807.37 | 1806.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 13:15:00 | 1825.80 | 1811.06 | 1808.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 09:15:00 | 1809.55 | 1815.12 | 1811.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 09:15:00 | 1809.55 | 1815.12 | 1811.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1809.55 | 1815.12 | 1811.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:00:00 | 1809.55 | 1815.12 | 1811.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 1809.05 | 1813.90 | 1811.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 1809.05 | 1813.90 | 1811.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 1806.90 | 1812.50 | 1810.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:45:00 | 1798.30 | 1812.50 | 1810.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1802.90 | 1810.58 | 1810.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:00:00 | 1802.90 | 1810.58 | 1810.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 1796.40 | 1807.75 | 1808.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 1788.10 | 1803.82 | 1806.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 14:15:00 | 1800.00 | 1799.22 | 1802.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-09 15:00:00 | 1800.00 | 1799.22 | 1802.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1802.10 | 1799.09 | 1801.73 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 1807.25 | 1803.29 | 1803.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 1812.00 | 1805.03 | 1804.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 09:15:00 | 1816.55 | 1821.32 | 1815.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 1816.55 | 1821.32 | 1815.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1816.55 | 1821.32 | 1815.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1809.85 | 1821.32 | 1815.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1819.50 | 1820.96 | 1815.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:45:00 | 1815.00 | 1820.96 | 1815.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 1876.00 | 1879.70 | 1868.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 1876.00 | 1879.70 | 1868.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1895.75 | 1902.21 | 1895.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:30:00 | 1895.80 | 1902.21 | 1895.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1900.00 | 1901.77 | 1896.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1905.55 | 1901.77 | 1896.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1895.10 | 1900.43 | 1896.20 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 1883.20 | 1892.60 | 1893.63 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 1907.15 | 1894.77 | 1894.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 12:15:00 | 1910.15 | 1899.53 | 1896.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 12:15:00 | 1904.60 | 1914.84 | 1907.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 12:15:00 | 1904.60 | 1914.84 | 1907.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1904.60 | 1914.84 | 1907.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 1904.60 | 1914.84 | 1907.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 1903.25 | 1912.52 | 1907.20 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 1885.20 | 1902.01 | 1903.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 1875.80 | 1891.47 | 1896.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 1891.85 | 1883.87 | 1890.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 14:15:00 | 1891.85 | 1883.87 | 1890.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1891.85 | 1883.87 | 1890.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 1891.85 | 1883.87 | 1890.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1896.00 | 1886.30 | 1890.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 1888.55 | 1886.30 | 1890.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1895.85 | 1889.73 | 1891.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 1895.85 | 1889.73 | 1891.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1895.00 | 1890.78 | 1891.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:45:00 | 1896.15 | 1890.78 | 1891.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 1914.50 | 1895.53 | 1893.92 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 14:15:00 | 1884.90 | 1893.97 | 1895.10 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 1904.20 | 1896.28 | 1895.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 1909.10 | 1901.08 | 1898.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 1899.50 | 1901.79 | 1899.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 11:15:00 | 1899.50 | 1901.79 | 1899.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 1899.50 | 1901.79 | 1899.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:45:00 | 1898.95 | 1901.79 | 1899.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 1901.85 | 1901.80 | 1899.53 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 1883.00 | 1897.99 | 1898.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 10:15:00 | 1874.50 | 1893.29 | 1896.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 1886.40 | 1876.27 | 1880.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 10:15:00 | 1886.40 | 1876.27 | 1880.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1886.40 | 1876.27 | 1880.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:30:00 | 1888.45 | 1876.27 | 1880.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1877.40 | 1876.50 | 1880.41 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 1893.95 | 1883.18 | 1882.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 1939.55 | 1894.46 | 1887.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 11:15:00 | 1982.25 | 1988.04 | 1967.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 11:45:00 | 1984.10 | 1988.04 | 1967.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1983.25 | 1990.06 | 1982.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 14:30:00 | 2008.15 | 1995.84 | 1988.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 15:00:00 | 2008.55 | 1995.84 | 1988.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 1973.20 | 1986.46 | 1988.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 1973.20 | 1986.46 | 1988.04 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 2001.40 | 1990.10 | 1989.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 2005.85 | 1995.74 | 1992.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 2072.35 | 2073.20 | 2049.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 09:30:00 | 2074.05 | 2073.20 | 2049.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 2040.50 | 2065.72 | 2057.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 2040.50 | 2065.72 | 2057.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 2051.95 | 2062.97 | 2056.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:30:00 | 2053.90 | 2056.00 | 2054.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 2027.65 | 2052.57 | 2053.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 2027.65 | 2052.57 | 2053.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 2000.50 | 2030.74 | 2042.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 15:15:00 | 2027.90 | 2025.28 | 2036.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:15:00 | 2011.00 | 2025.28 | 2036.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 2020.90 | 2024.41 | 2034.90 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 11:15:00 | 2048.65 | 2033.38 | 2033.01 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 14:15:00 | 2010.40 | 2030.31 | 2031.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 2007.70 | 2024.19 | 2028.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 10:15:00 | 2000.10 | 1999.22 | 2009.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:00:00 | 2000.10 | 1999.22 | 2009.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 1990.75 | 1997.68 | 2007.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:45:00 | 2004.80 | 1997.68 | 2007.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 1943.45 | 1930.90 | 1943.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 1943.45 | 1930.90 | 1943.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 1947.15 | 1934.15 | 1943.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 1943.80 | 1934.15 | 1943.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 1949.55 | 1937.23 | 1944.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 1949.55 | 1937.23 | 1944.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1952.00 | 1940.18 | 1945.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1951.75 | 1940.18 | 1945.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1966.35 | 1946.43 | 1947.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 1966.35 | 1946.43 | 1947.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1963.55 | 1949.85 | 1948.65 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 1926.90 | 1944.61 | 1946.54 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 1947.50 | 1943.61 | 1943.08 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 1927.00 | 1940.75 | 1942.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 10:15:00 | 1923.45 | 1934.24 | 1937.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 1941.55 | 1931.31 | 1934.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 14:15:00 | 1941.55 | 1931.31 | 1934.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 1941.55 | 1931.31 | 1934.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 1941.55 | 1931.31 | 1934.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 1944.35 | 1933.92 | 1935.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 1919.85 | 1933.92 | 1935.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 1674.90 | 1669.91 | 1679.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:45:00 | 1681.40 | 1669.91 | 1679.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1647.50 | 1663.40 | 1672.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 10:15:00 | 1640.00 | 1663.40 | 1672.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 10:45:00 | 1639.95 | 1658.15 | 1669.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 12:30:00 | 1641.80 | 1651.38 | 1664.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-01 18:45:00 | 1646.25 | 1646.54 | 1656.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1635.80 | 1620.23 | 1629.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:45:00 | 1636.45 | 1620.23 | 1629.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 1634.00 | 1622.99 | 1629.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 1661.00 | 1622.99 | 1629.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1663.00 | 1630.99 | 1632.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:45:00 | 1662.60 | 1630.99 | 1632.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 1662.65 | 1637.32 | 1635.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1662.65 | 1637.32 | 1635.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 1676.50 | 1645.16 | 1639.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1662.35 | 1663.76 | 1652.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 1662.35 | 1663.76 | 1652.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1653.35 | 1661.68 | 1652.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 1653.35 | 1661.68 | 1652.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1649.00 | 1659.14 | 1652.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 1648.00 | 1659.14 | 1652.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 1661.00 | 1659.51 | 1652.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:15:00 | 1664.50 | 1659.77 | 1653.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 1664.75 | 1660.71 | 1655.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:00:00 | 1664.95 | 1660.20 | 1656.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:30:00 | 1663.40 | 1658.18 | 1657.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 1659.10 | 1658.36 | 1657.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:15:00 | 1655.10 | 1658.36 | 1657.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-11 13:15:00 | 1649.00 | 1656.49 | 1656.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 1649.00 | 1656.49 | 1656.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 14:15:00 | 1643.85 | 1653.96 | 1655.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 13:15:00 | 1615.10 | 1611.84 | 1624.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:45:00 | 1618.85 | 1611.84 | 1624.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1626.00 | 1615.98 | 1623.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1626.00 | 1615.98 | 1623.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1620.40 | 1616.86 | 1623.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 1632.70 | 1616.86 | 1623.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1626.50 | 1618.79 | 1623.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:30:00 | 1624.00 | 1618.79 | 1623.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1628.60 | 1620.75 | 1623.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 1628.60 | 1620.75 | 1623.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1623.15 | 1621.23 | 1623.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:15:00 | 1618.35 | 1621.23 | 1623.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:45:00 | 1620.40 | 1621.08 | 1623.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 1620.25 | 1622.61 | 1623.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1650.15 | 1627.56 | 1625.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 1650.15 | 1627.56 | 1625.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 1660.95 | 1634.24 | 1628.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 1630.60 | 1639.57 | 1633.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 1630.60 | 1639.57 | 1633.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1630.60 | 1639.57 | 1633.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 1630.60 | 1639.57 | 1633.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 1630.30 | 1637.71 | 1633.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 1620.55 | 1637.71 | 1633.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1625.55 | 1635.28 | 1632.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 12:15:00 | 1632.25 | 1631.71 | 1631.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 13:45:00 | 1630.90 | 1632.06 | 1631.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 15:15:00 | 1730.00 | 1735.55 | 1735.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 15:15:00 | 1730.00 | 1735.55 | 1735.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 10:15:00 | 1725.35 | 1732.89 | 1734.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 12:15:00 | 1733.30 | 1731.83 | 1733.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 12:15:00 | 1733.30 | 1731.83 | 1733.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 1733.30 | 1731.83 | 1733.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:45:00 | 1731.50 | 1731.83 | 1733.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 1739.20 | 1733.30 | 1734.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:00:00 | 1739.20 | 1733.30 | 1734.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 1748.35 | 1736.31 | 1735.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 11:15:00 | 1752.10 | 1744.21 | 1739.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 14:15:00 | 1738.95 | 1744.51 | 1741.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 14:15:00 | 1738.95 | 1744.51 | 1741.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 1738.95 | 1744.51 | 1741.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 1738.95 | 1744.51 | 1741.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 1736.00 | 1742.81 | 1740.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 1732.15 | 1742.81 | 1740.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 10:15:00 | 1730.55 | 1737.86 | 1738.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 13:15:00 | 1718.80 | 1731.61 | 1735.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 1722.25 | 1714.37 | 1720.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 1722.25 | 1714.37 | 1720.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1722.25 | 1714.37 | 1720.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 1716.45 | 1714.37 | 1720.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1740.80 | 1719.65 | 1722.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:00:00 | 1740.80 | 1719.65 | 1722.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 1749.00 | 1725.52 | 1724.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 15:15:00 | 1754.00 | 1739.37 | 1732.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 10:15:00 | 1740.00 | 1740.36 | 1734.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 11:00:00 | 1740.00 | 1740.36 | 1734.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1748.20 | 1752.77 | 1745.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 1745.10 | 1752.77 | 1745.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 1756.00 | 1753.42 | 1746.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:45:00 | 1759.30 | 1754.32 | 1747.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 14:45:00 | 1758.25 | 1755.30 | 1748.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 1761.30 | 1755.75 | 1749.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 10:00:00 | 1758.20 | 1763.50 | 1758.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1752.90 | 1761.38 | 1757.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 1736.80 | 1754.12 | 1754.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 1736.80 | 1754.12 | 1754.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 1724.40 | 1748.18 | 1752.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 11:15:00 | 1700.00 | 1698.20 | 1708.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:45:00 | 1699.90 | 1698.20 | 1708.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1679.95 | 1670.28 | 1680.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 1679.95 | 1670.28 | 1680.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1682.50 | 1672.73 | 1681.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 1684.25 | 1672.73 | 1681.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 1682.35 | 1674.65 | 1681.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:00:00 | 1682.35 | 1674.65 | 1681.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 1676.20 | 1674.96 | 1680.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:45:00 | 1682.40 | 1674.96 | 1680.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1668.80 | 1672.71 | 1677.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:30:00 | 1664.50 | 1671.38 | 1676.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:15:00 | 1665.05 | 1669.51 | 1674.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:45:00 | 1666.55 | 1670.80 | 1674.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:45:00 | 1665.30 | 1671.05 | 1674.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1662.50 | 1652.58 | 1658.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 1664.00 | 1652.58 | 1658.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1668.35 | 1655.73 | 1659.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:45:00 | 1672.50 | 1655.73 | 1659.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1670.45 | 1658.67 | 1660.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-31 11:15:00 | 1671.90 | 1661.32 | 1661.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 11:15:00 | 1671.90 | 1661.32 | 1661.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 1685.50 | 1668.05 | 1664.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 1669.60 | 1670.44 | 1666.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 09:15:00 | 1669.60 | 1670.44 | 1666.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1669.60 | 1670.44 | 1666.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:45:00 | 1668.20 | 1670.44 | 1666.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1670.15 | 1675.82 | 1672.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 1670.15 | 1675.82 | 1672.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 1672.20 | 1675.09 | 1672.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:00:00 | 1680.85 | 1676.25 | 1673.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 1652.40 | 1684.98 | 1685.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1652.40 | 1684.98 | 1685.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 1646.15 | 1664.85 | 1674.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 1638.90 | 1631.54 | 1644.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 1638.90 | 1631.54 | 1644.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1646.85 | 1634.60 | 1644.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1646.85 | 1634.60 | 1644.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1636.00 | 1634.88 | 1643.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 1629.30 | 1634.88 | 1643.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1633.20 | 1634.55 | 1642.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 1614.85 | 1632.93 | 1638.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 1534.11 | 1574.80 | 1599.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 13:15:00 | 1540.00 | 1538.50 | 1562.53 | SL hit (close>ema200) qty=0.50 sl=1538.50 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 1556.85 | 1547.88 | 1547.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 1568.40 | 1551.98 | 1548.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1586.75 | 1591.58 | 1580.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 1586.75 | 1591.58 | 1580.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1586.20 | 1590.51 | 1581.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 1585.70 | 1590.51 | 1581.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 1593.75 | 1591.09 | 1583.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 1586.60 | 1591.09 | 1583.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 1587.70 | 1590.41 | 1583.53 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 1551.05 | 1577.78 | 1579.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 1538.50 | 1562.16 | 1570.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1585.80 | 1564.27 | 1569.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1585.80 | 1564.27 | 1569.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1585.80 | 1564.27 | 1569.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1585.80 | 1564.27 | 1569.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1580.60 | 1567.53 | 1570.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1588.45 | 1567.53 | 1570.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 1584.85 | 1573.91 | 1572.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 13:15:00 | 1596.30 | 1578.39 | 1574.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 1579.50 | 1583.25 | 1578.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 1579.50 | 1583.25 | 1578.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1579.50 | 1583.25 | 1578.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 1579.50 | 1583.25 | 1578.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1576.10 | 1581.82 | 1578.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 1576.10 | 1581.82 | 1578.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1562.45 | 1577.94 | 1576.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:00:00 | 1562.45 | 1577.94 | 1576.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 1547.00 | 1571.76 | 1574.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1533.85 | 1564.17 | 1570.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1507.75 | 1501.99 | 1515.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 1507.75 | 1501.99 | 1515.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1510.90 | 1503.77 | 1515.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 1511.00 | 1503.77 | 1515.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 1524.85 | 1509.98 | 1515.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:45:00 | 1524.50 | 1509.98 | 1515.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 1529.90 | 1513.96 | 1516.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:45:00 | 1530.35 | 1513.96 | 1516.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 1537.45 | 1521.33 | 1519.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 11:15:00 | 1547.80 | 1530.36 | 1524.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 1528.25 | 1532.16 | 1526.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 1528.25 | 1532.16 | 1526.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1528.25 | 1532.16 | 1526.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 1524.65 | 1532.16 | 1526.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1545.85 | 1534.90 | 1528.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:30:00 | 1548.45 | 1537.77 | 1531.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:30:00 | 1550.50 | 1542.25 | 1533.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:00:00 | 1563.00 | 1559.01 | 1548.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-03 09:15:00 | 1703.30 | 1628.22 | 1590.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 1607.15 | 1624.63 | 1625.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 1604.95 | 1617.79 | 1621.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 1616.90 | 1607.38 | 1612.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 10:15:00 | 1616.90 | 1607.38 | 1612.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1616.90 | 1607.38 | 1612.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 1616.90 | 1607.38 | 1612.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1608.00 | 1607.50 | 1611.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 1605.50 | 1607.50 | 1611.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:45:00 | 1605.00 | 1606.90 | 1611.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:30:00 | 1606.75 | 1608.24 | 1610.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1525.22 | 1550.28 | 1572.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1524.75 | 1550.28 | 1572.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1526.41 | 1550.28 | 1572.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-14 09:15:00 | 1547.55 | 1529.91 | 1540.05 | SL hit (close>ema200) qty=0.50 sl=1529.91 alert=retest2 |

### Cycle 54 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 1517.00 | 1514.87 | 1514.75 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 09:15:00 | 1509.25 | 1513.74 | 1514.25 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 1521.70 | 1514.51 | 1514.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 1532.15 | 1518.04 | 1516.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1519.30 | 1520.52 | 1517.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 1519.30 | 1520.52 | 1517.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1501.95 | 1516.80 | 1516.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 1501.95 | 1516.80 | 1516.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 1515.20 | 1516.48 | 1516.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:30:00 | 1520.90 | 1517.49 | 1516.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:45:00 | 1519.80 | 1518.57 | 1517.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 1503.95 | 1515.87 | 1516.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 1503.95 | 1515.87 | 1516.27 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 13:15:00 | 1521.10 | 1516.58 | 1516.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 14:15:00 | 1523.10 | 1517.89 | 1516.90 | Break + close above crossover candle high |

### Cycle 59 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 1448.00 | 1518.67 | 1521.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 1385.00 | 1422.39 | 1451.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 1424.80 | 1420.86 | 1445.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:00:00 | 1424.80 | 1420.86 | 1445.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1438.20 | 1417.56 | 1426.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 1438.25 | 1417.56 | 1426.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1430.10 | 1420.07 | 1427.18 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 1445.20 | 1433.07 | 1432.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1473.80 | 1444.66 | 1437.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 1448.65 | 1454.94 | 1446.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 1448.65 | 1454.94 | 1446.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 1448.05 | 1453.56 | 1446.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 1454.65 | 1453.56 | 1446.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1452.00 | 1453.25 | 1447.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:30:00 | 1445.30 | 1453.25 | 1447.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1464.00 | 1471.80 | 1464.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1452.80 | 1471.80 | 1464.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1464.50 | 1470.34 | 1464.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 1456.70 | 1470.34 | 1464.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1472.20 | 1470.71 | 1465.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:30:00 | 1466.25 | 1470.71 | 1465.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1474.60 | 1471.49 | 1466.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:30:00 | 1470.80 | 1471.49 | 1466.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1464.05 | 1470.00 | 1466.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 1464.05 | 1470.00 | 1466.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 1473.95 | 1470.79 | 1466.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:15:00 | 1475.95 | 1470.79 | 1466.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:45:00 | 1474.55 | 1470.74 | 1467.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 15:15:00 | 1475.00 | 1470.74 | 1467.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 09:15:00 | 1460.90 | 1469.46 | 1467.22 | SL hit (close<static) qty=1.00 sl=1462.55 alert=retest2 |

### Cycle 61 — SELL (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 12:15:00 | 1458.45 | 1464.83 | 1465.49 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 1479.65 | 1467.53 | 1466.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 15:15:00 | 1484.00 | 1470.83 | 1468.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 12:15:00 | 1472.55 | 1475.40 | 1471.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 13:00:00 | 1472.55 | 1475.40 | 1471.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 1466.55 | 1473.63 | 1471.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:00:00 | 1466.55 | 1473.63 | 1471.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1458.55 | 1470.61 | 1470.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 1458.55 | 1470.61 | 1470.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 1452.25 | 1466.94 | 1468.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 09:15:00 | 1440.30 | 1461.61 | 1465.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 11:15:00 | 1460.40 | 1459.92 | 1464.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 11:15:00 | 1460.40 | 1459.92 | 1464.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 1460.40 | 1459.92 | 1464.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:45:00 | 1461.95 | 1459.92 | 1464.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 1458.95 | 1459.73 | 1463.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:30:00 | 1461.85 | 1459.73 | 1463.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 1460.15 | 1459.76 | 1462.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 1472.80 | 1459.76 | 1462.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1480.35 | 1463.88 | 1464.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 1480.35 | 1463.88 | 1464.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1486.70 | 1468.44 | 1466.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 1515.00 | 1477.75 | 1470.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1490.40 | 1532.65 | 1515.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 1490.40 | 1532.65 | 1515.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1490.40 | 1532.65 | 1515.76 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 15:15:00 | 1500.00 | 1506.94 | 1507.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 09:15:00 | 1494.10 | 1504.37 | 1506.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 1479.85 | 1478.11 | 1483.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 1479.85 | 1478.11 | 1483.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1479.85 | 1478.11 | 1483.56 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 13:15:00 | 1503.50 | 1489.08 | 1487.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 09:15:00 | 1507.00 | 1493.57 | 1490.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 13:15:00 | 1519.40 | 1520.74 | 1510.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 14:00:00 | 1519.40 | 1520.74 | 1510.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1511.90 | 1520.12 | 1512.70 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 1501.35 | 1508.24 | 1508.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 1490.15 | 1502.36 | 1505.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 1503.10 | 1501.26 | 1504.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 11:15:00 | 1503.10 | 1501.26 | 1504.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1503.10 | 1501.26 | 1504.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 1503.10 | 1501.26 | 1504.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1509.05 | 1502.82 | 1504.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 1508.60 | 1502.82 | 1504.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1509.00 | 1504.05 | 1505.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 1506.25 | 1504.05 | 1505.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1509.30 | 1505.10 | 1505.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 1509.30 | 1505.10 | 1505.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 1512.95 | 1506.67 | 1506.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 1519.60 | 1509.88 | 1507.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1486.20 | 1510.94 | 1510.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1486.20 | 1510.94 | 1510.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1486.20 | 1510.94 | 1510.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1486.20 | 1510.94 | 1510.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1484.30 | 1505.61 | 1508.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 1462.80 | 1497.05 | 1503.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 1459.20 | 1451.44 | 1469.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 1459.20 | 1451.44 | 1469.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1452.60 | 1453.20 | 1467.16 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1500.60 | 1472.45 | 1469.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 1517.75 | 1481.51 | 1473.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 1611.00 | 1649.12 | 1639.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 1611.00 | 1649.12 | 1639.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1611.00 | 1649.12 | 1639.40 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 12:15:00 | 1615.90 | 1631.12 | 1632.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 09:15:00 | 1595.20 | 1617.40 | 1625.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 14:15:00 | 1604.10 | 1603.41 | 1614.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 15:00:00 | 1604.10 | 1603.41 | 1614.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1591.80 | 1601.34 | 1611.34 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1619.90 | 1607.26 | 1606.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 12:15:00 | 1629.60 | 1616.02 | 1611.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 1614.90 | 1618.58 | 1614.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1614.90 | 1618.58 | 1614.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1614.90 | 1618.58 | 1614.15 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 12:15:00 | 1596.10 | 1610.60 | 1611.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 09:15:00 | 1591.00 | 1603.23 | 1607.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 1580.10 | 1577.66 | 1588.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:45:00 | 1580.10 | 1577.66 | 1588.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1588.40 | 1579.81 | 1588.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1588.40 | 1579.81 | 1588.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1583.20 | 1580.49 | 1588.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 15:15:00 | 1579.30 | 1581.38 | 1587.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:15:00 | 1580.00 | 1582.36 | 1586.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:30:00 | 1581.00 | 1584.30 | 1586.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:45:00 | 1579.10 | 1583.86 | 1585.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1571.00 | 1580.69 | 1584.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 1579.90 | 1580.69 | 1584.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1585.00 | 1581.95 | 1584.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 1584.10 | 1581.95 | 1584.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1580.70 | 1581.70 | 1583.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 1585.00 | 1581.70 | 1583.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1578.80 | 1581.07 | 1583.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1578.80 | 1581.07 | 1583.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1574.20 | 1579.37 | 1581.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 10:30:00 | 1572.00 | 1578.35 | 1581.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:15:00 | 1572.10 | 1578.35 | 1581.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 11:15:00 | 1579.60 | 1559.82 | 1559.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1579.60 | 1559.82 | 1559.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1583.40 | 1569.88 | 1564.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 11:15:00 | 1593.90 | 1597.83 | 1590.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:30:00 | 1595.80 | 1597.83 | 1590.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1587.50 | 1595.76 | 1589.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 1582.10 | 1595.76 | 1589.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1586.90 | 1593.99 | 1589.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:30:00 | 1586.80 | 1593.99 | 1589.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1578.30 | 1590.38 | 1589.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 1578.30 | 1590.38 | 1589.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 10:15:00 | 1573.20 | 1586.94 | 1587.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1565.20 | 1576.21 | 1580.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1589.60 | 1574.94 | 1578.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1589.60 | 1574.94 | 1578.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1589.60 | 1574.94 | 1578.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1589.60 | 1574.94 | 1578.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1582.00 | 1576.35 | 1578.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 1575.80 | 1574.66 | 1577.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:45:00 | 1577.90 | 1575.45 | 1577.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 1577.60 | 1575.45 | 1577.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:00:00 | 1576.70 | 1572.76 | 1572.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 15:15:00 | 1574.00 | 1573.05 | 1573.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 1574.00 | 1573.05 | 1573.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1577.10 | 1573.86 | 1573.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 1572.40 | 1573.57 | 1573.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 1572.40 | 1573.57 | 1573.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1572.40 | 1573.57 | 1573.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 1572.40 | 1573.57 | 1573.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 1575.70 | 1573.99 | 1573.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:45:00 | 1575.60 | 1573.99 | 1573.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1574.50 | 1574.09 | 1573.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 1571.20 | 1574.09 | 1573.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 13:15:00 | 1567.80 | 1572.84 | 1573.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 1564.00 | 1569.35 | 1571.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 1572.40 | 1569.96 | 1571.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 10:15:00 | 1572.40 | 1569.96 | 1571.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1572.40 | 1569.96 | 1571.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 1572.40 | 1569.96 | 1571.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1569.70 | 1569.91 | 1571.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:45:00 | 1567.70 | 1570.00 | 1571.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1567.80 | 1570.87 | 1571.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 11:15:00 | 1489.32 | 1497.85 | 1505.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 11:15:00 | 1489.41 | 1497.85 | 1505.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 1494.50 | 1492.90 | 1499.75 | SL hit (close>ema200) qty=0.50 sl=1492.90 alert=retest2 |

### Cycle 78 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 1522.90 | 1505.01 | 1503.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 15:15:00 | 1525.00 | 1511.81 | 1507.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 1575.40 | 1575.58 | 1564.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:45:00 | 1573.90 | 1575.58 | 1564.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1560.10 | 1571.96 | 1565.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1560.10 | 1571.96 | 1565.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1556.50 | 1568.87 | 1564.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 1556.50 | 1568.87 | 1564.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 1542.70 | 1561.25 | 1561.55 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1564.60 | 1553.61 | 1552.36 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 1530.80 | 1548.46 | 1550.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1522.30 | 1534.21 | 1540.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 1528.80 | 1525.01 | 1530.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 1528.80 | 1525.01 | 1530.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1528.80 | 1525.01 | 1530.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 1531.20 | 1525.01 | 1530.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1528.30 | 1525.96 | 1530.33 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 1551.50 | 1533.18 | 1532.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 1563.00 | 1541.92 | 1537.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 1574.80 | 1578.12 | 1568.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:00:00 | 1574.80 | 1578.12 | 1568.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1569.20 | 1576.34 | 1568.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 1569.20 | 1576.34 | 1568.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1572.00 | 1575.47 | 1568.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 1582.70 | 1575.47 | 1568.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 1564.90 | 1573.36 | 1568.23 | SL hit (close<static) qty=1.00 sl=1566.60 alert=retest2 |

### Cycle 83 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1544.40 | 1566.32 | 1568.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 1540.40 | 1561.14 | 1565.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 1555.00 | 1553.13 | 1558.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 1555.00 | 1553.13 | 1558.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1555.00 | 1553.13 | 1558.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 1545.20 | 1551.42 | 1557.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 1546.10 | 1548.06 | 1553.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 1548.30 | 1546.00 | 1549.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 1548.90 | 1547.94 | 1550.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 1573.00 | 1552.95 | 1552.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 1573.00 | 1552.95 | 1552.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 1582.00 | 1558.76 | 1554.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 15:15:00 | 1570.00 | 1570.75 | 1563.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 09:15:00 | 1580.60 | 1570.75 | 1563.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1585.00 | 1573.60 | 1565.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 1589.40 | 1578.88 | 1572.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 12:15:00 | 1546.70 | 1572.15 | 1571.18 | SL hit (close<static) qty=1.00 sl=1564.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 1530.60 | 1563.84 | 1567.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 1526.60 | 1548.75 | 1559.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 1525.00 | 1522.01 | 1529.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:30:00 | 1525.10 | 1522.01 | 1529.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1527.30 | 1523.06 | 1529.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 1527.30 | 1523.06 | 1529.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1529.80 | 1524.41 | 1529.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 1529.80 | 1524.41 | 1529.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1531.00 | 1525.73 | 1529.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1527.20 | 1525.73 | 1529.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1530.60 | 1526.70 | 1529.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:30:00 | 1519.90 | 1526.70 | 1528.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 1519.40 | 1527.39 | 1528.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 1533.70 | 1527.94 | 1527.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1533.70 | 1527.94 | 1527.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 1536.20 | 1529.59 | 1528.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1523.60 | 1531.87 | 1530.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1523.60 | 1531.87 | 1530.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1523.60 | 1531.87 | 1530.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:15:00 | 1523.40 | 1531.87 | 1530.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1525.00 | 1530.49 | 1529.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:45:00 | 1530.10 | 1530.15 | 1529.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 1530.00 | 1530.15 | 1529.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 1525.40 | 1528.60 | 1528.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 1525.40 | 1528.60 | 1528.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 10:15:00 | 1523.90 | 1526.38 | 1527.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 12:15:00 | 1531.40 | 1527.23 | 1527.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 1531.40 | 1527.23 | 1527.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1531.40 | 1527.23 | 1527.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:45:00 | 1532.90 | 1527.23 | 1527.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 13:15:00 | 1532.30 | 1528.24 | 1528.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 1535.60 | 1530.94 | 1529.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 12:15:00 | 1519.60 | 1530.25 | 1529.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 12:15:00 | 1519.60 | 1530.25 | 1529.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1519.60 | 1530.25 | 1529.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 1519.60 | 1530.25 | 1529.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1520.10 | 1528.22 | 1528.88 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1540.70 | 1529.50 | 1528.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 13:15:00 | 1571.30 | 1543.19 | 1535.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 1560.00 | 1563.41 | 1554.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:45:00 | 1564.40 | 1563.41 | 1554.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1558.40 | 1563.47 | 1556.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 1558.40 | 1563.47 | 1556.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 1557.80 | 1562.33 | 1556.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:45:00 | 1559.30 | 1562.33 | 1556.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 1550.20 | 1559.91 | 1555.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 1550.20 | 1559.91 | 1555.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1552.70 | 1558.46 | 1555.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 1552.00 | 1558.46 | 1555.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1537.90 | 1552.58 | 1553.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 1527.90 | 1543.84 | 1548.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1510.80 | 1509.86 | 1521.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 1510.80 | 1509.86 | 1521.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1519.30 | 1513.21 | 1518.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 1519.30 | 1513.21 | 1518.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 1524.00 | 1515.37 | 1519.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:15:00 | 1524.00 | 1515.37 | 1519.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 1525.80 | 1517.45 | 1519.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 1525.80 | 1517.45 | 1519.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 1529.00 | 1522.54 | 1521.66 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1514.30 | 1520.89 | 1520.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 1498.90 | 1509.98 | 1514.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 1491.80 | 1491.46 | 1499.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:00:00 | 1491.80 | 1491.46 | 1499.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1498.80 | 1492.44 | 1498.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 1495.80 | 1492.44 | 1498.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1505.50 | 1495.05 | 1499.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1505.50 | 1495.05 | 1499.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1508.40 | 1497.72 | 1500.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1508.40 | 1497.72 | 1500.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 1510.00 | 1502.34 | 1502.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 1513.30 | 1504.53 | 1503.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 12:15:00 | 1504.40 | 1504.50 | 1503.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 12:15:00 | 1504.40 | 1504.50 | 1503.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1504.40 | 1504.50 | 1503.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 1504.40 | 1504.50 | 1503.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 1500.00 | 1503.60 | 1502.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 1500.00 | 1503.60 | 1502.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 1504.20 | 1503.72 | 1502.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 1504.20 | 1503.72 | 1502.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1501.00 | 1503.18 | 1502.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 1496.10 | 1503.18 | 1502.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 1489.00 | 1500.34 | 1501.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1480.70 | 1489.60 | 1494.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1491.20 | 1485.51 | 1490.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 1491.20 | 1485.51 | 1490.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1491.20 | 1485.51 | 1490.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1491.20 | 1485.51 | 1490.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1495.90 | 1487.58 | 1490.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1488.40 | 1487.58 | 1490.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1497.00 | 1489.98 | 1491.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 1497.00 | 1489.98 | 1491.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 1503.70 | 1492.73 | 1492.37 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 1471.60 | 1488.44 | 1490.63 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 11:15:00 | 1488.90 | 1474.56 | 1473.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 12:15:00 | 1491.50 | 1477.95 | 1474.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 1478.00 | 1478.50 | 1475.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 14:30:00 | 1479.10 | 1478.50 | 1475.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1476.00 | 1478.00 | 1475.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1547.40 | 1478.00 | 1475.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1545.70 | 1556.71 | 1557.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1545.70 | 1556.71 | 1557.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1514.40 | 1538.51 | 1547.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1535.50 | 1525.21 | 1532.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 1535.50 | 1525.21 | 1532.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1535.50 | 1525.21 | 1532.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:45:00 | 1534.00 | 1525.21 | 1532.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1531.70 | 1526.51 | 1532.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 1521.70 | 1526.29 | 1530.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1540.50 | 1528.44 | 1531.08 | SL hit (close>static) qty=1.00 sl=1536.30 alert=retest2 |

### Cycle 100 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1554.00 | 1536.96 | 1534.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 1561.30 | 1541.83 | 1537.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1582.40 | 1582.93 | 1569.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1582.40 | 1582.93 | 1569.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1571.90 | 1580.13 | 1570.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1571.90 | 1580.13 | 1570.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1566.80 | 1577.46 | 1570.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 1564.50 | 1577.46 | 1570.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1568.80 | 1575.73 | 1570.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 1568.80 | 1575.73 | 1570.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1574.30 | 1582.13 | 1577.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1574.30 | 1582.13 | 1577.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1574.10 | 1580.53 | 1577.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:00:00 | 1577.30 | 1578.84 | 1577.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1573.50 | 1580.56 | 1581.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 1573.50 | 1580.56 | 1581.09 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 1586.40 | 1581.94 | 1581.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1596.50 | 1585.50 | 1583.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 11:15:00 | 1581.10 | 1585.93 | 1583.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 1581.10 | 1585.93 | 1583.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1581.10 | 1585.93 | 1583.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 1581.10 | 1585.93 | 1583.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1577.60 | 1584.27 | 1583.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1577.60 | 1584.27 | 1583.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 1575.00 | 1582.41 | 1582.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 1570.10 | 1578.07 | 1580.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 1575.60 | 1574.44 | 1577.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 1575.60 | 1574.44 | 1577.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1575.60 | 1574.44 | 1577.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 1576.40 | 1574.44 | 1577.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1572.00 | 1573.95 | 1576.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:30:00 | 1569.00 | 1572.82 | 1575.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:00:00 | 1569.90 | 1572.24 | 1575.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1584.00 | 1575.07 | 1575.65 | SL hit (close>static) qty=1.00 sl=1576.80 alert=retest2 |

### Cycle 104 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 1590.00 | 1578.05 | 1576.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1595.10 | 1583.58 | 1579.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 1608.20 | 1608.20 | 1598.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:45:00 | 1609.10 | 1608.20 | 1598.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1597.80 | 1605.03 | 1598.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:45:00 | 1598.80 | 1605.03 | 1598.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1604.40 | 1604.91 | 1599.47 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 1589.50 | 1596.43 | 1596.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 1587.30 | 1593.34 | 1595.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 1593.00 | 1592.17 | 1594.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 15:00:00 | 1593.00 | 1592.17 | 1594.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1594.00 | 1592.54 | 1594.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 1603.90 | 1592.54 | 1594.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1604.30 | 1594.89 | 1595.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 1603.50 | 1594.89 | 1595.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 11:15:00 | 1597.80 | 1595.89 | 1595.69 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 1589.60 | 1594.79 | 1595.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1580.40 | 1590.31 | 1592.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 1555.80 | 1553.68 | 1563.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:45:00 | 1555.00 | 1553.68 | 1563.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1488.70 | 1485.21 | 1491.35 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 1498.60 | 1493.43 | 1493.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 1500.00 | 1494.75 | 1493.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1500.30 | 1500.79 | 1497.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 1500.30 | 1500.79 | 1497.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1502.10 | 1501.22 | 1498.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:45:00 | 1504.20 | 1501.79 | 1498.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 1504.70 | 1502.19 | 1500.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 1493.20 | 1498.21 | 1498.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 1493.20 | 1498.21 | 1498.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 13:15:00 | 1484.30 | 1494.50 | 1496.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 10:15:00 | 1505.20 | 1494.01 | 1495.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 10:15:00 | 1505.20 | 1494.01 | 1495.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1505.20 | 1494.01 | 1495.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 1505.20 | 1494.01 | 1495.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 1519.00 | 1499.01 | 1497.68 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1477.70 | 1495.85 | 1497.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1468.50 | 1490.38 | 1494.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1461.00 | 1456.00 | 1467.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:45:00 | 1458.70 | 1456.00 | 1467.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1462.50 | 1459.03 | 1466.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 1462.50 | 1459.03 | 1466.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1465.60 | 1460.34 | 1466.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 1465.90 | 1460.34 | 1466.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1464.00 | 1461.07 | 1465.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 1465.60 | 1461.07 | 1465.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1460.60 | 1461.51 | 1465.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 1458.90 | 1461.51 | 1465.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 1472.90 | 1464.11 | 1465.20 | SL hit (close>static) qty=1.00 sl=1467.90 alert=retest2 |

### Cycle 112 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 1476.70 | 1466.63 | 1466.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 1481.30 | 1472.38 | 1469.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 1473.20 | 1480.38 | 1475.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 1473.20 | 1480.38 | 1475.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1473.20 | 1480.38 | 1475.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 13:45:00 | 1483.00 | 1477.23 | 1475.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 15:15:00 | 1489.00 | 1491.07 | 1491.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 15:15:00 | 1489.00 | 1491.07 | 1491.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 1472.10 | 1487.28 | 1489.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1492.80 | 1484.19 | 1486.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1492.80 | 1484.19 | 1486.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1492.80 | 1484.19 | 1486.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 1492.60 | 1484.19 | 1486.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 1504.80 | 1488.31 | 1487.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 1508.60 | 1499.06 | 1493.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 15:15:00 | 1504.00 | 1507.84 | 1502.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 15:15:00 | 1504.00 | 1507.84 | 1502.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 1504.00 | 1507.84 | 1502.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 1505.40 | 1507.84 | 1502.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1501.10 | 1506.49 | 1502.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:15:00 | 1497.90 | 1506.49 | 1502.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1497.10 | 1504.61 | 1501.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 1494.40 | 1504.61 | 1501.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1492.00 | 1502.09 | 1500.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 1490.90 | 1502.09 | 1500.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 1497.50 | 1500.85 | 1500.55 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1493.20 | 1499.32 | 1499.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1488.50 | 1495.83 | 1498.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 1495.20 | 1493.88 | 1496.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 1495.20 | 1493.88 | 1496.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1498.10 | 1494.64 | 1496.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 1498.10 | 1494.64 | 1496.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1498.00 | 1495.32 | 1496.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1491.00 | 1495.32 | 1496.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 1454.20 | 1452.70 | 1452.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 15:15:00 | 1454.20 | 1452.70 | 1452.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 1461.80 | 1454.52 | 1453.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1460.00 | 1462.89 | 1459.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 1460.00 | 1462.89 | 1459.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1460.00 | 1462.89 | 1459.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 1467.50 | 1463.81 | 1460.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 1467.50 | 1464.55 | 1460.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 1467.00 | 1464.92 | 1461.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:00:00 | 1468.00 | 1465.54 | 1461.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1456.40 | 1472.72 | 1469.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1456.40 | 1472.72 | 1469.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1469.20 | 1472.01 | 1469.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 1475.70 | 1472.65 | 1470.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1454.10 | 1468.43 | 1469.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 1454.10 | 1468.43 | 1469.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 1450.00 | 1459.25 | 1464.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 1451.00 | 1450.75 | 1456.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:00:00 | 1451.00 | 1450.75 | 1456.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1435.90 | 1440.15 | 1444.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 1442.80 | 1440.15 | 1444.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1440.50 | 1426.40 | 1431.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1440.50 | 1426.40 | 1431.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1436.40 | 1428.40 | 1431.55 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1439.30 | 1433.18 | 1433.17 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 1430.00 | 1433.63 | 1433.86 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 15:15:00 | 1434.60 | 1434.04 | 1434.02 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 1430.50 | 1433.33 | 1433.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1426.70 | 1432.01 | 1433.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 1435.60 | 1432.15 | 1432.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 12:15:00 | 1435.60 | 1432.15 | 1432.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 1435.60 | 1432.15 | 1432.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:00:00 | 1435.60 | 1432.15 | 1432.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 13:15:00 | 1439.30 | 1433.58 | 1433.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 1442.20 | 1435.30 | 1434.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 1433.10 | 1437.47 | 1435.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 1433.10 | 1437.47 | 1435.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1433.10 | 1437.47 | 1435.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 1433.10 | 1437.47 | 1435.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1436.20 | 1437.22 | 1435.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 1438.50 | 1438.05 | 1436.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 1441.40 | 1438.05 | 1436.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 1431.80 | 1436.60 | 1436.08 | SL hit (close<static) qty=1.00 sl=1432.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1428.00 | 1434.88 | 1435.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 1424.00 | 1432.10 | 1433.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 1425.10 | 1421.93 | 1425.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 13:15:00 | 1425.10 | 1421.93 | 1425.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1425.10 | 1421.93 | 1425.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 1425.10 | 1421.93 | 1425.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1420.80 | 1421.71 | 1425.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:45:00 | 1420.20 | 1421.71 | 1425.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1430.10 | 1422.95 | 1425.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1430.10 | 1422.95 | 1425.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1429.10 | 1424.18 | 1425.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1431.80 | 1424.18 | 1425.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1431.00 | 1426.67 | 1426.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 1433.90 | 1428.73 | 1427.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 1425.00 | 1428.83 | 1427.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1425.00 | 1428.83 | 1427.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1425.00 | 1428.83 | 1427.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 1425.00 | 1428.83 | 1427.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1429.70 | 1429.00 | 1428.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 1426.20 | 1429.00 | 1428.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1427.80 | 1429.08 | 1428.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 1429.90 | 1429.08 | 1428.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1427.50 | 1428.76 | 1428.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 14:15:00 | 1432.20 | 1428.76 | 1428.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 1434.80 | 1430.45 | 1429.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 1421.40 | 1429.61 | 1428.96 | SL hit (close<static) qty=1.00 sl=1426.50 alert=retest2 |

### Cycle 125 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1415.40 | 1426.77 | 1427.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 1403.50 | 1420.20 | 1424.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1412.90 | 1412.27 | 1417.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 1412.90 | 1412.27 | 1417.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1416.50 | 1413.12 | 1417.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 1419.10 | 1413.12 | 1417.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 1421.00 | 1414.69 | 1417.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 1421.00 | 1414.69 | 1417.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 1420.30 | 1415.82 | 1418.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 1420.30 | 1415.82 | 1418.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 1417.00 | 1416.05 | 1418.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 1422.70 | 1416.32 | 1417.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1414.90 | 1416.04 | 1417.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1407.70 | 1414.15 | 1416.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 1407.20 | 1412.12 | 1415.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 1406.70 | 1400.65 | 1404.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1409.90 | 1406.13 | 1405.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1409.90 | 1406.13 | 1405.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 1412.00 | 1408.05 | 1406.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 13:15:00 | 1412.70 | 1415.15 | 1412.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:00:00 | 1412.70 | 1415.15 | 1412.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1414.10 | 1414.94 | 1412.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 1413.30 | 1414.94 | 1412.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1411.10 | 1414.17 | 1412.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1404.60 | 1414.17 | 1412.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 1397.00 | 1410.74 | 1410.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 1396.60 | 1404.77 | 1407.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1402.60 | 1399.71 | 1403.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 1402.60 | 1399.71 | 1403.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1402.60 | 1399.71 | 1403.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 1404.60 | 1399.71 | 1403.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1410.90 | 1401.95 | 1404.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1410.90 | 1401.95 | 1404.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1407.00 | 1402.96 | 1404.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 1405.80 | 1402.61 | 1404.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:00:00 | 1405.50 | 1402.69 | 1403.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 1405.90 | 1403.66 | 1404.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 1411.80 | 1405.28 | 1404.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1411.80 | 1405.28 | 1404.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1416.20 | 1407.47 | 1405.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 1429.20 | 1429.56 | 1422.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:00:00 | 1429.20 | 1429.56 | 1422.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1429.70 | 1428.65 | 1423.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:15:00 | 1429.90 | 1428.65 | 1423.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1432.30 | 1427.83 | 1424.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:30:00 | 1430.00 | 1428.80 | 1425.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:00:00 | 1431.90 | 1428.80 | 1425.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1425.20 | 1428.17 | 1425.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 1425.50 | 1428.17 | 1425.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1424.00 | 1427.33 | 1425.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 1422.10 | 1427.33 | 1425.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1421.00 | 1426.07 | 1425.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 1421.00 | 1426.07 | 1425.21 | SL hit (close<static) qty=1.00 sl=1423.10 alert=retest2 |

### Cycle 129 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1418.40 | 1424.68 | 1425.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 15:15:00 | 1417.50 | 1423.24 | 1424.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1420.00 | 1413.29 | 1415.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 1420.00 | 1413.29 | 1415.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1420.00 | 1413.29 | 1415.85 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 1426.40 | 1417.56 | 1417.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1435.50 | 1422.52 | 1420.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1479.40 | 1487.82 | 1476.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 1479.40 | 1487.82 | 1476.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1476.90 | 1484.23 | 1476.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1476.90 | 1484.23 | 1476.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1482.50 | 1483.88 | 1477.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:30:00 | 1475.40 | 1483.88 | 1477.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 1479.60 | 1484.36 | 1478.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 1479.60 | 1484.36 | 1478.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1479.90 | 1483.47 | 1478.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 1491.80 | 1483.47 | 1478.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 1476.90 | 1484.96 | 1481.08 | SL hit (close<static) qty=1.00 sl=1478.10 alert=retest2 |

### Cycle 131 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 1463.80 | 1477.28 | 1478.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1447.70 | 1469.35 | 1474.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 1435.30 | 1431.93 | 1442.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:00:00 | 1435.30 | 1431.93 | 1442.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1435.20 | 1432.58 | 1441.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 1435.70 | 1432.58 | 1441.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1444.50 | 1436.53 | 1440.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1444.50 | 1436.53 | 1440.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1435.50 | 1436.33 | 1440.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 1442.00 | 1436.33 | 1440.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1427.00 | 1428.17 | 1433.89 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 14:15:00 | 1447.40 | 1436.70 | 1436.18 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 1405.00 | 1430.25 | 1433.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1383.60 | 1420.92 | 1428.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1289.60 | 1287.58 | 1302.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 1289.60 | 1287.58 | 1302.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1282.60 | 1278.56 | 1283.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 1280.40 | 1278.56 | 1283.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 13:15:00 | 1280.00 | 1279.95 | 1283.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 1292.70 | 1283.92 | 1284.51 | SL hit (close>static) qty=1.00 sl=1287.20 alert=retest2 |

### Cycle 134 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1294.20 | 1285.98 | 1285.39 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1281.00 | 1284.46 | 1284.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1277.50 | 1283.07 | 1284.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1279.90 | 1267.82 | 1273.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 1279.90 | 1267.82 | 1273.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1279.90 | 1267.82 | 1273.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1279.90 | 1267.82 | 1273.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1280.20 | 1270.30 | 1274.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1323.40 | 1270.30 | 1274.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1318.20 | 1279.88 | 1278.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1357.90 | 1317.59 | 1301.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1334.80 | 1336.54 | 1321.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1334.80 | 1336.54 | 1321.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1334.30 | 1342.62 | 1332.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 1332.00 | 1342.62 | 1332.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1340.80 | 1342.26 | 1333.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 1344.00 | 1342.26 | 1333.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 1408.40 | 1421.04 | 1421.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1408.40 | 1421.04 | 1421.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1401.20 | 1417.07 | 1420.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 1407.90 | 1407.58 | 1412.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:00:00 | 1407.90 | 1407.58 | 1412.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1412.60 | 1408.59 | 1412.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 1412.60 | 1408.59 | 1412.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1411.30 | 1409.13 | 1412.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 1409.30 | 1409.29 | 1411.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:45:00 | 1409.10 | 1409.13 | 1411.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:00:00 | 1408.10 | 1408.92 | 1411.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1407.90 | 1409.81 | 1411.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1408.10 | 1409.47 | 1410.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 1408.10 | 1409.47 | 1410.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1405.00 | 1408.58 | 1410.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:30:00 | 1412.40 | 1408.58 | 1410.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1400.10 | 1403.52 | 1407.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 1403.80 | 1403.52 | 1407.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1403.40 | 1403.50 | 1406.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 1416.00 | 1408.75 | 1407.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 1416.00 | 1408.75 | 1407.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 1421.40 | 1411.28 | 1409.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 1411.20 | 1413.07 | 1410.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:00:00 | 1411.20 | 1413.07 | 1410.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 1408.80 | 1412.22 | 1410.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:15:00 | 1407.40 | 1412.22 | 1410.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 1407.90 | 1411.36 | 1410.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:30:00 | 1407.40 | 1411.36 | 1410.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 1413.70 | 1412.25 | 1410.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 1407.00 | 1412.25 | 1410.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1398.90 | 1409.58 | 1409.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 1392.70 | 1402.59 | 1405.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 1338.70 | 1331.65 | 1348.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 1338.70 | 1331.65 | 1348.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1342.60 | 1336.76 | 1347.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 1345.90 | 1336.76 | 1347.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1348.10 | 1339.03 | 1347.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1348.10 | 1339.03 | 1347.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1349.70 | 1341.16 | 1347.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1355.00 | 1341.16 | 1347.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1359.70 | 1344.87 | 1348.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 1359.70 | 1344.87 | 1348.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1357.10 | 1347.32 | 1349.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 1356.40 | 1347.32 | 1349.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1355.60 | 1348.97 | 1349.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 1355.60 | 1348.97 | 1349.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 1360.40 | 1351.26 | 1350.91 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1323.40 | 1345.97 | 1348.71 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1379.00 | 1345.77 | 1344.20 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 1352.90 | 1356.56 | 1356.71 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 1360.10 | 1357.27 | 1357.02 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 1353.30 | 1356.47 | 1356.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 1347.60 | 1354.70 | 1355.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1300.80 | 1297.89 | 1312.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:15:00 | 1300.60 | 1297.89 | 1312.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1313.00 | 1304.07 | 1310.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 1313.00 | 1304.07 | 1310.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1319.20 | 1307.10 | 1311.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 1321.10 | 1307.10 | 1311.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1308.00 | 1307.28 | 1311.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1330.70 | 1307.28 | 1311.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1329.10 | 1311.64 | 1312.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1330.20 | 1311.64 | 1312.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1329.90 | 1315.29 | 1314.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1335.30 | 1321.62 | 1317.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1306.60 | 1324.14 | 1320.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1306.60 | 1324.14 | 1320.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1306.60 | 1324.14 | 1320.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1305.70 | 1324.14 | 1320.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1300.90 | 1319.49 | 1318.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1300.90 | 1319.49 | 1318.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1301.40 | 1315.87 | 1317.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1293.00 | 1306.56 | 1312.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1311.00 | 1306.54 | 1311.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1311.00 | 1306.54 | 1311.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1311.00 | 1306.54 | 1311.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 1311.00 | 1306.54 | 1311.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1299.80 | 1305.19 | 1310.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:00:00 | 1295.10 | 1303.17 | 1308.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 1230.34 | 1271.20 | 1289.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 1249.80 | 1238.11 | 1255.97 | SL hit (close>ema200) qty=0.50 sl=1238.11 alert=retest2 |

### Cycle 148 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1282.80 | 1258.81 | 1258.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 1290.00 | 1272.08 | 1265.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1247.90 | 1267.24 | 1264.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1247.90 | 1267.24 | 1264.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1247.90 | 1267.24 | 1264.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1247.90 | 1267.24 | 1264.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1237.50 | 1261.29 | 1261.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 1233.50 | 1251.91 | 1257.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1212.30 | 1204.36 | 1221.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1212.30 | 1204.36 | 1221.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1212.30 | 1204.36 | 1221.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1204.60 | 1204.36 | 1221.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1144.37 | 1184.59 | 1202.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1176.10 | 1173.06 | 1190.13 | SL hit (close>ema200) qty=0.50 sl=1173.06 alert=retest2 |

### Cycle 150 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 1205.00 | 1192.07 | 1191.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 1213.70 | 1198.43 | 1194.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 15:15:00 | 1245.00 | 1245.00 | 1229.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:15:00 | 1261.80 | 1245.00 | 1229.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1264.90 | 1272.14 | 1260.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 1274.30 | 1272.57 | 1261.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1283.50 | 1274.37 | 1267.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:45:00 | 1278.40 | 1275.30 | 1270.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 14:15:00 | 1401.73 | 1330.93 | 1321.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1264.50 | 1316.70 | 1316.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1258.80 | 1280.62 | 1296.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1250.30 | 1247.19 | 1265.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:45:00 | 1250.80 | 1247.19 | 1265.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1273.00 | 1252.35 | 1266.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1271.70 | 1252.35 | 1266.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1276.60 | 1257.20 | 1267.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:45:00 | 1279.50 | 1257.20 | 1267.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1272.50 | 1269.68 | 1270.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:45:00 | 1274.00 | 1269.68 | 1270.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1275.00 | 1270.75 | 1270.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 1275.00 | 1270.75 | 1270.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 1272.30 | 1271.06 | 1270.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1284.00 | 1273.91 | 1272.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 10:15:00 | 1273.90 | 1273.91 | 1272.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 11:00:00 | 1273.90 | 1273.91 | 1272.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 1270.70 | 1273.27 | 1272.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:45:00 | 1271.30 | 1273.27 | 1272.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 1264.40 | 1271.49 | 1271.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 1255.10 | 1268.22 | 1270.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1260.50 | 1249.17 | 1255.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1260.50 | 1249.17 | 1255.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1260.50 | 1249.17 | 1255.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 1260.50 | 1249.17 | 1255.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1257.50 | 1250.84 | 1255.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1244.30 | 1254.83 | 1256.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:30:00 | 1248.00 | 1247.57 | 1249.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 1249.10 | 1247.57 | 1249.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 1256.90 | 1250.70 | 1250.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 1256.90 | 1250.70 | 1250.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 1271.30 | 1254.82 | 1252.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 1264.50 | 1264.66 | 1259.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 11:00:00 | 1264.50 | 1264.66 | 1259.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1262.30 | 1264.06 | 1260.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 1262.30 | 1264.06 | 1260.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1260.80 | 1263.41 | 1260.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 1260.80 | 1263.41 | 1260.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1255.60 | 1261.85 | 1260.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 1255.60 | 1261.85 | 1260.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1253.00 | 1260.08 | 1259.40 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-24 10:15:00 | 1918.55 | 2024-05-30 13:15:00 | 1877.25 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-05-24 10:45:00 | 1911.80 | 2024-05-30 13:15:00 | 1877.25 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-05-24 11:30:00 | 1910.00 | 2024-05-30 13:15:00 | 1877.25 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-05-27 10:45:00 | 1919.10 | 2024-05-30 13:15:00 | 1877.25 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-05-29 09:15:00 | 1896.50 | 2024-05-30 13:15:00 | 1877.25 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-06-19 13:45:00 | 1819.80 | 2024-06-21 09:15:00 | 1832.95 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-06-19 14:30:00 | 1819.35 | 2024-06-21 09:15:00 | 1832.95 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-06-20 13:15:00 | 1818.50 | 2024-06-21 09:15:00 | 1832.95 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-06-26 10:15:00 | 1944.50 | 2024-06-27 09:15:00 | 1862.45 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2024-07-02 12:00:00 | 1820.60 | 2024-07-03 12:15:00 | 1859.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-07-02 12:30:00 | 1818.25 | 2024-07-03 12:15:00 | 1859.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-07-09 10:15:00 | 1922.45 | 2024-07-12 14:15:00 | 1907.70 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-07-09 12:00:00 | 1923.10 | 2024-07-12 14:15:00 | 1907.70 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-07-09 15:15:00 | 1925.00 | 2024-07-12 14:15:00 | 1907.70 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-07-10 12:30:00 | 1921.80 | 2024-07-12 14:15:00 | 1907.70 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-07-16 11:15:00 | 1887.00 | 2024-07-19 10:15:00 | 1792.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 12:30:00 | 1882.45 | 2024-07-19 10:15:00 | 1788.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 11:15:00 | 1887.00 | 2024-07-23 12:15:00 | 1698.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-16 12:30:00 | 1882.45 | 2024-07-23 12:15:00 | 1694.21 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-09-17 14:30:00 | 2008.15 | 2024-09-19 09:15:00 | 1973.20 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-09-17 15:00:00 | 2008.55 | 2024-09-19 09:15:00 | 1973.20 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-09-25 14:30:00 | 2053.90 | 2024-09-26 09:15:00 | 2027.65 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-10-31 10:15:00 | 1640.00 | 2024-11-06 10:15:00 | 1662.65 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-10-31 10:45:00 | 1639.95 | 2024-11-06 10:15:00 | 1662.65 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-10-31 12:30:00 | 1641.80 | 2024-11-06 10:15:00 | 1662.65 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-11-01 18:45:00 | 1646.25 | 2024-11-06 10:15:00 | 1662.65 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-11-07 14:15:00 | 1664.50 | 2024-11-11 13:15:00 | 1649.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-11-08 09:15:00 | 1664.75 | 2024-11-11 13:15:00 | 1649.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-11-08 13:00:00 | 1664.95 | 2024-11-11 13:15:00 | 1649.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-11-11 11:30:00 | 1663.40 | 2024-11-11 13:15:00 | 1649.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-11-14 14:15:00 | 1618.35 | 2024-11-19 09:15:00 | 1650.15 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-11-14 14:45:00 | 1620.40 | 2024-11-19 09:15:00 | 1650.15 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-11-18 14:15:00 | 1620.25 | 2024-11-19 09:15:00 | 1650.15 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-11-21 12:15:00 | 1632.25 | 2024-12-04 15:15:00 | 1730.00 | STOP_HIT | 1.00 | 5.99% |
| BUY | retest2 | 2024-11-21 13:45:00 | 1630.90 | 2024-12-04 15:15:00 | 1730.00 | STOP_HIT | 1.00 | 6.08% |
| BUY | retest2 | 2024-12-13 12:45:00 | 1759.30 | 2024-12-17 12:15:00 | 1736.80 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-12-13 14:45:00 | 1758.25 | 2024-12-17 12:15:00 | 1736.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-12-16 09:15:00 | 1761.30 | 2024-12-17 12:15:00 | 1736.80 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-12-17 10:00:00 | 1758.20 | 2024-12-17 12:15:00 | 1736.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-12-26 10:30:00 | 1664.50 | 2024-12-31 11:15:00 | 1671.90 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-12-26 13:15:00 | 1665.05 | 2024-12-31 11:15:00 | 1671.90 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-12-26 13:45:00 | 1666.55 | 2024-12-31 11:15:00 | 1671.90 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-12-27 10:45:00 | 1665.30 | 2024-12-31 11:15:00 | 1671.90 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-01-02 12:00:00 | 1680.85 | 2025-01-06 10:15:00 | 1652.40 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1614.85 | 2025-01-13 12:15:00 | 1534.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1614.85 | 2025-01-14 13:15:00 | 1540.00 | STOP_HIT | 0.50 | 4.64% |
| BUY | retest2 | 2025-01-31 10:30:00 | 1548.45 | 2025-02-03 09:15:00 | 1703.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-31 11:30:00 | 1550.50 | 2025-02-03 10:15:00 | 1705.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-01 12:00:00 | 1563.00 | 2025-02-05 14:15:00 | 1607.15 | STOP_HIT | 1.00 | 2.82% |
| SELL | retest2 | 2025-02-07 12:15:00 | 1605.50 | 2025-02-12 09:15:00 | 1525.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:45:00 | 1605.00 | 2025-02-12 09:15:00 | 1524.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:30:00 | 1606.75 | 2025-02-12 09:15:00 | 1526.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:15:00 | 1605.50 | 2025-02-14 09:15:00 | 1547.55 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-02-07 12:45:00 | 1605.00 | 2025-02-14 09:15:00 | 1547.55 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-02-10 10:30:00 | 1606.75 | 2025-02-14 09:15:00 | 1547.55 | STOP_HIT | 0.50 | 3.68% |
| BUY | retest2 | 2025-02-21 12:30:00 | 1520.90 | 2025-02-24 09:15:00 | 1503.95 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-02-21 14:45:00 | 1519.80 | 2025-02-24 09:15:00 | 1503.95 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-03-11 14:15:00 | 1475.95 | 2025-03-12 09:15:00 | 1460.90 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-03-11 14:45:00 | 1474.55 | 2025-03-12 09:15:00 | 1460.90 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-03-11 15:15:00 | 1475.00 | 2025-03-12 09:15:00 | 1460.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-05-05 15:15:00 | 1579.30 | 2025-05-12 11:15:00 | 1579.60 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-05-06 11:15:00 | 1580.00 | 2025-05-12 11:15:00 | 1579.60 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-05-06 13:30:00 | 1581.00 | 2025-05-12 11:15:00 | 1579.60 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-05-06 14:45:00 | 1579.10 | 2025-05-12 11:15:00 | 1579.60 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-05-08 10:30:00 | 1572.00 | 2025-05-12 11:15:00 | 1579.60 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-05-08 11:15:00 | 1572.10 | 2025-05-12 11:15:00 | 1579.60 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-05-21 11:45:00 | 1575.80 | 2025-05-23 15:15:00 | 1574.00 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-05-21 12:45:00 | 1577.90 | 2025-05-23 15:15:00 | 1574.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-05-21 13:15:00 | 1577.60 | 2025-05-23 15:15:00 | 1574.00 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-05-23 14:00:00 | 1576.70 | 2025-05-23 15:15:00 | 1574.00 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-05-27 13:45:00 | 1567.70 | 2025-06-05 11:15:00 | 1489.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-28 09:15:00 | 1567.80 | 2025-06-05 11:15:00 | 1489.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-27 13:45:00 | 1567.70 | 2025-06-06 09:15:00 | 1494.50 | STOP_HIT | 0.50 | 4.67% |
| SELL | retest2 | 2025-05-28 09:15:00 | 1567.80 | 2025-06-06 09:15:00 | 1494.50 | STOP_HIT | 0.50 | 4.68% |
| BUY | retest2 | 2025-06-26 09:15:00 | 1582.70 | 2025-06-26 09:15:00 | 1564.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-26 14:30:00 | 1575.90 | 2025-06-30 09:15:00 | 1544.40 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1586.40 | 2025-06-30 09:15:00 | 1544.40 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-07-01 10:45:00 | 1545.20 | 2025-07-03 09:15:00 | 1573.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-07-02 09:15:00 | 1546.10 | 2025-07-03 09:15:00 | 1573.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-07-02 14:15:00 | 1548.30 | 2025-07-03 09:15:00 | 1573.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-07-03 09:15:00 | 1548.90 | 2025-07-03 09:15:00 | 1573.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-07 10:15:00 | 1589.40 | 2025-07-07 12:15:00 | 1546.70 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-07-14 09:30:00 | 1519.90 | 2025-07-15 11:15:00 | 1533.70 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-14 12:15:00 | 1519.40 | 2025-07-15 11:15:00 | 1533.70 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-07-16 11:45:00 | 1530.10 | 2025-07-16 13:15:00 | 1525.40 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-07-16 12:15:00 | 1530.00 | 2025-07-16 13:15:00 | 1525.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1547.40 | 2025-08-26 09:15:00 | 1545.70 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-08-29 15:00:00 | 1521.70 | 2025-09-01 09:15:00 | 1540.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-09-05 13:00:00 | 1577.30 | 2025-09-09 10:15:00 | 1573.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-09-12 11:30:00 | 1569.00 | 2025-09-15 09:15:00 | 1584.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-09-12 13:00:00 | 1569.90 | 2025-09-15 09:15:00 | 1584.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-10-08 10:45:00 | 1504.20 | 2025-10-09 11:15:00 | 1493.20 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-10-08 15:15:00 | 1504.70 | 2025-10-09 11:15:00 | 1493.20 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-16 10:15:00 | 1458.90 | 2025-10-16 13:15:00 | 1472.90 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-20 13:45:00 | 1483.00 | 2025-10-27 15:15:00 | 1489.00 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1491.00 | 2025-11-12 15:15:00 | 1454.20 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2025-11-14 11:00:00 | 1467.50 | 2025-11-19 09:15:00 | 1454.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-14 12:00:00 | 1467.50 | 2025-11-19 09:15:00 | 1454.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-14 13:15:00 | 1467.00 | 2025-11-19 09:15:00 | 1454.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-14 14:00:00 | 1468.00 | 2025-11-19 09:15:00 | 1454.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-18 11:30:00 | 1475.70 | 2025-11-19 09:15:00 | 1454.10 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-12-01 13:30:00 | 1438.50 | 2025-12-01 15:15:00 | 1431.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-01 14:00:00 | 1441.40 | 2025-12-01 15:15:00 | 1431.80 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-12-05 14:15:00 | 1432.20 | 2025-12-08 09:15:00 | 1421.40 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-12-05 14:45:00 | 1434.80 | 2025-12-08 09:15:00 | 1421.40 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1407.70 | 2025-12-12 15:15:00 | 1409.90 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-12-10 13:30:00 | 1407.20 | 2025-12-12 15:15:00 | 1409.90 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-12-12 09:45:00 | 1406.70 | 2025-12-12 15:15:00 | 1409.90 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-12-18 13:30:00 | 1405.80 | 2025-12-19 13:15:00 | 1411.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-12-19 10:00:00 | 1405.50 | 2025-12-19 13:15:00 | 1411.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-12-19 12:45:00 | 1405.90 | 2025-12-19 13:15:00 | 1411.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-12-23 14:15:00 | 1429.90 | 2025-12-24 15:15:00 | 1421.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1432.30 | 2025-12-24 15:15:00 | 1421.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-12-24 10:30:00 | 1430.00 | 2025-12-24 15:15:00 | 1421.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-12-24 11:00:00 | 1431.90 | 2025-12-24 15:15:00 | 1421.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-12-26 09:15:00 | 1435.60 | 2025-12-26 14:15:00 | 1418.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1427.00 | 2025-12-26 14:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-26 12:45:00 | 1426.40 | 2025-12-26 14:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-01-09 09:15:00 | 1491.80 | 2026-01-09 11:15:00 | 1476.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-30 11:15:00 | 1280.40 | 2026-02-01 09:15:00 | 1292.70 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-01-30 13:15:00 | 1280.00 | 2026-02-01 09:15:00 | 1292.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-06 11:15:00 | 1344.00 | 2026-02-19 12:15:00 | 1408.40 | STOP_HIT | 1.00 | 4.79% |
| SELL | retest2 | 2026-02-23 10:45:00 | 1409.30 | 2026-02-25 15:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-02-23 12:45:00 | 1409.10 | 2026-02-25 15:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-02-23 14:00:00 | 1408.10 | 2026-02-25 15:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-02-24 09:30:00 | 1407.90 | 2026-02-25 15:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-03-20 12:00:00 | 1295.10 | 2026-03-23 10:15:00 | 1230.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:00:00 | 1295.10 | 2026-03-24 12:15:00 | 1249.80 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1204.60 | 2026-04-02 09:15:00 | 1144.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1204.60 | 2026-04-02 13:15:00 | 1176.10 | STOP_HIT | 0.50 | 2.37% |
| BUY | retest2 | 2026-04-13 11:00:00 | 1274.30 | 2026-04-22 14:15:00 | 1401.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1283.50 | 2026-04-22 14:15:00 | 1406.24 | TARGET_HIT | 1.00 | 9.56% |
| BUY | retest2 | 2026-04-15 13:45:00 | 1278.40 | 2026-04-23 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-05-05 09:15:00 | 1244.30 | 2026-05-06 15:15:00 | 1256.90 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-05-06 11:30:00 | 1248.00 | 2026-05-06 15:15:00 | 1256.90 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-05-06 12:15:00 | 1249.10 | 2026-05-06 15:15:00 | 1256.90 | STOP_HIT | 1.00 | -0.62% |
