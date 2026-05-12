# Cummins India Ltd. (CUMMINSIND)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 5391.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 238 |
| ALERT1 | 157 |
| ALERT2 | 153 |
| ALERT2_SKIP | 82 |
| ALERT3 | 458 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 198 |
| PARTIAL | 21 |
| TARGET_HIT | 31 |
| STOP_HIT | 172 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 219 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 86 / 133
- **Target hits / Stop hits / Partials:** 31 / 167 / 21
- **Avg / median % per leg:** 1.35% / -0.62%
- **Sum % (uncompounded):** 294.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 116 | 39 | 33.6% | 22 | 94 | 0 | 1.15% | 133.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 116 | 39 | 33.6% | 22 | 94 | 0 | 1.15% | 133.0% |
| SELL (all) | 103 | 47 | 45.6% | 9 | 73 | 21 | 1.57% | 161.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 103 | 47 | 45.6% | 9 | 73 | 21 | 1.57% | 161.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 219 | 86 | 39.3% | 31 | 167 | 21 | 1.35% | 294.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 1618.05 | 1643.94 | 1645.37 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 1658.80 | 1643.77 | 1642.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 13:15:00 | 1665.65 | 1651.94 | 1647.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 14:15:00 | 1615.10 | 1673.57 | 1671.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 14:15:00 | 1615.10 | 1673.57 | 1671.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 14:15:00 | 1615.10 | 1673.57 | 1671.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 15:00:00 | 1615.10 | 1673.57 | 1671.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 15:15:00 | 1591.95 | 1657.24 | 1664.07 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 1709.70 | 1655.56 | 1655.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 1715.20 | 1667.48 | 1660.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 10:15:00 | 1744.30 | 1748.61 | 1734.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 11:00:00 | 1744.30 | 1748.61 | 1734.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 1751.00 | 1757.06 | 1750.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 1756.90 | 1757.06 | 1750.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 11:30:00 | 1752.20 | 1753.88 | 1750.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 13:15:00 | 1744.50 | 1750.78 | 1749.52 | SL hit (close<static) qty=1.00 sl=1745.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 1764.15 | 1783.00 | 1783.77 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 14:15:00 | 1786.15 | 1780.24 | 1779.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 15:15:00 | 1790.00 | 1782.19 | 1780.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 09:15:00 | 1846.70 | 1848.89 | 1831.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 09:45:00 | 1839.60 | 1848.89 | 1831.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 1837.00 | 1845.49 | 1834.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:30:00 | 1835.00 | 1845.49 | 1834.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 1833.70 | 1843.13 | 1834.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:00:00 | 1833.70 | 1843.13 | 1834.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 1830.35 | 1840.57 | 1833.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:45:00 | 1828.40 | 1840.57 | 1833.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 1832.00 | 1838.86 | 1833.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 09:15:00 | 1833.30 | 1838.86 | 1833.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 1828.90 | 1836.87 | 1833.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 10:00:00 | 1828.90 | 1836.87 | 1833.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 1833.00 | 1836.09 | 1833.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 11:30:00 | 1839.75 | 1836.10 | 1833.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 12:15:00 | 1837.70 | 1836.10 | 1833.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 12:45:00 | 1839.00 | 1837.50 | 1834.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 15:15:00 | 1855.10 | 1868.73 | 1869.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 15:15:00 | 1855.10 | 1868.73 | 1869.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 09:15:00 | 1844.70 | 1863.93 | 1867.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 14:15:00 | 1845.00 | 1837.18 | 1845.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 14:15:00 | 1845.00 | 1837.18 | 1845.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 14:15:00 | 1845.00 | 1837.18 | 1845.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 15:00:00 | 1845.00 | 1837.18 | 1845.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 15:15:00 | 1854.00 | 1840.55 | 1846.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:15:00 | 1855.90 | 1840.55 | 1846.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 1854.00 | 1843.24 | 1847.02 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 13:15:00 | 1858.10 | 1849.10 | 1848.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 14:15:00 | 1869.40 | 1853.16 | 1850.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 09:15:00 | 1875.20 | 1876.15 | 1867.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 10:00:00 | 1875.20 | 1876.15 | 1867.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 1912.45 | 1925.43 | 1912.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 12:00:00 | 1912.45 | 1925.43 | 1912.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 12:15:00 | 1910.45 | 1922.44 | 1912.12 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 15:15:00 | 1888.65 | 1906.42 | 1906.55 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 09:15:00 | 1923.95 | 1909.92 | 1908.13 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 11:15:00 | 1885.45 | 1903.04 | 1905.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 12:15:00 | 1869.95 | 1896.42 | 1901.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 11:15:00 | 1883.30 | 1876.45 | 1887.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-05 11:30:00 | 1880.00 | 1876.45 | 1887.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 1897.00 | 1878.68 | 1883.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 10:00:00 | 1897.00 | 1878.68 | 1883.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 10:15:00 | 1916.10 | 1886.16 | 1886.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 11:00:00 | 1916.10 | 1886.16 | 1886.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 11:15:00 | 1908.10 | 1890.55 | 1888.55 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 13:15:00 | 1893.75 | 1897.38 | 1897.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 15:15:00 | 1888.55 | 1895.48 | 1896.86 | Break + close below crossover candle low |

### Cycle 14 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 1915.50 | 1899.49 | 1898.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 10:15:00 | 1924.50 | 1904.49 | 1900.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 12:15:00 | 1898.55 | 1904.50 | 1901.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 12:15:00 | 1898.55 | 1904.50 | 1901.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 1898.55 | 1904.50 | 1901.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 13:00:00 | 1898.55 | 1904.50 | 1901.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 13:15:00 | 1891.25 | 1901.85 | 1900.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 13:30:00 | 1895.00 | 1901.85 | 1900.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2023-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 14:15:00 | 1887.85 | 1899.05 | 1899.51 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 1931.50 | 1903.98 | 1901.57 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 12:15:00 | 1892.55 | 1903.08 | 1903.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 1881.95 | 1898.85 | 1901.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-13 15:15:00 | 1899.00 | 1897.63 | 1900.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 09:15:00 | 1908.80 | 1897.63 | 1900.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 1913.00 | 1900.71 | 1901.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:30:00 | 1916.75 | 1900.71 | 1901.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 1897.50 | 1900.07 | 1901.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-14 11:30:00 | 1895.50 | 1900.05 | 1901.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 12:15:00 | 1915.95 | 1903.23 | 1902.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 12:15:00 | 1915.95 | 1903.23 | 1902.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 15:15:00 | 1920.40 | 1908.38 | 1905.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 10:15:00 | 1907.70 | 1910.86 | 1906.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 10:15:00 | 1907.70 | 1910.86 | 1906.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 1907.70 | 1910.86 | 1906.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:30:00 | 1907.00 | 1910.86 | 1906.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 1906.55 | 1910.00 | 1906.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:45:00 | 1905.50 | 1910.00 | 1906.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 1905.00 | 1909.00 | 1906.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 12:45:00 | 1901.00 | 1909.00 | 1906.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 1903.95 | 1907.99 | 1906.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:30:00 | 1904.15 | 1907.99 | 1906.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 1899.05 | 1906.20 | 1905.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 15:00:00 | 1899.05 | 1906.20 | 1905.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 15:15:00 | 1898.05 | 1904.57 | 1905.07 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 09:15:00 | 1919.10 | 1907.48 | 1906.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 13:15:00 | 1924.05 | 1916.87 | 1912.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 10:15:00 | 1911.15 | 1923.49 | 1917.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 10:15:00 | 1911.15 | 1923.49 | 1917.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 1911.15 | 1923.49 | 1917.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:45:00 | 1911.95 | 1923.49 | 1917.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 1911.30 | 1921.06 | 1917.33 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-07-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 13:15:00 | 1900.75 | 1914.44 | 1914.81 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 15:15:00 | 1917.20 | 1915.39 | 1915.20 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 09:15:00 | 1888.60 | 1910.03 | 1912.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 10:15:00 | 1873.15 | 1902.65 | 1909.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 1904.90 | 1891.33 | 1898.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 1904.90 | 1891.33 | 1898.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1904.90 | 1891.33 | 1898.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:45:00 | 1904.45 | 1891.33 | 1898.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 1902.85 | 1893.63 | 1899.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 10:30:00 | 1912.35 | 1893.63 | 1899.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 1905.60 | 1895.61 | 1899.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 13:00:00 | 1905.60 | 1895.61 | 1899.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 1905.90 | 1897.67 | 1899.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 13:45:00 | 1905.15 | 1897.67 | 1899.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 1900.60 | 1899.08 | 1900.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:15:00 | 1906.55 | 1899.08 | 1900.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 10:15:00 | 1905.25 | 1901.50 | 1901.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 11:15:00 | 1915.20 | 1904.24 | 1902.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 09:15:00 | 1906.90 | 1911.76 | 1907.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 1906.90 | 1911.76 | 1907.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 1906.90 | 1911.76 | 1907.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 09:45:00 | 1907.45 | 1911.76 | 1907.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 1909.80 | 1911.37 | 1907.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:00:00 | 1909.80 | 1911.37 | 1907.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 1916.65 | 1912.43 | 1908.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:30:00 | 1909.20 | 1912.43 | 1908.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 1925.45 | 1918.49 | 1913.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 10:15:00 | 1932.80 | 1918.49 | 1913.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 09:45:00 | 1939.00 | 1925.34 | 1919.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 15:15:00 | 1925.00 | 1945.86 | 1947.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 15:15:00 | 1925.00 | 1945.86 | 1947.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 09:15:00 | 1892.25 | 1935.14 | 1942.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 1907.75 | 1906.55 | 1919.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 11:00:00 | 1907.75 | 1906.55 | 1919.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 1779.00 | 1868.68 | 1893.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 09:15:00 | 1732.40 | 1774.19 | 1811.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 11:15:00 | 1740.30 | 1762.60 | 1799.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 10:45:00 | 1744.00 | 1743.91 | 1769.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 12:15:00 | 1744.30 | 1744.42 | 1767.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 1749.90 | 1747.35 | 1760.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 10:30:00 | 1739.40 | 1749.45 | 1760.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 11:15:00 | 1762.40 | 1752.04 | 1760.45 | SL hit (close>static) qty=1.00 sl=1760.75 alert=retest2 |

### Cycle 26 — BUY (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 12:15:00 | 1748.55 | 1744.41 | 1744.13 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 13:15:00 | 1741.75 | 1743.88 | 1743.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 15:15:00 | 1739.40 | 1742.95 | 1743.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 10:15:00 | 1745.25 | 1742.57 | 1743.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 10:15:00 | 1745.25 | 1742.57 | 1743.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 1745.25 | 1742.57 | 1743.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 11:00:00 | 1745.25 | 1742.57 | 1743.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 1736.05 | 1741.26 | 1742.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 11:30:00 | 1744.60 | 1741.26 | 1742.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 1747.35 | 1737.39 | 1739.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:45:00 | 1744.25 | 1737.39 | 1739.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 1746.20 | 1739.15 | 1740.08 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 11:15:00 | 1752.25 | 1741.77 | 1741.19 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 15:15:00 | 1742.00 | 1747.18 | 1747.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 13:15:00 | 1731.90 | 1740.53 | 1743.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 1726.10 | 1720.27 | 1727.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 1726.10 | 1720.27 | 1727.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 1726.10 | 1720.27 | 1727.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:30:00 | 1725.25 | 1720.27 | 1727.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 1730.75 | 1722.36 | 1727.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:00:00 | 1730.75 | 1722.36 | 1727.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 1736.20 | 1725.13 | 1728.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:45:00 | 1738.05 | 1725.13 | 1728.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 1736.65 | 1727.43 | 1729.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 12:45:00 | 1737.30 | 1727.43 | 1729.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 1713.90 | 1724.51 | 1727.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 14:45:00 | 1727.05 | 1724.51 | 1727.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 1732.00 | 1724.21 | 1726.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:45:00 | 1730.00 | 1724.21 | 1726.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 1734.65 | 1726.30 | 1727.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 11:00:00 | 1734.65 | 1726.30 | 1727.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 12:15:00 | 1732.50 | 1728.65 | 1728.47 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 15:15:00 | 1725.10 | 1728.19 | 1728.31 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 09:15:00 | 1740.30 | 1730.61 | 1729.40 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 13:15:00 | 1717.80 | 1728.33 | 1729.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 1703.10 | 1723.28 | 1727.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 14:15:00 | 1705.90 | 1700.46 | 1711.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 15:00:00 | 1705.90 | 1700.46 | 1711.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 1710.00 | 1703.09 | 1710.44 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 12:15:00 | 1738.30 | 1716.40 | 1715.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 14:15:00 | 1747.25 | 1733.22 | 1727.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 14:15:00 | 1740.00 | 1743.69 | 1737.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 14:15:00 | 1740.00 | 1743.69 | 1737.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 14:15:00 | 1740.00 | 1743.69 | 1737.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 15:00:00 | 1740.00 | 1743.69 | 1737.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 1742.35 | 1743.42 | 1737.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 09:30:00 | 1746.35 | 1743.32 | 1738.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 10:00:00 | 1742.90 | 1743.32 | 1738.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 1729.35 | 1756.73 | 1752.88 | SL hit (close<static) qty=1.00 sl=1736.40 alert=retest2 |

### Cycle 35 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1709.55 | 1743.32 | 1747.21 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 1734.45 | 1724.88 | 1724.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 14:15:00 | 1746.45 | 1733.06 | 1728.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 09:15:00 | 1734.00 | 1735.00 | 1730.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 1734.00 | 1735.00 | 1730.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 1734.00 | 1735.00 | 1730.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:30:00 | 1735.00 | 1735.00 | 1730.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 1729.80 | 1734.02 | 1730.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 11:30:00 | 1725.60 | 1734.02 | 1730.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 1726.00 | 1732.42 | 1730.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:00:00 | 1726.00 | 1732.42 | 1730.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 1721.80 | 1730.29 | 1729.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:45:00 | 1720.25 | 1730.29 | 1729.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 15:15:00 | 1727.00 | 1728.94 | 1729.06 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 09:15:00 | 1749.00 | 1732.95 | 1730.87 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 15:15:00 | 1728.00 | 1730.73 | 1730.77 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 09:15:00 | 1731.20 | 1730.82 | 1730.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-21 14:15:00 | 1746.65 | 1738.01 | 1734.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-22 09:15:00 | 1719.00 | 1735.18 | 1734.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 1719.00 | 1735.18 | 1734.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 1719.00 | 1735.18 | 1734.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:00:00 | 1719.00 | 1735.18 | 1734.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 1727.00 | 1733.54 | 1733.38 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 11:15:00 | 1731.65 | 1733.16 | 1733.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 09:15:00 | 1681.10 | 1719.76 | 1726.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 12:15:00 | 1717.50 | 1713.27 | 1721.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-25 13:00:00 | 1717.50 | 1713.27 | 1721.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 1738.25 | 1718.93 | 1722.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 15:00:00 | 1738.25 | 1718.93 | 1722.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 1731.20 | 1721.38 | 1723.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:15:00 | 1733.00 | 1721.38 | 1723.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 1740.60 | 1726.71 | 1725.55 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 09:15:00 | 1716.20 | 1727.33 | 1727.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 10:15:00 | 1703.60 | 1722.59 | 1725.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 1696.00 | 1693.84 | 1707.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 10:00:00 | 1696.00 | 1693.84 | 1707.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 1704.05 | 1698.44 | 1705.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 13:45:00 | 1705.70 | 1698.44 | 1705.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 1695.05 | 1697.76 | 1704.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 14:30:00 | 1701.70 | 1697.76 | 1704.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1676.70 | 1694.07 | 1701.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 14:45:00 | 1673.30 | 1684.13 | 1693.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 09:15:00 | 1700.00 | 1692.68 | 1692.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 1700.00 | 1692.68 | 1692.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 10:15:00 | 1704.85 | 1695.11 | 1693.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 1696.55 | 1699.48 | 1697.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 1696.55 | 1699.48 | 1697.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 1696.55 | 1699.48 | 1697.03 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 1686.95 | 1694.13 | 1694.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 13:15:00 | 1684.10 | 1692.12 | 1693.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-09 14:15:00 | 1697.40 | 1693.18 | 1694.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 14:15:00 | 1697.40 | 1693.18 | 1694.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 14:15:00 | 1697.40 | 1693.18 | 1694.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 14:45:00 | 1705.10 | 1693.18 | 1694.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 15:15:00 | 1692.90 | 1693.12 | 1694.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:15:00 | 1696.65 | 1693.12 | 1694.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1700.50 | 1694.60 | 1694.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:30:00 | 1699.15 | 1694.60 | 1694.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 1708.45 | 1697.37 | 1695.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 15:15:00 | 1711.00 | 1703.14 | 1699.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 09:15:00 | 1734.90 | 1739.78 | 1725.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 10:00:00 | 1734.90 | 1739.78 | 1725.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 11:15:00 | 1725.05 | 1734.74 | 1725.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 11:30:00 | 1721.35 | 1734.74 | 1725.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 12:15:00 | 1725.20 | 1732.83 | 1725.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 15:00:00 | 1730.75 | 1731.32 | 1725.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-13 09:15:00 | 1715.50 | 1727.47 | 1725.07 | SL hit (close<static) qty=1.00 sl=1724.55 alert=retest2 |

### Cycle 47 — SELL (started 2023-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 11:15:00 | 1714.90 | 1722.64 | 1723.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 10:15:00 | 1708.95 | 1717.01 | 1719.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 09:15:00 | 1718.45 | 1714.43 | 1716.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 1718.45 | 1714.43 | 1716.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 1718.45 | 1714.43 | 1716.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 09:30:00 | 1719.80 | 1714.43 | 1716.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 1724.90 | 1716.52 | 1717.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:30:00 | 1722.10 | 1716.52 | 1717.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 1718.50 | 1716.92 | 1717.54 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 14:15:00 | 1722.90 | 1718.62 | 1718.20 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 1705.55 | 1716.55 | 1717.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 14:15:00 | 1693.55 | 1699.48 | 1705.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-23 13:15:00 | 1704.40 | 1693.28 | 1698.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 13:15:00 | 1704.40 | 1693.28 | 1698.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 13:15:00 | 1704.40 | 1693.28 | 1698.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 14:00:00 | 1704.40 | 1693.28 | 1698.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 14:15:00 | 1701.20 | 1694.86 | 1698.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 14:30:00 | 1702.70 | 1694.86 | 1698.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 15:15:00 | 1699.05 | 1695.70 | 1698.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 09:15:00 | 1690.65 | 1695.70 | 1698.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 1709.80 | 1698.52 | 1699.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 10:00:00 | 1709.80 | 1698.52 | 1699.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 1704.35 | 1699.69 | 1700.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 11:15:00 | 1700.45 | 1699.69 | 1700.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 14:45:00 | 1700.00 | 1698.43 | 1699.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 09:45:00 | 1703.00 | 1686.13 | 1689.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 10:15:00 | 1691.75 | 1677.78 | 1677.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 10:15:00 | 1691.75 | 1677.78 | 1677.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 11:15:00 | 1701.25 | 1682.48 | 1679.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 09:15:00 | 1703.95 | 1706.05 | 1698.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-06 09:45:00 | 1703.65 | 1706.05 | 1698.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 1726.05 | 1722.08 | 1712.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 10:15:00 | 1727.00 | 1722.08 | 1712.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-07 12:15:00 | 1699.35 | 1716.90 | 1712.46 | SL hit (close<static) qty=1.00 sl=1711.15 alert=retest2 |

### Cycle 51 — SELL (started 2023-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 15:15:00 | 1701.80 | 1709.54 | 1709.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 09:15:00 | 1692.20 | 1706.08 | 1708.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 10:15:00 | 1722.20 | 1709.30 | 1709.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 10:15:00 | 1722.20 | 1709.30 | 1709.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 1722.20 | 1709.30 | 1709.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 10:30:00 | 1771.95 | 1709.30 | 1709.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 11:15:00 | 1745.75 | 1716.59 | 1712.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 12:15:00 | 1760.55 | 1725.38 | 1717.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 10:15:00 | 1750.00 | 1750.51 | 1734.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-09 11:00:00 | 1750.00 | 1750.51 | 1734.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 1735.75 | 1747.06 | 1739.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:45:00 | 1734.00 | 1747.06 | 1739.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 1745.95 | 1746.84 | 1740.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:30:00 | 1747.35 | 1746.84 | 1740.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 1759.95 | 1761.12 | 1752.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 11:00:00 | 1759.95 | 1761.12 | 1752.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 1838.35 | 1834.35 | 1825.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:45:00 | 1833.80 | 1834.35 | 1825.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 1847.40 | 1862.78 | 1853.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 10:00:00 | 1847.40 | 1862.78 | 1853.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 1851.75 | 1860.57 | 1852.97 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2023-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 15:15:00 | 1846.00 | 1849.40 | 1849.48 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 09:15:00 | 1868.00 | 1853.12 | 1851.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 10:15:00 | 1870.85 | 1856.67 | 1852.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 09:15:00 | 1856.85 | 1865.82 | 1860.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 1856.85 | 1865.82 | 1860.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 1856.85 | 1865.82 | 1860.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:45:00 | 1858.25 | 1865.82 | 1860.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 1859.45 | 1864.54 | 1860.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 11:15:00 | 1865.45 | 1864.54 | 1860.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 13:15:00 | 1865.00 | 1866.03 | 1864.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 12:15:00 | 1939.00 | 1942.01 | 1942.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 12:15:00 | 1939.00 | 1942.01 | 1942.26 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-12-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 13:15:00 | 1948.60 | 1943.33 | 1942.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 14:15:00 | 1956.65 | 1945.99 | 1944.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 10:15:00 | 1939.45 | 1947.66 | 1945.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 10:15:00 | 1939.45 | 1947.66 | 1945.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 1939.45 | 1947.66 | 1945.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 1939.45 | 1947.66 | 1945.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 1940.20 | 1946.17 | 1945.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:30:00 | 1936.05 | 1946.17 | 1945.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 1936.70 | 1943.44 | 1943.99 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2023-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 14:15:00 | 1949.75 | 1944.70 | 1944.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 15:15:00 | 1953.00 | 1946.36 | 1945.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 1982.15 | 1984.77 | 1971.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 12:45:00 | 1984.70 | 1984.77 | 1971.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 13:15:00 | 1962.95 | 1980.40 | 1970.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:00:00 | 1962.95 | 1980.40 | 1970.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 1924.90 | 1969.30 | 1966.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:45:00 | 1932.90 | 1969.30 | 1966.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 1930.00 | 1961.44 | 1962.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 10:15:00 | 1917.50 | 1946.81 | 1955.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 13:15:00 | 1945.80 | 1940.64 | 1950.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 13:15:00 | 1945.80 | 1940.64 | 1950.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 1945.80 | 1940.64 | 1950.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 13:45:00 | 1947.00 | 1940.64 | 1950.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 1969.20 | 1946.35 | 1951.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 15:00:00 | 1969.20 | 1946.35 | 1951.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 1962.30 | 1949.54 | 1952.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 1955.30 | 1949.54 | 1952.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 15:15:00 | 1956.50 | 1946.98 | 1949.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-15 09:15:00 | 1966.80 | 1946.98 | 1949.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 1965.80 | 1950.74 | 1950.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 10:30:00 | 1946.70 | 1950.86 | 1950.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 11:15:00 | 1947.00 | 1950.86 | 1950.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 10:30:00 | 1948.40 | 1942.58 | 1945.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 11:15:00 | 1948.80 | 1942.58 | 1945.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 1950.65 | 1944.20 | 1945.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 12:00:00 | 1950.65 | 1944.20 | 1945.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 1955.70 | 1946.50 | 1946.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 13:00:00 | 1955.70 | 1946.50 | 1946.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-12-18 13:15:00 | 1959.30 | 1949.06 | 1947.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 13:15:00 | 1959.30 | 1949.06 | 1947.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 10:15:00 | 1969.50 | 1957.65 | 1952.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 12:15:00 | 1980.05 | 1983.67 | 1971.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 13:00:00 | 1980.05 | 1983.67 | 1971.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 1940.00 | 1974.93 | 1968.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:30:00 | 1941.10 | 1974.93 | 1968.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1905.50 | 1961.05 | 1962.96 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 1987.80 | 1954.63 | 1953.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 12:15:00 | 1989.10 | 1965.59 | 1959.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 09:15:00 | 2012.15 | 2016.43 | 1999.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 09:45:00 | 2014.20 | 2016.43 | 1999.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 1996.85 | 2010.81 | 2000.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 1999.10 | 2010.81 | 2000.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 1999.15 | 2008.48 | 2000.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 1996.00 | 2008.48 | 2000.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 1989.85 | 2004.76 | 1999.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 15:00:00 | 1989.85 | 2004.76 | 1999.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 15:15:00 | 1990.00 | 2001.80 | 1998.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 09:15:00 | 2003.40 | 2001.80 | 1998.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 10:15:00 | 1982.75 | 1997.22 | 1997.20 | SL hit (close<static) qty=1.00 sl=1982.80 alert=retest2 |

### Cycle 63 — SELL (started 2023-12-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 11:15:00 | 1969.75 | 1991.72 | 1994.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 14:15:00 | 1936.20 | 1974.77 | 1985.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 09:15:00 | 1971.50 | 1969.38 | 1981.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-29 10:00:00 | 1971.50 | 1969.38 | 1981.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 1961.85 | 1962.82 | 1971.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:00:00 | 1961.85 | 1962.82 | 1971.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 12:15:00 | 1958.95 | 1952.66 | 1959.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-02 12:30:00 | 1955.50 | 1952.66 | 1959.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 13:15:00 | 1962.00 | 1954.53 | 1959.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-02 14:00:00 | 1962.00 | 1954.53 | 1959.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 14:15:00 | 1961.70 | 1955.96 | 1959.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-02 15:00:00 | 1961.70 | 1955.96 | 1959.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 15:15:00 | 1959.50 | 1956.67 | 1959.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:15:00 | 1959.20 | 1956.67 | 1959.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 1957.30 | 1956.80 | 1959.49 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 12:15:00 | 1965.60 | 1960.76 | 1960.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 1991.90 | 1971.55 | 1966.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 1990.55 | 1995.19 | 1987.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 1990.55 | 1995.19 | 1987.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 1990.55 | 1995.19 | 1987.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 10:00:00 | 1990.55 | 1995.19 | 1987.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 1975.10 | 1991.17 | 1986.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 1975.10 | 1991.17 | 1986.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 1996.85 | 1992.31 | 1987.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 12:15:00 | 1998.75 | 1992.31 | 1987.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 12:15:00 | 2014.55 | 2041.20 | 2042.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 2014.55 | 2041.20 | 2042.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 13:15:00 | 2007.75 | 2034.51 | 2039.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 12:15:00 | 2019.40 | 2017.86 | 2027.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-17 13:00:00 | 2019.40 | 2017.86 | 2027.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 1960.70 | 2004.23 | 2018.18 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 09:15:00 | 2070.45 | 2029.94 | 2024.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 10:15:00 | 2083.00 | 2040.55 | 2030.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 2080.25 | 2086.27 | 2062.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 10:00:00 | 2080.25 | 2086.27 | 2062.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 2096.65 | 2096.16 | 2086.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 10:15:00 | 2113.15 | 2096.16 | 2086.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 12:15:00 | 2113.35 | 2098.45 | 2089.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 12:45:00 | 2112.00 | 2102.80 | 2092.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-02 14:15:00 | 2323.20 | 2274.93 | 2258.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 13:15:00 | 2539.30 | 2551.16 | 2552.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 12:15:00 | 2525.90 | 2538.72 | 2545.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 13:15:00 | 2549.35 | 2540.84 | 2545.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 13:15:00 | 2549.35 | 2540.84 | 2545.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 2549.35 | 2540.84 | 2545.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 2549.35 | 2540.84 | 2545.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 2560.00 | 2544.68 | 2546.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 2560.00 | 2544.68 | 2546.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 2570.05 | 2549.75 | 2548.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 10:15:00 | 2595.00 | 2563.67 | 2555.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 11:15:00 | 2608.05 | 2612.69 | 2592.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 12:00:00 | 2608.05 | 2612.69 | 2592.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 2599.60 | 2610.08 | 2592.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 12:45:00 | 2595.50 | 2610.08 | 2592.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 2628.95 | 2634.79 | 2625.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:45:00 | 2623.35 | 2634.79 | 2625.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 2631.10 | 2633.68 | 2626.25 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 2613.55 | 2622.01 | 2622.82 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-02-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 12:15:00 | 2680.95 | 2632.31 | 2626.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 13:15:00 | 2726.20 | 2651.09 | 2635.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 2722.35 | 2769.61 | 2729.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 2722.35 | 2769.61 | 2729.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 2722.35 | 2769.61 | 2729.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 10:00:00 | 2722.35 | 2769.61 | 2729.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 2672.70 | 2750.23 | 2723.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:00:00 | 2672.70 | 2750.23 | 2723.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 2670.35 | 2734.25 | 2719.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:45:00 | 2655.20 | 2734.25 | 2719.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 2718.55 | 2725.35 | 2718.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 09:15:00 | 2765.65 | 2725.35 | 2718.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 2780.40 | 2736.36 | 2724.34 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 15:15:00 | 2706.00 | 2731.25 | 2734.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 09:15:00 | 2683.95 | 2721.79 | 2729.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 2748.95 | 2714.39 | 2721.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 2748.95 | 2714.39 | 2721.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 2748.95 | 2714.39 | 2721.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 2748.95 | 2714.39 | 2721.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 2754.00 | 2722.31 | 2724.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 2731.40 | 2722.31 | 2724.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 09:15:00 | 2750.00 | 2727.85 | 2726.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 14:15:00 | 2783.95 | 2752.13 | 2741.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 15:15:00 | 2768.05 | 2771.07 | 2760.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-06 09:15:00 | 2768.45 | 2771.07 | 2760.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 2727.65 | 2762.38 | 2757.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 2727.65 | 2762.38 | 2757.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 2750.50 | 2760.01 | 2756.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 11:30:00 | 2758.40 | 2759.37 | 2756.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 12:00:00 | 2756.85 | 2759.37 | 2756.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 12:15:00 | 2764.90 | 2789.81 | 2791.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 12:15:00 | 2764.90 | 2789.81 | 2791.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 10:15:00 | 2741.90 | 2765.68 | 2777.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 2714.00 | 2706.31 | 2735.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:45:00 | 2715.00 | 2706.31 | 2735.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 2745.00 | 2714.05 | 2736.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 2745.00 | 2714.05 | 2736.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 2724.60 | 2716.16 | 2735.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 13:30:00 | 2700.90 | 2712.54 | 2731.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:15:00 | 2714.70 | 2717.54 | 2730.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 14:15:00 | 2715.85 | 2713.04 | 2722.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 15:00:00 | 2704.10 | 2711.25 | 2720.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 2698.60 | 2707.32 | 2717.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:30:00 | 2741.75 | 2707.32 | 2717.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 11:15:00 | 2719.15 | 2704.53 | 2714.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:00:00 | 2719.15 | 2704.53 | 2714.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 2701.85 | 2703.99 | 2712.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:30:00 | 2706.70 | 2703.99 | 2712.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 2708.70 | 2704.93 | 2712.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 14:00:00 | 2708.70 | 2704.93 | 2712.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 2721.00 | 2708.15 | 2713.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 15:00:00 | 2721.00 | 2708.15 | 2713.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 15:15:00 | 2726.55 | 2711.83 | 2714.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 09:15:00 | 2728.65 | 2711.83 | 2714.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 2681.60 | 2705.78 | 2711.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 2741.10 | 2715.15 | 2713.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 09:15:00 | 2741.10 | 2715.15 | 2713.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 2837.15 | 2783.55 | 2754.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 12:15:00 | 2961.55 | 2970.48 | 2933.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 12:45:00 | 2966.35 | 2970.48 | 2933.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 2971.10 | 2969.14 | 2944.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:45:00 | 2963.65 | 2969.14 | 2944.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 2971.00 | 2969.51 | 2946.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:30:00 | 2957.60 | 2969.51 | 2946.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 3007.75 | 3005.42 | 2986.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-01 14:45:00 | 2983.90 | 3005.42 | 2986.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 3008.85 | 3005.74 | 2990.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 11:45:00 | 3034.65 | 3010.57 | 2995.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 12:15:00 | 3036.25 | 3010.57 | 2995.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 13:15:00 | 3028.15 | 3013.65 | 2997.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 14:15:00 | 3028.40 | 3016.32 | 3000.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 3008.45 | 3018.74 | 3006.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 11:30:00 | 3028.70 | 3022.19 | 3009.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 10:00:00 | 3021.05 | 3033.82 | 3021.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 11:30:00 | 3019.45 | 3029.29 | 3021.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 10:15:00 | 3020.00 | 3026.61 | 3023.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 3029.00 | 3027.08 | 3023.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 14:00:00 | 3034.90 | 3026.48 | 3024.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-08 10:15:00 | 2954.25 | 3010.70 | 3017.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 10:15:00 | 2954.25 | 3010.70 | 3017.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 2943.70 | 2972.47 | 2989.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 2995.65 | 2966.93 | 2981.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 2995.65 | 2966.93 | 2981.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 2995.65 | 2966.93 | 2981.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:00:00 | 2995.65 | 2966.93 | 2981.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 2985.05 | 2970.55 | 2981.91 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 15:15:00 | 3006.25 | 2990.43 | 2988.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 09:15:00 | 3086.55 | 3009.65 | 2997.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 11:15:00 | 3030.30 | 3044.54 | 3028.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-15 12:00:00 | 3030.30 | 3044.54 | 3028.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 3039.00 | 3043.43 | 3029.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 14:00:00 | 3045.40 | 3043.83 | 3031.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 09:30:00 | 3050.30 | 3047.68 | 3036.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 11:00:00 | 3043.25 | 3098.24 | 3091.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 14:15:00 | 3041.40 | 3079.23 | 3084.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-04-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 14:15:00 | 3041.40 | 3079.23 | 3084.07 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 3116.70 | 3089.75 | 3087.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 12:15:00 | 3144.50 | 3100.70 | 3092.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 15:15:00 | 3230.00 | 3235.97 | 3212.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 10:00:00 | 3230.00 | 3234.78 | 3213.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 3217.80 | 3231.38 | 3214.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:30:00 | 3211.95 | 3231.38 | 3214.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 3251.35 | 3235.38 | 3217.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 11:45:00 | 3231.25 | 3235.38 | 3217.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 3312.70 | 3296.25 | 3267.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:30:00 | 3288.50 | 3296.25 | 3267.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 3271.90 | 3294.57 | 3278.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 3271.90 | 3294.57 | 3278.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 3270.00 | 3289.66 | 3277.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 3265.60 | 3289.66 | 3277.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 3276.65 | 3287.06 | 3277.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:45:00 | 3249.50 | 3287.06 | 3277.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 3296.30 | 3288.90 | 3279.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 11:45:00 | 3307.05 | 3291.10 | 3281.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 12:15:00 | 3310.40 | 3291.10 | 3281.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 15:15:00 | 3310.00 | 3302.22 | 3289.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:45:00 | 3391.80 | 3327.02 | 3303.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 3377.00 | 3399.67 | 3378.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:00:00 | 3377.00 | 3399.67 | 3378.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 3359.60 | 3391.66 | 3376.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:30:00 | 3357.80 | 3391.66 | 3376.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 3335.60 | 3380.45 | 3373.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 13:00:00 | 3335.60 | 3380.45 | 3373.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 3384.80 | 3378.07 | 3373.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:15:00 | 3392.00 | 3378.07 | 3373.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 3383.85 | 3379.23 | 3374.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:00:00 | 3417.25 | 3386.51 | 3378.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 11:00:00 | 3416.30 | 3415.88 | 3400.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 13:00:00 | 3406.95 | 3415.54 | 3402.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:15:00 | 3413.00 | 3408.54 | 3402.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 3395.00 | 3405.83 | 3401.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:00:00 | 3395.00 | 3405.83 | 3401.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 3395.70 | 3403.81 | 3401.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:30:00 | 3394.00 | 3403.81 | 3401.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 3412.25 | 3405.50 | 3402.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:30:00 | 3398.95 | 3405.50 | 3402.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 3394.00 | 3403.20 | 3401.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:00:00 | 3394.00 | 3403.20 | 3401.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 3411.90 | 3404.94 | 3402.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 3442.75 | 3407.60 | 3404.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 10:00:00 | 3426.35 | 3411.35 | 3406.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-15 10:15:00 | 3637.76 | 3574.88 | 3520.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 3702.05 | 3746.01 | 3746.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 11:15:00 | 3692.45 | 3735.30 | 3741.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 3741.05 | 3725.26 | 3734.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 14:15:00 | 3741.05 | 3725.26 | 3734.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 3741.05 | 3725.26 | 3734.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 3741.05 | 3725.26 | 3734.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 3724.35 | 3725.08 | 3733.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 3769.70 | 3725.08 | 3733.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 3754.80 | 3731.02 | 3735.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:30:00 | 3718.00 | 3732.06 | 3735.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 12:15:00 | 3755.45 | 3740.47 | 3739.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 3755.45 | 3740.47 | 3739.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 3773.50 | 3747.08 | 3742.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 3747.75 | 3750.47 | 3745.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 3747.75 | 3750.47 | 3745.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 3747.75 | 3750.47 | 3745.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:45:00 | 3748.50 | 3750.47 | 3745.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 3734.25 | 3747.22 | 3744.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:45:00 | 3742.00 | 3747.22 | 3744.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 3723.75 | 3742.53 | 3742.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:45:00 | 3722.20 | 3742.53 | 3742.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 12:15:00 | 3729.85 | 3739.99 | 3741.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 3709.15 | 3732.73 | 3737.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 09:15:00 | 3773.05 | 3738.28 | 3739.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 3773.05 | 3738.28 | 3739.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 3773.05 | 3738.28 | 3739.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 3773.05 | 3738.28 | 3739.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 10:15:00 | 3755.90 | 3741.80 | 3740.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 11:15:00 | 3828.05 | 3759.05 | 3748.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 09:15:00 | 3766.05 | 3787.33 | 3770.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 3766.05 | 3787.33 | 3770.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 3766.05 | 3787.33 | 3770.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 3766.05 | 3787.33 | 3770.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 3779.35 | 3785.74 | 3771.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:15:00 | 3756.30 | 3785.74 | 3771.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 3749.05 | 3778.40 | 3769.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:30:00 | 3758.65 | 3778.40 | 3769.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 3777.00 | 3778.12 | 3769.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 3752.90 | 3778.12 | 3769.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 3772.65 | 3777.03 | 3770.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:45:00 | 3774.50 | 3777.03 | 3770.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 3826.55 | 3786.93 | 3775.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:00:00 | 3843.95 | 3809.37 | 3791.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 3739.35 | 3805.16 | 3796.74 | SL hit (close<static) qty=1.00 sl=3769.75 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 10:15:00 | 3677.65 | 3779.66 | 3785.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 3612.00 | 3746.13 | 3770.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 3693.25 | 3588.11 | 3633.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 3693.25 | 3588.11 | 3633.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 3693.25 | 3588.11 | 3633.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 3693.25 | 3588.11 | 3633.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 3690.90 | 3608.67 | 3638.84 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 14:15:00 | 3711.75 | 3665.15 | 3659.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 15:15:00 | 3723.00 | 3676.72 | 3665.20 | Break + close above crossover candle high |

### Cycle 85 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 3566.10 | 3654.59 | 3656.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 3363.90 | 3596.46 | 3629.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 3352.70 | 3326.66 | 3419.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 3352.70 | 3326.66 | 3419.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 3411.15 | 3343.56 | 3418.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 3417.85 | 3343.56 | 3418.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 3510.60 | 3381.05 | 3423.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 3510.60 | 3381.05 | 3423.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 3529.65 | 3410.77 | 3432.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 3529.65 | 3410.77 | 3432.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 3496.75 | 3447.84 | 3447.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 3552.55 | 3494.97 | 3471.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 3542.60 | 3549.84 | 3515.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 09:30:00 | 3545.30 | 3549.84 | 3515.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 3523.05 | 3539.88 | 3516.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 12:30:00 | 3552.10 | 3537.83 | 3517.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 3569.60 | 3539.11 | 3523.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-19 09:15:00 | 3907.31 | 3816.05 | 3778.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 15:15:00 | 3962.00 | 4003.04 | 4006.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 11:15:00 | 3951.80 | 3984.72 | 3996.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 3977.15 | 3975.48 | 3988.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 14:15:00 | 3977.15 | 3975.48 | 3988.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 3977.15 | 3975.48 | 3988.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:00:00 | 3977.15 | 3975.48 | 3988.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 3999.00 | 3980.19 | 3989.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 4019.80 | 3980.19 | 3989.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 4006.40 | 3985.43 | 3991.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 10:30:00 | 3980.00 | 3979.86 | 3988.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 4040.85 | 3961.09 | 3959.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 4040.85 | 3961.09 | 3959.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 10:15:00 | 4083.25 | 3985.52 | 3970.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 4086.40 | 4100.82 | 4064.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 09:45:00 | 4084.90 | 4100.82 | 4064.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 4015.25 | 4083.71 | 4059.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 4015.25 | 4083.71 | 4059.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 3974.70 | 4061.91 | 4052.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 3974.70 | 4061.91 | 4052.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 3988.90 | 4037.39 | 4042.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 3961.75 | 3987.80 | 4007.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 11:15:00 | 3994.70 | 3984.55 | 4002.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 11:15:00 | 3994.70 | 3984.55 | 4002.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 3994.70 | 3984.55 | 4002.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:45:00 | 3998.00 | 3984.55 | 4002.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 4013.90 | 3990.42 | 4003.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:30:00 | 4018.10 | 3990.42 | 4003.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 4008.65 | 3994.06 | 4003.62 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 4031.00 | 4011.57 | 4010.15 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 11:15:00 | 3991.25 | 4008.04 | 4008.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 13:15:00 | 3977.55 | 3999.85 | 4004.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 14:15:00 | 4001.50 | 4000.18 | 4004.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 15:00:00 | 4001.50 | 4000.18 | 4004.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 3987.00 | 3997.55 | 4002.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 3984.50 | 3997.55 | 4002.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:15:00 | 3785.27 | 3869.84 | 3900.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-19 11:15:00 | 3586.05 | 3722.52 | 3797.64 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 92 — BUY (started 2024-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 14:15:00 | 3613.35 | 3588.92 | 3587.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 3675.55 | 3610.26 | 3597.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 3824.05 | 3835.33 | 3802.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 10:00:00 | 3824.05 | 3835.33 | 3802.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 3813.00 | 3830.87 | 3803.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 3801.75 | 3830.87 | 3803.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 3812.50 | 3827.19 | 3803.89 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 3550.70 | 3766.87 | 3784.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 3451.35 | 3542.69 | 3635.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3523.00 | 3507.07 | 3571.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 3523.00 | 3507.07 | 3571.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3523.00 | 3507.07 | 3571.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 3482.50 | 3526.04 | 3556.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 3743.15 | 3569.22 | 3568.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 3743.15 | 3569.22 | 3568.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 3789.85 | 3613.35 | 3588.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 13:15:00 | 3751.35 | 3753.55 | 3713.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:30:00 | 3739.00 | 3753.55 | 3713.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 3721.25 | 3747.09 | 3714.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 3721.25 | 3747.09 | 3714.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 3733.00 | 3744.27 | 3715.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 3725.15 | 3744.27 | 3715.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 3680.10 | 3731.44 | 3712.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 13:00:00 | 3748.65 | 3719.33 | 3710.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 3668.85 | 3706.66 | 3711.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 3668.85 | 3706.66 | 3711.20 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 13:15:00 | 3721.45 | 3710.40 | 3709.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 10:15:00 | 3742.15 | 3723.82 | 3716.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 3741.00 | 3747.04 | 3734.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 3741.00 | 3747.04 | 3734.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 3741.00 | 3747.04 | 3734.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 3741.00 | 3747.04 | 3734.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 3757.70 | 3749.17 | 3736.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 11:30:00 | 3790.00 | 3751.43 | 3738.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 15:15:00 | 3768.05 | 3749.30 | 3740.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 3732.50 | 3745.75 | 3741.82 | SL hit (close<static) qty=1.00 sl=3734.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 11:15:00 | 3790.65 | 3833.01 | 3833.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 13:15:00 | 3782.15 | 3820.96 | 3827.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 3733.15 | 3732.07 | 3757.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 3733.15 | 3732.07 | 3757.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 3745.70 | 3734.56 | 3753.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 3755.05 | 3734.56 | 3753.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 3750.50 | 3737.75 | 3753.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 3750.50 | 3737.75 | 3753.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 3740.00 | 3735.43 | 3746.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 3774.60 | 3735.43 | 3746.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 3756.85 | 3739.71 | 3747.18 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 10:15:00 | 3804.85 | 3752.74 | 3752.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 15:15:00 | 3812.00 | 3785.21 | 3770.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 10:15:00 | 3831.70 | 3846.92 | 3828.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 10:15:00 | 3831.70 | 3846.92 | 3828.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 3831.70 | 3846.92 | 3828.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:45:00 | 3837.50 | 3846.92 | 3828.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 3846.90 | 3846.92 | 3830.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:30:00 | 3857.75 | 3845.13 | 3835.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 12:15:00 | 3805.50 | 3831.28 | 3831.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 12:15:00 | 3805.50 | 3831.28 | 3831.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 3763.00 | 3812.92 | 3822.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 12:15:00 | 3718.50 | 3707.72 | 3739.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 13:00:00 | 3718.50 | 3707.72 | 3739.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 3757.55 | 3717.68 | 3740.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:00:00 | 3757.55 | 3717.68 | 3740.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 3746.90 | 3723.53 | 3741.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:30:00 | 3757.20 | 3723.53 | 3741.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 3744.40 | 3727.70 | 3741.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 3779.80 | 3727.70 | 3741.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 3792.00 | 3740.56 | 3746.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 3792.00 | 3740.56 | 3746.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 3813.80 | 3755.21 | 3752.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 3848.75 | 3782.14 | 3765.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 3789.60 | 3790.11 | 3772.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 15:00:00 | 3789.60 | 3790.11 | 3772.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 3770.85 | 3786.57 | 3773.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:45:00 | 3762.40 | 3786.57 | 3773.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 3760.95 | 3781.45 | 3772.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:45:00 | 3756.35 | 3781.45 | 3772.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 3760.30 | 3774.20 | 3770.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:30:00 | 3763.70 | 3774.20 | 3770.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 3808.80 | 3799.13 | 3786.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:45:00 | 3824.95 | 3801.11 | 3788.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:30:00 | 3823.30 | 3796.31 | 3790.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 10:45:00 | 3821.25 | 3801.85 | 3793.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:15:00 | 3828.10 | 3808.94 | 3801.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 3786.40 | 3822.31 | 3816.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 3786.40 | 3822.31 | 3816.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-18 13:15:00 | 3770.25 | 3811.90 | 3812.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 3770.25 | 3811.90 | 3812.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 14:15:00 | 3759.35 | 3801.39 | 3807.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 3719.50 | 3702.02 | 3745.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 3719.50 | 3702.02 | 3745.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 3774.80 | 3721.74 | 3744.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 3774.80 | 3721.74 | 3744.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 3756.85 | 3728.76 | 3745.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 12:15:00 | 3750.85 | 3728.76 | 3745.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 3733.70 | 3741.73 | 3748.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 3752.00 | 3741.73 | 3748.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 3816.00 | 3756.58 | 3754.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 3816.00 | 3756.58 | 3754.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 15:15:00 | 3820.00 | 3769.27 | 3760.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 3849.90 | 3859.08 | 3832.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 3849.90 | 3859.08 | 3832.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 3865.25 | 3859.67 | 3837.67 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 3768.05 | 3827.43 | 3828.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 3733.05 | 3790.42 | 3809.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 3797.35 | 3777.97 | 3794.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 3797.35 | 3777.97 | 3794.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 3797.35 | 3777.97 | 3794.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 3797.35 | 3777.97 | 3794.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 3789.95 | 3780.37 | 3793.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 3795.95 | 3780.37 | 3793.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 3837.00 | 3791.69 | 3797.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 3837.00 | 3791.69 | 3797.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 3848.65 | 3803.09 | 3802.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 11:15:00 | 3860.00 | 3814.47 | 3807.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 3834.95 | 3837.17 | 3823.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 3834.95 | 3837.17 | 3823.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 3834.95 | 3837.17 | 3823.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 3802.80 | 3837.17 | 3823.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 3806.85 | 3831.10 | 3821.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:45:00 | 3801.25 | 3831.10 | 3821.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 3821.70 | 3829.22 | 3821.75 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 14:15:00 | 3801.85 | 3814.63 | 3816.14 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 3835.05 | 3818.61 | 3817.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 12:15:00 | 3858.00 | 3828.48 | 3822.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 3810.70 | 3844.93 | 3834.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 3810.70 | 3844.93 | 3834.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 3810.70 | 3844.93 | 3834.27 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 3727.95 | 3811.45 | 3820.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 3709.00 | 3790.96 | 3810.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 3649.95 | 3614.17 | 3655.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 3649.95 | 3614.17 | 3655.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 3649.95 | 3614.17 | 3655.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 3662.25 | 3614.17 | 3655.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 3702.15 | 3631.77 | 3659.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 3702.15 | 3631.77 | 3659.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 3692.95 | 3644.00 | 3662.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 3699.85 | 3644.00 | 3662.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 3687.35 | 3658.67 | 3666.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 3692.10 | 3658.67 | 3666.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 3725.00 | 3680.15 | 3675.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 3747.75 | 3693.67 | 3681.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 3766.40 | 3770.31 | 3742.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 3766.40 | 3770.31 | 3742.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 3586.65 | 3739.70 | 3738.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 3586.65 | 3739.70 | 3738.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 3591.35 | 3710.03 | 3724.92 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 09:15:00 | 3759.45 | 3682.20 | 3673.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 10:15:00 | 3790.05 | 3703.77 | 3683.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 3745.30 | 3767.32 | 3732.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 3745.30 | 3767.32 | 3732.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 3745.30 | 3767.32 | 3732.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 3745.30 | 3767.32 | 3732.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 3733.00 | 3760.46 | 3732.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 3733.00 | 3760.46 | 3732.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 3723.40 | 3753.05 | 3731.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 3717.10 | 3753.05 | 3731.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 3725.80 | 3747.60 | 3730.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:45:00 | 3724.90 | 3747.60 | 3730.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 3707.85 | 3739.65 | 3728.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:45:00 | 3715.05 | 3739.65 | 3728.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 3695.05 | 3730.73 | 3725.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 3695.05 | 3730.73 | 3725.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 3687.00 | 3721.98 | 3722.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 3675.45 | 3712.68 | 3718.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 3702.60 | 3700.95 | 3709.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:45:00 | 3698.00 | 3700.95 | 3709.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 3709.70 | 3702.70 | 3709.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 3708.85 | 3702.70 | 3709.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 3711.80 | 3704.52 | 3709.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 3705.10 | 3704.52 | 3709.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 3683.75 | 3700.37 | 3707.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:45:00 | 3658.80 | 3693.99 | 3703.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 3475.86 | 3566.15 | 3611.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 3411.90 | 3376.74 | 3411.44 | SL hit (close>ema200) qty=0.50 sl=3376.74 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 3460.00 | 3423.36 | 3420.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 3493.05 | 3455.70 | 3438.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 3505.20 | 3510.33 | 3483.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:00:00 | 3505.20 | 3510.33 | 3483.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 3512.40 | 3510.74 | 3485.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:30:00 | 3498.65 | 3510.74 | 3485.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 3470.30 | 3502.66 | 3484.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 3471.50 | 3502.66 | 3484.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 3472.75 | 3496.67 | 3483.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:45:00 | 3500.50 | 3498.31 | 3485.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 3514.60 | 3501.15 | 3488.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:45:00 | 3499.95 | 3490.25 | 3488.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 14:15:00 | 3468.70 | 3485.94 | 3486.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 14:15:00 | 3468.70 | 3485.94 | 3486.37 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 15:15:00 | 3505.00 | 3489.75 | 3488.07 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 3414.55 | 3474.71 | 3481.38 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 3566.25 | 3489.33 | 3482.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 3626.30 | 3533.97 | 3505.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 3565.90 | 3574.96 | 3543.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:30:00 | 3550.20 | 3574.96 | 3543.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 3563.95 | 3572.96 | 3547.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 3563.95 | 3572.96 | 3547.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 3586.95 | 3575.76 | 3551.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 09:45:00 | 3589.95 | 3554.70 | 3547.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 10:15:00 | 3532.95 | 3550.35 | 3545.98 | SL hit (close<static) qty=1.00 sl=3544.40 alert=retest2 |

### Cycle 117 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 3493.00 | 3569.50 | 3572.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 3475.40 | 3529.75 | 3552.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 3349.50 | 3341.08 | 3380.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 11:30:00 | 3359.00 | 3341.08 | 3380.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 3383.85 | 3351.92 | 3370.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 3388.05 | 3351.92 | 3370.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 3369.70 | 3355.48 | 3370.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 3390.45 | 3355.48 | 3370.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 3359.95 | 3356.37 | 3369.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:30:00 | 3376.75 | 3356.37 | 3369.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 3365.90 | 3358.28 | 3369.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:00:00 | 3365.90 | 3358.28 | 3369.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 3350.00 | 3356.62 | 3367.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:45:00 | 3368.50 | 3356.62 | 3367.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 3308.25 | 3297.69 | 3313.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 3308.25 | 3297.69 | 3313.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 3332.40 | 3304.63 | 3315.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 3332.40 | 3304.63 | 3315.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 3314.45 | 3306.60 | 3315.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:45:00 | 3330.00 | 3306.60 | 3315.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 3327.00 | 3310.68 | 3316.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 3452.90 | 3310.68 | 3316.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 3485.00 | 3345.54 | 3331.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 3501.65 | 3376.76 | 3347.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 3487.55 | 3492.09 | 3439.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 3487.55 | 3492.09 | 3439.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 3474.00 | 3480.27 | 3453.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 3451.15 | 3480.27 | 3453.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 3448.30 | 3473.87 | 3452.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 3451.00 | 3473.87 | 3452.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 3486.30 | 3476.36 | 3455.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 3490.05 | 3476.36 | 3455.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:45:00 | 3496.75 | 3480.63 | 3461.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:45:00 | 3494.15 | 3485.09 | 3465.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:30:00 | 3490.05 | 3486.54 | 3469.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 3466.45 | 3482.35 | 3470.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 3466.45 | 3482.35 | 3470.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 3443.30 | 3474.54 | 3468.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-28 12:15:00 | 3443.30 | 3474.54 | 3468.03 | SL hit (close<static) qty=1.00 sl=3448.30 alert=retest2 |

### Cycle 119 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 3424.60 | 3462.86 | 3467.39 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 3531.65 | 3467.62 | 3463.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 10:15:00 | 3576.95 | 3519.92 | 3495.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 3497.45 | 3526.37 | 3510.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 3497.45 | 3526.37 | 3510.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 3497.45 | 3526.37 | 3510.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 3497.45 | 3526.37 | 3510.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 3449.10 | 3510.92 | 3505.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 3449.10 | 3510.92 | 3505.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 11:15:00 | 3459.95 | 3500.72 | 3501.15 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 3513.00 | 3486.46 | 3484.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 10:15:00 | 3540.00 | 3508.29 | 3497.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 12:15:00 | 3593.45 | 3603.90 | 3575.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 12:30:00 | 3591.00 | 3603.90 | 3575.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 3565.00 | 3598.45 | 3582.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 3548.90 | 3598.45 | 3582.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 3577.55 | 3594.27 | 3582.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 3567.00 | 3594.27 | 3582.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 3569.00 | 3589.22 | 3581.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:30:00 | 3546.55 | 3589.22 | 3581.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 3582.65 | 3588.12 | 3582.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 15:00:00 | 3601.50 | 3590.80 | 3583.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 13:15:00 | 3566.45 | 3618.34 | 3616.76 | SL hit (close<static) qty=1.00 sl=3576.35 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 3548.40 | 3604.36 | 3610.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 15:15:00 | 3520.00 | 3587.48 | 3602.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 3367.90 | 3352.80 | 3402.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 3367.90 | 3352.80 | 3402.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 3390.00 | 3364.14 | 3383.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:45:00 | 3393.95 | 3364.14 | 3383.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 3361.20 | 3363.55 | 3381.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:30:00 | 3390.00 | 3363.55 | 3381.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 3379.35 | 3367.90 | 3380.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 3379.35 | 3367.90 | 3380.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 3370.00 | 3368.32 | 3379.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:15:00 | 3337.10 | 3368.32 | 3379.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 3256.85 | 3346.03 | 3368.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:15:00 | 3245.00 | 3346.03 | 3368.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:15:00 | 3246.85 | 3295.23 | 3316.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 3242.15 | 3286.97 | 3310.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:00:00 | 3246.80 | 3278.93 | 3304.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 3287.00 | 3258.17 | 3276.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:30:00 | 3282.25 | 3258.17 | 3276.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 3305.05 | 3267.55 | 3279.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 3305.05 | 3267.55 | 3279.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 3239.35 | 3225.19 | 3241.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:45:00 | 3240.50 | 3225.19 | 3241.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 3252.45 | 3230.64 | 3242.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 3252.45 | 3230.64 | 3242.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 3265.00 | 3237.51 | 3244.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 3294.20 | 3237.51 | 3244.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 3219.15 | 3234.80 | 3241.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 12:15:00 | 3211.70 | 3234.80 | 3241.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 12:45:00 | 3213.45 | 3230.01 | 3238.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 09:30:00 | 3200.00 | 3219.75 | 3230.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 09:45:00 | 3173.15 | 3185.93 | 3204.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 3178.35 | 3172.89 | 3187.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 3130.45 | 3160.88 | 3179.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:30:00 | 3146.10 | 3157.89 | 3170.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 3138.50 | 3154.08 | 3167.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 3082.75 | 3110.66 | 3138.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 3084.51 | 3110.66 | 3138.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 3080.04 | 3110.66 | 3138.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 3084.46 | 3110.66 | 3138.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 3051.11 | 3110.66 | 3138.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 3052.78 | 3110.66 | 3138.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 3040.00 | 3110.66 | 3138.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 3014.49 | 3063.55 | 3105.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 2973.93 | 3025.53 | 3075.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 2988.79 | 3025.53 | 3075.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 2981.57 | 3025.53 | 3075.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 11:15:00 | 2920.50 | 2990.06 | 3049.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 11:15:00 | 2947.20 | 2926.50 | 2925.88 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 2925.00 | 2927.50 | 2927.66 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 2931.00 | 2928.20 | 2927.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 14:15:00 | 2945.00 | 2931.98 | 2929.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 2926.00 | 2932.60 | 2930.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 2926.00 | 2932.60 | 2930.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 2926.00 | 2932.60 | 2930.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 2926.00 | 2932.60 | 2930.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 2932.80 | 2932.64 | 2930.73 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 2905.10 | 2928.17 | 2929.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 2872.60 | 2912.71 | 2921.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 2885.20 | 2883.94 | 2902.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 2885.20 | 2883.94 | 2902.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2931.25 | 2893.49 | 2903.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 2931.25 | 2893.49 | 2903.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2911.00 | 2896.99 | 2903.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:00:00 | 2891.95 | 2898.28 | 2902.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 2747.35 | 2823.70 | 2858.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 12:15:00 | 2804.00 | 2800.81 | 2837.78 | SL hit (close>ema200) qty=0.50 sl=2800.81 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 2850.00 | 2804.06 | 2799.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 15:15:00 | 2865.00 | 2845.15 | 2829.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 2834.90 | 2908.47 | 2884.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 2834.90 | 2908.47 | 2884.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 2834.90 | 2908.47 | 2884.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 2834.90 | 2908.47 | 2884.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 2789.50 | 2884.67 | 2876.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 2801.00 | 2884.67 | 2876.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 2801.05 | 2867.95 | 2869.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 2737.40 | 2822.23 | 2846.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 2798.00 | 2748.68 | 2787.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 2798.00 | 2748.68 | 2787.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 2798.00 | 2748.68 | 2787.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 2798.00 | 2748.68 | 2787.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 2815.70 | 2762.08 | 2790.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 2815.70 | 2762.08 | 2790.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 2827.40 | 2775.15 | 2793.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 2827.40 | 2775.15 | 2793.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 2855.60 | 2810.74 | 2807.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 2936.60 | 2843.31 | 2822.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 2896.10 | 2937.07 | 2912.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 2896.10 | 2937.07 | 2912.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 2896.10 | 2937.07 | 2912.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 2896.10 | 2937.07 | 2912.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2873.75 | 2924.41 | 2908.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 2857.35 | 2924.41 | 2908.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 2849.10 | 2898.67 | 2899.89 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 09:15:00 | 2953.15 | 2902.22 | 2900.56 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 2885.35 | 2899.51 | 2900.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 2830.00 | 2882.21 | 2891.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 2821.95 | 2816.33 | 2842.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 2816.95 | 2816.33 | 2842.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2840.00 | 2818.23 | 2834.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:45:00 | 2791.20 | 2819.33 | 2830.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:30:00 | 2797.85 | 2815.27 | 2827.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 2799.00 | 2815.27 | 2827.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 2763.65 | 2813.41 | 2825.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2651.64 | 2724.33 | 2764.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2657.96 | 2724.33 | 2764.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2659.05 | 2724.33 | 2764.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 2718.00 | 2710.48 | 2743.90 | SL hit (close>ema200) qty=0.50 sl=2710.48 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 2783.00 | 2725.14 | 2720.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 11:15:00 | 2792.75 | 2738.66 | 2726.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 2812.35 | 2823.31 | 2793.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 2781.90 | 2823.31 | 2793.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 2808.15 | 2820.28 | 2794.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 2775.95 | 2820.28 | 2794.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 2803.20 | 2814.19 | 2796.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:30:00 | 2813.95 | 2814.19 | 2796.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 2806.15 | 2812.58 | 2797.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:30:00 | 2820.80 | 2815.52 | 2801.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 2793.40 | 2814.99 | 2804.84 | SL hit (close<static) qty=1.00 sl=2793.90 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 2779.55 | 2797.57 | 2798.49 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 2816.85 | 2800.36 | 2798.54 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 2749.85 | 2790.36 | 2794.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 2719.40 | 2769.11 | 2783.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 2768.10 | 2742.93 | 2758.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 11:15:00 | 2768.10 | 2742.93 | 2758.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 2768.10 | 2742.93 | 2758.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:00:00 | 2768.10 | 2742.93 | 2758.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 2790.00 | 2752.34 | 2761.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:45:00 | 2784.80 | 2752.34 | 2761.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 2826.75 | 2776.58 | 2771.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 11:15:00 | 2854.50 | 2809.25 | 2790.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 2929.85 | 2952.67 | 2918.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 2929.85 | 2952.67 | 2918.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 2910.40 | 2944.21 | 2917.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 2910.40 | 2944.21 | 2917.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 2899.35 | 2935.24 | 2915.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 2896.30 | 2935.24 | 2915.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 2899.25 | 2928.04 | 2914.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 2899.25 | 2928.04 | 2914.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 2910.00 | 2924.43 | 2913.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 2900.10 | 2924.43 | 2913.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 2904.30 | 2920.02 | 2913.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 2904.30 | 2920.02 | 2913.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 2866.70 | 2909.35 | 2909.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 2846.30 | 2882.34 | 2895.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 2866.95 | 2862.89 | 2879.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 2866.95 | 2862.89 | 2879.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 2872.55 | 2864.82 | 2878.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 2872.55 | 2864.82 | 2878.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 2876.85 | 2867.23 | 2878.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 2858.45 | 2867.23 | 2878.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2825.50 | 2858.88 | 2873.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 2817.50 | 2858.88 | 2873.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 14:15:00 | 2862.70 | 2850.68 | 2849.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 2862.70 | 2850.68 | 2849.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 2909.95 | 2863.66 | 2855.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 2958.40 | 2972.73 | 2938.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 2958.40 | 2972.73 | 2938.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 2967.00 | 2985.62 | 2968.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:30:00 | 2965.20 | 2985.62 | 2968.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 2954.85 | 2979.46 | 2967.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 2948.85 | 2979.46 | 2967.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2934.75 | 2970.52 | 2964.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2934.75 | 2970.52 | 2964.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2945.00 | 2965.42 | 2962.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 3011.90 | 2965.42 | 2962.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 2958.00 | 2992.16 | 2995.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 2958.00 | 2992.16 | 2995.97 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 10:15:00 | 3047.45 | 3006.94 | 3001.69 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 12:15:00 | 3010.90 | 3026.81 | 3027.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 2983.45 | 3010.03 | 3019.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 3036.05 | 3015.23 | 3020.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 3036.05 | 3015.23 | 3020.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 3036.05 | 3015.23 | 3020.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 3036.05 | 3015.23 | 3020.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 3032.60 | 3018.71 | 3021.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 3032.60 | 3018.71 | 3021.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 3044.15 | 3023.79 | 3023.69 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 13:15:00 | 3019.00 | 3022.84 | 3023.27 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 3049.90 | 3028.25 | 3025.69 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 3013.65 | 3023.99 | 3024.50 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 3030.80 | 3025.35 | 3025.07 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 12:15:00 | 3020.50 | 3024.38 | 3024.66 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 3033.95 | 3025.98 | 3025.32 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 15:15:00 | 3015.05 | 3023.79 | 3024.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 2915.00 | 3002.03 | 3014.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2754.10 | 2743.20 | 2818.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 2754.10 | 2743.20 | 2818.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2754.10 | 2743.20 | 2818.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 2746.05 | 2751.56 | 2815.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 2746.00 | 2756.59 | 2798.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:00:00 | 2747.05 | 2741.05 | 2776.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 2840.80 | 2792.08 | 2787.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 2840.80 | 2792.08 | 2787.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 2908.80 | 2824.38 | 2804.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 13:15:00 | 2894.20 | 2903.85 | 2873.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 14:00:00 | 2894.20 | 2903.85 | 2873.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 2844.10 | 2889.68 | 2874.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 2844.10 | 2889.68 | 2874.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 2869.40 | 2885.62 | 2873.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 2873.50 | 2885.62 | 2873.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 2872.70 | 2899.91 | 2900.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 2872.70 | 2899.91 | 2900.24 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 2946.00 | 2901.87 | 2899.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 2966.90 | 2914.88 | 2906.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 09:15:00 | 2917.70 | 2922.66 | 2911.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 2917.70 | 2922.66 | 2911.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 2917.70 | 2922.66 | 2911.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 2905.00 | 2922.66 | 2911.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 2910.00 | 2920.04 | 2912.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:30:00 | 2909.10 | 2920.04 | 2912.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 2912.40 | 2918.51 | 2912.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:30:00 | 2910.10 | 2918.51 | 2912.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 2925.40 | 2919.89 | 2913.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 14:45:00 | 2930.80 | 2923.25 | 2915.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 2863.90 | 2913.23 | 2912.50 | SL hit (close<static) qty=1.00 sl=2911.40 alert=retest2 |

### Cycle 155 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 2836.10 | 2897.80 | 2905.55 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 2922.30 | 2898.61 | 2895.84 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 2887.10 | 2901.53 | 2902.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 2881.00 | 2897.36 | 2900.49 | Break + close below crossover candle low |

### Cycle 158 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 2953.30 | 2908.55 | 2905.29 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 14:15:00 | 2892.50 | 2902.69 | 2903.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 15:15:00 | 2880.20 | 2898.19 | 2901.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 2916.40 | 2901.84 | 2903.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 2916.40 | 2901.84 | 2903.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 2916.40 | 2901.84 | 2903.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 2916.40 | 2901.84 | 2903.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 2933.00 | 2908.07 | 2905.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 2937.00 | 2913.85 | 2908.65 | Break + close above crossover candle high |

### Cycle 161 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 2785.00 | 2899.59 | 2905.78 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 2844.90 | 2800.87 | 2797.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 2878.50 | 2816.40 | 2804.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 2872.30 | 2878.68 | 2859.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 12:15:00 | 2872.30 | 2878.68 | 2859.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 2872.30 | 2878.68 | 2859.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 2860.60 | 2878.68 | 2859.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 2871.40 | 2874.77 | 2860.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 2871.10 | 2874.77 | 2860.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 2988.70 | 3014.68 | 2984.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 2988.70 | 3014.68 | 2984.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2951.80 | 3000.49 | 2982.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:15:00 | 2946.00 | 3000.49 | 2982.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2981.70 | 2996.73 | 2982.70 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 2950.90 | 2974.19 | 2975.54 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 2989.80 | 2976.91 | 2975.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 3006.80 | 2984.39 | 2979.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 2975.00 | 2982.51 | 2979.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 2975.00 | 2982.51 | 2979.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2975.00 | 2982.51 | 2979.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 2975.00 | 2982.51 | 2979.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 2972.60 | 2980.53 | 2978.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 2970.50 | 2980.53 | 2978.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 2971.00 | 2978.62 | 2977.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 2970.90 | 2978.62 | 2977.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 2970.50 | 2977.00 | 2977.12 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 2980.20 | 2977.67 | 2977.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 3000.00 | 2982.14 | 2979.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 10:15:00 | 2981.00 | 2981.91 | 2979.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 2981.00 | 2981.91 | 2979.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2981.00 | 2981.91 | 2979.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 2981.00 | 2981.91 | 2979.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 11:15:00 | 2962.00 | 2977.93 | 2978.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 12:15:00 | 2946.10 | 2971.56 | 2975.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 2962.80 | 2956.59 | 2965.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 2962.80 | 2956.59 | 2965.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2962.80 | 2956.59 | 2965.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 2968.40 | 2956.59 | 2965.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 2937.90 | 2952.85 | 2963.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:45:00 | 2924.00 | 2946.98 | 2959.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:45:00 | 2931.60 | 2942.08 | 2954.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 2927.90 | 2942.63 | 2952.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:45:00 | 2930.90 | 2942.82 | 2952.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2966.30 | 2947.52 | 2953.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 2966.30 | 2947.52 | 2953.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 2953.80 | 2948.77 | 2953.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 2958.20 | 2948.77 | 2953.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 2967.40 | 2952.50 | 2954.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 2967.40 | 2952.50 | 2954.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 2954.50 | 2952.90 | 2954.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:30:00 | 2962.30 | 2952.90 | 2954.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-27 14:15:00 | 2973.70 | 2957.06 | 2956.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 2973.70 | 2957.06 | 2956.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 15:15:00 | 2979.00 | 2961.45 | 2958.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 2985.50 | 2991.92 | 2978.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 2985.50 | 2991.92 | 2978.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 2960.00 | 2985.53 | 2976.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 3061.80 | 2985.53 | 2976.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-04 09:15:00 | 3367.98 | 3317.44 | 3280.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 3359.90 | 3381.43 | 3382.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 3347.30 | 3370.84 | 3377.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 3321.00 | 3314.95 | 3333.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 3318.90 | 3314.95 | 3333.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 3292.10 | 3310.38 | 3330.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 3268.70 | 3297.81 | 3309.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 3274.10 | 3290.14 | 3304.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 3273.50 | 3287.45 | 3301.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 3273.00 | 3281.96 | 3295.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 3295.90 | 3263.84 | 3275.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 3295.90 | 3263.84 | 3275.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 3286.00 | 3268.28 | 3276.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 3277.10 | 3268.28 | 3276.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:00:00 | 3280.00 | 3273.49 | 3277.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 3267.30 | 3272.25 | 3276.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 3322.60 | 3269.68 | 3268.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 3322.60 | 3269.68 | 3268.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 10:15:00 | 3339.10 | 3314.29 | 3301.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 3373.50 | 3377.32 | 3352.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 14:15:00 | 3376.00 | 3377.32 | 3352.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 3330.20 | 3367.89 | 3350.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 3330.20 | 3367.89 | 3350.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 3358.40 | 3365.99 | 3351.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 3384.00 | 3365.99 | 3351.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 3352.00 | 3369.45 | 3371.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 3352.00 | 3369.45 | 3371.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 3329.70 | 3361.50 | 3367.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3362.80 | 3346.71 | 3354.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 3362.80 | 3346.71 | 3354.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 3362.80 | 3346.71 | 3354.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 3357.80 | 3346.71 | 3354.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 3372.50 | 3351.87 | 3356.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 3372.50 | 3351.87 | 3356.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 3367.30 | 3359.49 | 3359.17 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 3330.80 | 3353.75 | 3356.59 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 3377.90 | 3353.39 | 3352.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 3403.90 | 3369.24 | 3359.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 3509.60 | 3513.26 | 3483.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:00:00 | 3509.60 | 3513.26 | 3483.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 3546.70 | 3523.26 | 3499.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 10:45:00 | 3555.00 | 3528.39 | 3504.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 11:30:00 | 3556.60 | 3533.17 | 3508.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 3557.70 | 3534.99 | 3511.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:30:00 | 3551.60 | 3538.00 | 3517.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 3506.80 | 3531.36 | 3519.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 3506.80 | 3531.36 | 3519.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 3487.30 | 3522.55 | 3516.39 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 3487.30 | 3522.55 | 3516.39 | SL hit (close<static) qty=1.00 sl=3496.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 3547.90 | 3561.08 | 3562.79 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 3586.70 | 3561.73 | 3559.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 13:15:00 | 3626.10 | 3574.61 | 3565.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 3563.90 | 3584.21 | 3573.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 3563.90 | 3584.21 | 3573.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3563.90 | 3584.21 | 3573.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 3563.90 | 3584.21 | 3573.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 3582.90 | 3583.95 | 3574.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 3598.00 | 3586.96 | 3576.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 3552.60 | 3581.54 | 3578.36 | SL hit (close<static) qty=1.00 sl=3562.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 3567.00 | 3588.11 | 3589.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 3555.80 | 3581.65 | 3586.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 3529.70 | 3525.44 | 3544.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:45:00 | 3531.30 | 3525.44 | 3544.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 3540.00 | 3528.35 | 3543.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 3566.60 | 3528.35 | 3543.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 3546.00 | 3531.88 | 3544.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 3555.00 | 3531.88 | 3544.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 3543.40 | 3534.19 | 3543.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:00:00 | 3515.50 | 3531.51 | 3538.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 12:15:00 | 3560.00 | 3532.42 | 3537.02 | SL hit (close>static) qty=1.00 sl=3551.80 alert=retest2 |

### Cycle 178 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 3582.20 | 3542.37 | 3541.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 10:15:00 | 3583.10 | 3563.52 | 3555.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 3563.10 | 3578.30 | 3566.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 3563.10 | 3578.30 | 3566.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 3563.10 | 3578.30 | 3566.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 3563.10 | 3578.30 | 3566.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 3559.20 | 3574.48 | 3565.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 3576.30 | 3574.48 | 3565.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 3572.60 | 3574.10 | 3566.30 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 3499.70 | 3552.58 | 3558.20 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 3569.70 | 3561.89 | 3561.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 15:15:00 | 3600.00 | 3573.89 | 3567.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 11:15:00 | 3588.40 | 3612.12 | 3598.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 11:15:00 | 3588.40 | 3612.12 | 3598.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 3588.40 | 3612.12 | 3598.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 3588.40 | 3612.12 | 3598.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 3602.40 | 3610.17 | 3599.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:30:00 | 3602.60 | 3610.17 | 3599.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 3616.80 | 3611.50 | 3600.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:30:00 | 3585.00 | 3611.50 | 3600.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 3760.00 | 3777.06 | 3760.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 3760.00 | 3777.06 | 3760.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 3750.50 | 3771.75 | 3759.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 3773.10 | 3771.75 | 3759.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:45:00 | 3771.80 | 3791.64 | 3790.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 3749.50 | 3783.21 | 3786.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 3749.50 | 3783.21 | 3786.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 11:15:00 | 3746.80 | 3775.93 | 3782.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 3776.30 | 3762.73 | 3772.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 3776.30 | 3762.73 | 3772.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 3776.30 | 3762.73 | 3772.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 3776.30 | 3762.73 | 3772.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 3789.90 | 3768.17 | 3773.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 3797.80 | 3768.17 | 3773.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 3823.70 | 3785.22 | 3780.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 3858.90 | 3810.08 | 3793.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 3884.90 | 3898.00 | 3870.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:00:00 | 3884.90 | 3898.00 | 3870.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 3901.60 | 3897.02 | 3874.46 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 3844.90 | 3869.24 | 3870.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 3841.10 | 3861.80 | 3866.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 3836.70 | 3836.22 | 3846.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 15:15:00 | 3836.70 | 3836.22 | 3846.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 3836.70 | 3836.22 | 3846.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 3851.00 | 3836.22 | 3846.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 3860.10 | 3841.00 | 3847.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 3858.70 | 3841.00 | 3847.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 3872.70 | 3847.34 | 3849.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 3872.70 | 3847.34 | 3849.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 3898.00 | 3857.47 | 3854.15 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 3823.20 | 3852.28 | 3852.93 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 3878.10 | 3854.51 | 3853.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 3899.30 | 3863.47 | 3857.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 3858.60 | 3872.45 | 3865.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 3858.60 | 3872.45 | 3865.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 3858.60 | 3872.45 | 3865.69 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 3837.60 | 3859.76 | 3861.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 14:15:00 | 3827.60 | 3853.33 | 3858.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 12:15:00 | 3860.00 | 3841.37 | 3849.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 12:15:00 | 3860.00 | 3841.37 | 3849.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 3860.00 | 3841.37 | 3849.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 3860.00 | 3841.37 | 3849.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 3867.10 | 3846.52 | 3850.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 3867.10 | 3846.52 | 3850.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 3886.40 | 3854.50 | 3853.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 15:15:00 | 3892.50 | 3862.10 | 3857.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 11:15:00 | 3915.50 | 3926.21 | 3905.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 12:00:00 | 3915.50 | 3926.21 | 3905.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 3919.10 | 3924.79 | 3906.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 3907.40 | 3924.79 | 3906.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 4020.00 | 4022.09 | 4002.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:30:00 | 4060.40 | 4040.67 | 4030.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 4059.30 | 4047.70 | 4035.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:15:00 | 4059.90 | 4047.70 | 4035.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 4060.00 | 4052.85 | 4042.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 4055.40 | 4053.36 | 4043.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 4078.80 | 4057.49 | 4047.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 4080.50 | 4098.18 | 4098.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 4080.50 | 4098.18 | 4098.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 4057.40 | 4087.67 | 4093.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 4016.60 | 4009.64 | 4032.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 4016.60 | 4009.64 | 4032.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 4040.00 | 4007.62 | 4019.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 4040.00 | 4007.62 | 4019.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 4013.50 | 4008.79 | 4018.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 4001.00 | 4004.91 | 4016.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 3954.20 | 3925.19 | 3922.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 3954.20 | 3925.19 | 3922.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 3965.00 | 3944.75 | 3937.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 3913.00 | 3940.41 | 3936.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 3913.00 | 3940.41 | 3936.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3913.00 | 3940.41 | 3936.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 3913.00 | 3940.41 | 3936.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 3888.00 | 3929.92 | 3932.26 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 3950.90 | 3930.25 | 3929.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 4010.70 | 3960.42 | 3946.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 3969.80 | 3972.50 | 3959.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 3969.80 | 3972.50 | 3959.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 3969.80 | 3972.50 | 3959.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 3960.00 | 3972.50 | 3959.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3947.60 | 3965.81 | 3958.36 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 3922.10 | 3953.13 | 3953.65 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 3980.00 | 3950.57 | 3949.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 3984.80 | 3960.99 | 3954.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 14:15:00 | 3958.60 | 3962.69 | 3957.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 3958.60 | 3962.69 | 3957.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 3958.60 | 3962.69 | 3957.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 3958.60 | 3962.69 | 3957.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 3958.00 | 3961.75 | 3957.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 3975.40 | 3961.75 | 3957.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 3946.50 | 3959.99 | 3958.02 | SL hit (close<static) qty=1.00 sl=3948.70 alert=retest2 |

### Cycle 195 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 3941.90 | 3954.28 | 3955.78 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 3984.60 | 3958.08 | 3957.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 4003.40 | 3967.14 | 3961.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 3971.10 | 3974.77 | 3967.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 13:15:00 | 3971.10 | 3974.77 | 3967.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 3971.10 | 3974.77 | 3967.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 3971.10 | 3974.77 | 3967.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 3978.00 | 3996.84 | 3987.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 3978.00 | 3996.84 | 3987.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 4357.20 | 4366.45 | 4335.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 4333.60 | 4366.45 | 4335.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 4345.00 | 4357.93 | 4337.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 4345.00 | 4357.93 | 4337.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 4351.00 | 4356.54 | 4338.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:45:00 | 4347.10 | 4356.54 | 4338.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 4351.00 | 4352.79 | 4339.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 4373.30 | 4352.79 | 4339.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 4327.60 | 4363.03 | 4355.61 | SL hit (close<static) qty=1.00 sl=4335.50 alert=retest2 |

### Cycle 197 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 4307.30 | 4342.44 | 4346.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 4261.80 | 4314.98 | 4330.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 4317.40 | 4309.71 | 4324.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 14:00:00 | 4317.40 | 4309.71 | 4324.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 4321.30 | 4312.03 | 4324.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:45:00 | 4315.70 | 4312.03 | 4324.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 4401.40 | 4330.31 | 4330.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:15:00 | 4486.50 | 4330.31 | 4330.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 4292.80 | 4322.81 | 4327.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 4256.00 | 4322.81 | 4327.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:45:00 | 4268.80 | 4288.32 | 4303.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 4344.00 | 4311.55 | 4311.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 4344.00 | 4311.55 | 4311.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 4382.80 | 4345.73 | 4330.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 4385.00 | 4388.94 | 4369.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:15:00 | 4434.90 | 4388.94 | 4369.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 4420.60 | 4395.28 | 4373.78 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 4313.20 | 4369.73 | 4372.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 10:15:00 | 4296.00 | 4354.99 | 4365.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 4331.40 | 4316.57 | 4337.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 4331.40 | 4316.57 | 4337.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 4331.40 | 4316.57 | 4337.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 4331.40 | 4316.57 | 4337.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 4298.60 | 4312.98 | 4333.83 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 14:15:00 | 4387.20 | 4348.69 | 4345.80 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 4299.20 | 4343.64 | 4344.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 4284.30 | 4331.77 | 4338.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 4261.90 | 4257.13 | 4287.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:30:00 | 4258.90 | 4257.13 | 4287.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 4348.20 | 4275.46 | 4285.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 4352.80 | 4275.46 | 4285.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 4335.00 | 4287.37 | 4290.14 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 4332.00 | 4296.30 | 4293.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 12:15:00 | 4370.50 | 4311.14 | 4300.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 4344.30 | 4345.50 | 4323.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:15:00 | 4330.10 | 4345.50 | 4323.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 4337.80 | 4343.96 | 4324.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:45:00 | 4353.00 | 4342.39 | 4327.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:45:00 | 4355.00 | 4346.84 | 4333.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 4298.00 | 4336.52 | 4334.70 | SL hit (close<static) qty=1.00 sl=4315.60 alert=retest2 |

### Cycle 203 — SELL (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 15:15:00 | 4295.70 | 4328.36 | 4331.15 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 4401.60 | 4333.47 | 4330.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 4418.40 | 4350.46 | 4338.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 12:15:00 | 4525.60 | 4527.48 | 4496.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:00:00 | 4525.60 | 4527.48 | 4496.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 4461.70 | 4508.22 | 4497.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 13:30:00 | 4521.60 | 4497.90 | 4494.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 4545.90 | 4508.00 | 4500.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 4468.00 | 4498.56 | 4498.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 4468.00 | 4498.56 | 4498.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 4453.90 | 4485.08 | 4492.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 11:15:00 | 4474.90 | 4474.73 | 4484.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:45:00 | 4471.70 | 4474.73 | 4484.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 4464.30 | 4472.64 | 4482.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 4480.30 | 4472.64 | 4482.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 4481.50 | 4472.18 | 4480.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 4481.50 | 4472.18 | 4480.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 4457.70 | 4469.29 | 4478.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 4510.80 | 4469.29 | 4478.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 4493.80 | 4474.19 | 4480.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 4520.90 | 4474.19 | 4480.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 4494.00 | 4478.15 | 4481.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:45:00 | 4494.80 | 4478.15 | 4481.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 4471.00 | 4476.70 | 4480.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 4471.10 | 4476.70 | 4480.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 4466.10 | 4472.21 | 4477.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 4419.30 | 4469.99 | 4475.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 4479.80 | 4471.83 | 4473.45 | SL hit (close>static) qty=1.00 sl=4479.10 alert=retest2 |

### Cycle 206 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 4525.10 | 4482.19 | 4477.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 4528.20 | 4491.40 | 4482.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 4525.40 | 4536.75 | 4518.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 12:15:00 | 4525.40 | 4536.75 | 4518.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 4525.40 | 4536.75 | 4518.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 4525.40 | 4536.75 | 4518.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 4520.00 | 4532.35 | 4520.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 4540.40 | 4532.35 | 4520.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 4552.50 | 4537.68 | 4524.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 11:00:00 | 4543.60 | 4538.86 | 4525.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 4515.30 | 4557.54 | 4545.92 | SL hit (close<static) qty=1.00 sl=4516.10 alert=retest2 |

### Cycle 207 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 4500.20 | 4537.97 | 4540.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 4490.00 | 4515.33 | 4528.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 4503.80 | 4503.23 | 4517.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 4503.80 | 4503.23 | 4517.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 4503.80 | 4503.23 | 4517.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 4518.10 | 4503.23 | 4517.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 4527.50 | 4508.09 | 4518.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 4527.50 | 4508.09 | 4518.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 4514.90 | 4509.45 | 4518.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:45:00 | 4503.10 | 4508.40 | 4516.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 4495.90 | 4509.43 | 4515.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 4523.90 | 4445.72 | 4437.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 4523.90 | 4445.72 | 4437.28 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 4416.50 | 4472.56 | 4476.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 4399.30 | 4457.90 | 4469.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 4376.10 | 4374.12 | 4404.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 12:00:00 | 4376.10 | 4374.12 | 4404.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 4441.20 | 4389.06 | 4406.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 4441.20 | 4389.06 | 4406.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 4401.00 | 4391.44 | 4405.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 4369.00 | 4393.22 | 4405.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 4386.80 | 4391.96 | 4400.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 4390.10 | 4392.86 | 4399.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 4455.80 | 4401.63 | 4401.95 | SL hit (close>static) qty=1.00 sl=4443.20 alert=retest2 |

### Cycle 210 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 4443.10 | 4409.92 | 4405.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 4478.90 | 4441.21 | 4426.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 4449.30 | 4463.09 | 4446.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 4449.30 | 4463.09 | 4446.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 4449.30 | 4463.09 | 4446.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 4449.30 | 4463.09 | 4446.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 4448.00 | 4460.07 | 4446.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 4448.00 | 4460.07 | 4446.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 4450.00 | 4458.06 | 4447.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:45:00 | 4465.40 | 4463.55 | 4451.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 4425.20 | 4467.00 | 4458.38 | SL hit (close<static) qty=1.00 sl=4443.50 alert=retest2 |

### Cycle 211 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 4376.60 | 4448.92 | 4450.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 4313.00 | 4421.74 | 4438.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 4214.40 | 4173.14 | 4227.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 4214.40 | 4173.14 | 4227.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 4214.40 | 4173.14 | 4227.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:30:00 | 4137.30 | 4158.70 | 4207.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:30:00 | 4140.40 | 4141.17 | 4182.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:00:00 | 4145.20 | 4141.17 | 4182.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:30:00 | 4147.80 | 4141.14 | 4178.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 3937.94 | 4070.30 | 4124.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 3940.41 | 4070.30 | 4124.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 3930.43 | 4037.66 | 4105.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 3933.38 | 4037.66 | 4105.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 3986.50 | 3968.06 | 4015.73 | SL hit (close>ema200) qty=0.50 sl=3968.06 alert=retest2 |

### Cycle 212 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 4064.60 | 4014.82 | 4012.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 4067.90 | 4025.44 | 4017.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 4031.50 | 4043.55 | 4031.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 4031.50 | 4043.55 | 4031.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 4031.50 | 4043.55 | 4031.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 4035.30 | 4043.55 | 4031.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 4030.70 | 4040.98 | 4031.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 4030.70 | 4040.98 | 4031.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 4019.30 | 4036.64 | 4030.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 4010.40 | 4036.64 | 4030.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 4017.70 | 4032.86 | 4029.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:45:00 | 4026.10 | 4030.88 | 4028.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 4037.90 | 4030.65 | 4028.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 4030.20 | 4027.74 | 4027.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 4025.20 | 4027.23 | 4027.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 4025.20 | 4027.23 | 4027.46 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 12:15:00 | 4057.80 | 4033.23 | 4030.10 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 4002.10 | 4023.31 | 4026.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 3995.80 | 4017.81 | 4023.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 4026.60 | 4009.30 | 4017.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 4026.60 | 4009.30 | 4017.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 4026.60 | 4009.30 | 4017.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 4026.60 | 4009.30 | 4017.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 3987.70 | 4004.98 | 4014.35 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 4035.70 | 4018.28 | 4017.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 4066.10 | 4027.84 | 4021.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 4026.10 | 4034.72 | 4026.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 4026.10 | 4034.72 | 4026.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 4026.10 | 4034.72 | 4026.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 4026.10 | 4034.72 | 4026.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 4021.00 | 4031.98 | 4025.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 4021.00 | 4031.98 | 4025.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 3973.90 | 4020.36 | 4021.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 3949.50 | 4001.71 | 4012.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 3994.80 | 3947.32 | 3966.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 3994.80 | 3947.32 | 3966.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3994.80 | 3947.32 | 3966.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 4010.90 | 3947.32 | 3966.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 4025.10 | 3962.88 | 3971.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 4025.10 | 3962.88 | 3971.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 4008.90 | 3979.17 | 3977.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 4020.90 | 3990.63 | 3983.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 3996.50 | 3997.63 | 3989.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 3996.50 | 3997.63 | 3989.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 3996.50 | 3997.63 | 3989.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 4020.00 | 3997.63 | 3989.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 12:45:00 | 4031.00 | 4004.84 | 3994.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:45:00 | 4018.30 | 4072.41 | 4062.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 4038.10 | 4072.41 | 4062.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 3996.60 | 4051.70 | 4054.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 3996.60 | 4051.70 | 4054.66 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 4083.00 | 4057.41 | 4055.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 4205.10 | 4086.94 | 4068.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 4161.70 | 4169.39 | 4123.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 15:00:00 | 4161.70 | 4169.39 | 4123.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 4174.90 | 4170.77 | 4132.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 11:45:00 | 4196.10 | 4177.23 | 4142.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:00:00 | 4199.00 | 4181.58 | 4147.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 14:00:00 | 4202.10 | 4185.69 | 4152.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 4211.20 | 4190.79 | 4157.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 4192.00 | 4192.51 | 4164.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 10:15:00 | 4244.60 | 4192.51 | 4164.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-18 09:15:00 | 4615.71 | 4584.20 | 4532.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 4888.50 | 4906.35 | 4907.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 4830.00 | 4891.08 | 4900.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 4720.60 | 4680.63 | 4748.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 4720.60 | 4680.63 | 4748.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 4748.40 | 4700.96 | 4746.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 4748.40 | 4700.96 | 4746.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 4739.20 | 4708.61 | 4745.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 4742.10 | 4708.61 | 4745.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 4779.10 | 4722.71 | 4748.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 4779.10 | 4722.71 | 4748.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 4804.10 | 4738.99 | 4753.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 4825.10 | 4738.99 | 4753.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 4862.00 | 4777.10 | 4769.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 4910.90 | 4817.99 | 4790.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 4801.80 | 4818.93 | 4795.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 4801.80 | 4818.93 | 4795.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 4801.80 | 4818.93 | 4795.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 4801.80 | 4818.93 | 4795.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 4824.00 | 4819.94 | 4798.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 4653.90 | 4819.94 | 4798.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 4666.00 | 4789.15 | 4786.14 | EMA400 retest candle locked (from upside) |

### Cycle 223 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 4664.20 | 4764.16 | 4775.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 4590.20 | 4651.21 | 4684.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 4659.20 | 4652.80 | 4682.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 11:00:00 | 4659.20 | 4652.80 | 4682.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 4718.70 | 4665.98 | 4685.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 4718.70 | 4665.98 | 4685.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 4712.80 | 4675.35 | 4688.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:30:00 | 4720.30 | 4675.35 | 4688.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 4755.30 | 4702.80 | 4699.09 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 4666.40 | 4697.29 | 4698.13 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 11:15:00 | 4708.10 | 4699.45 | 4699.04 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 4651.00 | 4689.76 | 4694.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 4628.70 | 4673.59 | 4686.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 4683.90 | 4675.65 | 4686.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:15:00 | 4636.70 | 4675.65 | 4686.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 4624.00 | 4665.32 | 4680.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 4659.20 | 4665.32 | 4680.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 4626.00 | 4593.11 | 4625.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 4635.30 | 4593.11 | 4625.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 4602.20 | 4594.93 | 4623.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 4593.80 | 4594.93 | 4623.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:45:00 | 4600.00 | 4596.68 | 4621.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 4597.70 | 4596.68 | 4621.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 4597.00 | 4603.74 | 4620.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 4644.20 | 4611.37 | 4619.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 4649.10 | 4611.37 | 4619.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 4645.30 | 4618.16 | 4622.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 4695.20 | 4633.57 | 4628.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 4695.20 | 4633.57 | 4628.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 4744.50 | 4655.75 | 4639.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4592.10 | 4668.93 | 4654.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4592.10 | 4668.93 | 4654.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4592.10 | 4668.93 | 4654.01 | EMA400 retest candle locked (from upside) |

### Cycle 229 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 4556.30 | 4629.06 | 4637.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 4532.80 | 4595.73 | 4619.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 4630.00 | 4575.98 | 4602.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 4630.00 | 4575.98 | 4602.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 4630.00 | 4575.98 | 4602.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 4644.20 | 4575.98 | 4602.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 4626.50 | 4586.08 | 4604.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 4620.10 | 4586.08 | 4604.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 4597.40 | 4588.34 | 4603.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 4579.50 | 4588.34 | 4603.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 4512.80 | 4607.78 | 4609.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 4628.50 | 4573.07 | 4567.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 230 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 4628.50 | 4573.07 | 4567.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 4752.60 | 4629.63 | 4597.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 4601.00 | 4690.79 | 4654.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 4601.00 | 4690.79 | 4654.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 4601.00 | 4690.79 | 4654.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 4605.00 | 4690.79 | 4654.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 4601.30 | 4672.89 | 4650.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:30:00 | 4618.60 | 4666.30 | 4649.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 4582.20 | 4638.80 | 4641.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4582.20 | 4638.80 | 4641.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 4527.20 | 4616.48 | 4630.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4615.00 | 4559.61 | 4590.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4615.00 | 4559.61 | 4590.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4615.00 | 4559.61 | 4590.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 4625.40 | 4559.61 | 4590.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4560.00 | 4559.69 | 4587.35 | EMA400 retest candle locked (from downside) |

### Cycle 232 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 4618.00 | 4600.83 | 4600.44 | EMA200 above EMA400 |

### Cycle 233 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 4503.40 | 4581.35 | 4591.62 | EMA200 below EMA400 |

### Cycle 234 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 4654.90 | 4594.15 | 4590.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 4681.50 | 4620.62 | 4604.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 4639.40 | 4661.26 | 4636.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 4639.40 | 4661.26 | 4636.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 4639.40 | 4661.26 | 4636.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 4781.30 | 4640.39 | 4634.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 5259.43 | 5191.00 | 5138.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 235 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 5144.60 | 5175.13 | 5178.45 | EMA200 below EMA400 |

### Cycle 236 — BUY (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 14:15:00 | 5223.80 | 5180.41 | 5179.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 15:15:00 | 5241.80 | 5192.69 | 5185.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 5237.00 | 5238.19 | 5218.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 10:15:00 | 5224.80 | 5238.19 | 5218.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 5224.00 | 5235.35 | 5218.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:30:00 | 5219.10 | 5235.35 | 5218.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 5231.30 | 5234.54 | 5219.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 5226.40 | 5234.54 | 5219.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 5208.00 | 5229.23 | 5218.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 5208.00 | 5229.23 | 5218.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 5226.60 | 5228.71 | 5219.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:15:00 | 5231.20 | 5228.71 | 5219.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 5244.00 | 5266.08 | 5267.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 237 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 5244.00 | 5266.08 | 5267.88 | EMA200 below EMA400 |

### Cycle 238 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 5293.50 | 5268.50 | 5267.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 5362.50 | 5320.76 | 5299.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 5391.00 | 5408.85 | 5367.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 5391.00 | 5408.85 | 5367.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-02 09:15:00 | 1756.90 | 2023-06-02 13:15:00 | 1744.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-06-02 11:30:00 | 1752.20 | 2023-06-02 13:15:00 | 1744.50 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2023-06-02 15:00:00 | 1754.10 | 2023-06-08 12:15:00 | 1764.15 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2023-06-16 11:30:00 | 1839.75 | 2023-06-21 15:15:00 | 1855.10 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2023-06-16 12:15:00 | 1837.70 | 2023-06-21 15:15:00 | 1855.10 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2023-06-16 12:45:00 | 1839.00 | 2023-06-21 15:15:00 | 1855.10 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2023-07-14 11:30:00 | 1895.50 | 2023-07-14 12:15:00 | 1915.95 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-07-27 10:15:00 | 1932.80 | 2023-08-01 15:15:00 | 1925.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-07-28 09:45:00 | 1939.00 | 2023-08-01 15:15:00 | 1925.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-08-08 09:15:00 | 1732.40 | 2023-08-10 11:15:00 | 1762.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2023-08-08 11:15:00 | 1740.30 | 2023-08-17 12:15:00 | 1748.55 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-08-09 10:45:00 | 1744.00 | 2023-08-17 12:15:00 | 1748.55 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2023-08-09 12:15:00 | 1744.30 | 2023-08-17 12:15:00 | 1748.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2023-08-10 10:30:00 | 1739.40 | 2023-08-17 12:15:00 | 1748.55 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2023-08-11 15:00:00 | 1740.00 | 2023-08-17 12:15:00 | 1748.55 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-08-14 15:00:00 | 1740.80 | 2023-08-17 12:15:00 | 1748.55 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-08-16 12:30:00 | 1736.50 | 2023-08-17 12:15:00 | 1748.55 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-09-08 09:30:00 | 1746.35 | 2023-09-12 09:15:00 | 1729.35 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2023-09-08 10:00:00 | 1742.90 | 2023-09-12 09:15:00 | 1729.35 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-10-03 14:45:00 | 1673.30 | 2023-10-06 09:15:00 | 1700.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2023-10-12 15:00:00 | 1730.75 | 2023-10-13 09:15:00 | 1715.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-10-25 11:15:00 | 1700.45 | 2023-11-02 10:15:00 | 1691.75 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2023-10-25 14:45:00 | 1700.00 | 2023-11-02 10:15:00 | 1691.75 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2023-10-27 09:45:00 | 1703.00 | 2023-11-02 10:15:00 | 1691.75 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2023-11-07 10:15:00 | 1727.00 | 2023-11-07 12:15:00 | 1699.35 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2023-11-24 11:15:00 | 1865.45 | 2023-12-07 12:15:00 | 1939.00 | STOP_HIT | 1.00 | 3.94% |
| BUY | retest2 | 2023-11-28 13:15:00 | 1865.00 | 2023-12-07 12:15:00 | 1939.00 | STOP_HIT | 1.00 | 3.97% |
| SELL | retest2 | 2023-12-15 10:30:00 | 1946.70 | 2023-12-18 13:15:00 | 1959.30 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-12-15 11:15:00 | 1947.00 | 2023-12-18 13:15:00 | 1959.30 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-12-18 10:30:00 | 1948.40 | 2023-12-18 13:15:00 | 1959.30 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-12-18 11:15:00 | 1948.80 | 2023-12-18 13:15:00 | 1959.30 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-12-28 09:15:00 | 2003.40 | 2023-12-28 10:15:00 | 1982.75 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-01-08 12:15:00 | 1998.75 | 2024-01-16 12:15:00 | 2014.55 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2024-01-24 10:15:00 | 2113.15 | 2024-02-02 14:15:00 | 2323.20 | TARGET_HIT | 1.00 | 9.94% |
| BUY | retest2 | 2024-01-24 12:15:00 | 2113.35 | 2024-02-05 09:15:00 | 2324.47 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2024-01-24 12:45:00 | 2112.00 | 2024-02-05 09:15:00 | 2324.68 | TARGET_HIT | 1.00 | 10.07% |
| BUY | retest2 | 2024-03-06 11:30:00 | 2758.40 | 2024-03-12 12:15:00 | 2764.90 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-03-06 12:00:00 | 2756.85 | 2024-03-12 12:15:00 | 2764.90 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2024-03-14 13:30:00 | 2700.90 | 2024-03-20 09:15:00 | 2741.10 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-03-15 09:15:00 | 2714.70 | 2024-03-20 09:15:00 | 2741.10 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-03-15 14:15:00 | 2715.85 | 2024-03-20 09:15:00 | 2741.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-03-15 15:00:00 | 2704.10 | 2024-03-20 09:15:00 | 2741.10 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-04-02 11:45:00 | 3034.65 | 2024-04-08 10:15:00 | 2954.25 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-04-02 12:15:00 | 3036.25 | 2024-04-08 10:15:00 | 2954.25 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-04-02 13:15:00 | 3028.15 | 2024-04-08 10:15:00 | 2954.25 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-04-02 14:15:00 | 3028.40 | 2024-04-08 10:15:00 | 2954.25 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-04-03 11:30:00 | 3028.70 | 2024-04-08 10:15:00 | 2954.25 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-04-04 10:00:00 | 3021.05 | 2024-04-08 10:15:00 | 2954.25 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-04-04 11:30:00 | 3019.45 | 2024-04-08 10:15:00 | 2954.25 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-04-05 10:15:00 | 3020.00 | 2024-04-08 10:15:00 | 2954.25 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-04-05 14:00:00 | 3034.90 | 2024-04-08 10:15:00 | 2954.25 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-04-15 14:00:00 | 3045.40 | 2024-04-19 14:15:00 | 3041.40 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-04-16 09:30:00 | 3050.30 | 2024-04-19 14:15:00 | 3041.40 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-04-19 11:00:00 | 3043.25 | 2024-04-19 14:15:00 | 3041.40 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-05-02 11:45:00 | 3307.05 | 2024-05-15 10:15:00 | 3637.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-02 12:15:00 | 3310.40 | 2024-05-15 10:15:00 | 3641.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-02 15:15:00 | 3310.00 | 2024-05-15 10:15:00 | 3641.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-03 09:45:00 | 3391.80 | 2024-05-16 09:15:00 | 3730.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-08 12:00:00 | 3417.25 | 2024-05-16 09:15:00 | 3758.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-09 11:00:00 | 3416.30 | 2024-05-16 09:15:00 | 3757.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-09 13:00:00 | 3406.95 | 2024-05-16 09:15:00 | 3747.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-10 09:15:00 | 3413.00 | 2024-05-16 09:15:00 | 3754.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 09:15:00 | 3442.75 | 2024-05-16 10:15:00 | 3787.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 10:00:00 | 3426.35 | 2024-05-16 10:15:00 | 3768.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-23 10:30:00 | 3718.00 | 2024-05-23 12:15:00 | 3755.45 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-05-29 13:00:00 | 3843.95 | 2024-05-30 09:15:00 | 3739.35 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-05-30 09:30:00 | 3846.30 | 2024-05-30 10:15:00 | 3677.65 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2024-06-10 12:30:00 | 3552.10 | 2024-06-19 09:15:00 | 3907.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 09:15:00 | 3569.60 | 2024-06-19 09:15:00 | 3926.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-02 10:30:00 | 3980.00 | 2024-07-04 09:15:00 | 4040.85 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-07-12 09:15:00 | 3984.50 | 2024-07-18 09:15:00 | 3785.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 09:15:00 | 3984.50 | 2024-07-19 11:15:00 | 3586.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-07 09:15:00 | 3482.50 | 2024-08-07 11:15:00 | 3743.15 | STOP_HIT | 1.00 | -7.48% |
| BUY | retest2 | 2024-08-12 13:00:00 | 3748.65 | 2024-08-13 13:15:00 | 3668.85 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-08-19 11:30:00 | 3790.00 | 2024-08-20 11:15:00 | 3732.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-08-19 15:15:00 | 3768.05 | 2024-08-20 11:15:00 | 3732.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-08-20 12:45:00 | 3765.00 | 2024-08-27 11:15:00 | 3790.65 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2024-09-06 09:30:00 | 3857.75 | 2024-09-06 12:15:00 | 3805.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-09-13 11:45:00 | 3824.95 | 2024-09-18 13:15:00 | 3770.25 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-09-16 09:30:00 | 3823.30 | 2024-09-18 13:15:00 | 3770.25 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-09-16 10:45:00 | 3821.25 | 2024-09-18 13:15:00 | 3770.25 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-09-17 11:15:00 | 3828.10 | 2024-09-18 13:15:00 | 3770.25 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-09-20 12:15:00 | 3750.85 | 2024-09-20 14:15:00 | 3816.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-09-20 13:30:00 | 3733.70 | 2024-09-20 14:15:00 | 3816.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-09-20 14:15:00 | 3752.00 | 2024-09-20 14:15:00 | 3816.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-21 12:45:00 | 3658.80 | 2024-10-23 09:15:00 | 3475.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:45:00 | 3658.80 | 2024-10-28 10:15:00 | 3411.90 | STOP_HIT | 0.50 | 6.75% |
| BUY | retest2 | 2024-10-31 14:45:00 | 3500.50 | 2024-11-04 14:15:00 | 3468.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-11-01 18:00:00 | 3514.60 | 2024-11-04 14:15:00 | 3468.70 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-11-04 13:45:00 | 3499.95 | 2024-11-04 14:15:00 | 3468.70 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-11-08 09:45:00 | 3589.95 | 2024-11-08 10:15:00 | 3532.95 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-11-08 12:30:00 | 3608.85 | 2024-11-11 09:15:00 | 3513.65 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-11-11 11:45:00 | 3617.95 | 2024-11-12 09:15:00 | 3567.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-11-11 13:00:00 | 3602.00 | 2024-11-12 10:15:00 | 3493.00 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-11-12 09:15:00 | 3623.90 | 2024-11-12 10:15:00 | 3493.00 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2024-11-27 12:15:00 | 3490.05 | 2024-11-28 12:15:00 | 3443.30 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-11-27 13:45:00 | 3496.75 | 2024-11-28 12:15:00 | 3443.30 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-11-27 14:45:00 | 3494.15 | 2024-11-28 12:15:00 | 3443.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-11-28 09:30:00 | 3490.05 | 2024-11-28 12:15:00 | 3443.30 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-12-13 15:00:00 | 3601.50 | 2024-12-17 13:15:00 | 3566.45 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-12-26 10:15:00 | 3245.00 | 2025-01-10 09:15:00 | 3082.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-30 11:15:00 | 3246.85 | 2025-01-10 09:15:00 | 3084.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-30 12:15:00 | 3242.15 | 2025-01-10 09:15:00 | 3080.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-30 13:00:00 | 3246.80 | 2025-01-10 09:15:00 | 3084.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 12:15:00 | 3211.70 | 2025-01-10 09:15:00 | 3051.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 12:45:00 | 3213.45 | 2025-01-10 09:15:00 | 3052.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 09:30:00 | 3200.00 | 2025-01-10 09:15:00 | 3040.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 09:45:00 | 3173.15 | 2025-01-10 13:15:00 | 3014.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:30:00 | 3130.45 | 2025-01-13 09:15:00 | 2973.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:30:00 | 3146.10 | 2025-01-13 09:15:00 | 2988.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 3138.50 | 2025-01-13 09:15:00 | 2981.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-26 10:15:00 | 3245.00 | 2025-01-13 11:15:00 | 2920.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-30 11:15:00 | 3246.85 | 2025-01-13 11:15:00 | 2922.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-30 12:15:00 | 3242.15 | 2025-01-13 11:15:00 | 2917.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-30 13:00:00 | 3246.80 | 2025-01-13 11:15:00 | 2922.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 12:15:00 | 3211.70 | 2025-01-13 13:15:00 | 2890.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 12:45:00 | 3213.45 | 2025-01-13 13:15:00 | 2892.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-06 09:30:00 | 3200.00 | 2025-01-13 13:15:00 | 2880.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-07 09:45:00 | 3173.15 | 2025-01-13 14:15:00 | 2855.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 11:30:00 | 3130.45 | 2025-01-14 13:15:00 | 2930.90 | STOP_HIT | 0.50 | 6.37% |
| SELL | retest2 | 2025-01-09 09:30:00 | 3146.10 | 2025-01-14 13:15:00 | 2930.90 | STOP_HIT | 0.50 | 6.84% |
| SELL | retest2 | 2025-01-09 10:45:00 | 3138.50 | 2025-01-14 13:15:00 | 2930.90 | STOP_HIT | 0.50 | 6.61% |
| SELL | retest2 | 2025-01-23 14:00:00 | 2891.95 | 2025-01-27 09:15:00 | 2747.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:00:00 | 2891.95 | 2025-01-27 12:15:00 | 2804.00 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2025-02-13 13:45:00 | 2791.20 | 2025-02-17 09:15:00 | 2651.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:30:00 | 2797.85 | 2025-02-17 09:15:00 | 2657.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 2799.00 | 2025-02-17 09:15:00 | 2659.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:45:00 | 2791.20 | 2025-02-17 13:15:00 | 2718.00 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2025-02-13 14:30:00 | 2797.85 | 2025-02-17 13:15:00 | 2718.00 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2025-02-13 15:00:00 | 2799.00 | 2025-02-17 13:15:00 | 2718.00 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2025-02-14 09:15:00 | 2763.65 | 2025-02-20 09:15:00 | 2758.10 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-02-19 14:15:00 | 2668.80 | 2025-02-20 10:15:00 | 2783.00 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-02-24 14:30:00 | 2820.80 | 2025-02-25 10:15:00 | 2793.40 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-03-12 10:15:00 | 2817.50 | 2025-03-17 14:15:00 | 2862.70 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-03-24 09:15:00 | 3011.90 | 2025-03-25 14:15:00 | 2958.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-04-08 10:30:00 | 2746.05 | 2025-04-11 12:15:00 | 2840.80 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-04-08 15:00:00 | 2746.00 | 2025-04-11 12:15:00 | 2840.80 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-04-09 12:00:00 | 2747.05 | 2025-04-11 12:15:00 | 2840.80 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-04-17 11:15:00 | 2873.50 | 2025-04-23 09:15:00 | 2872.70 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-04-24 14:45:00 | 2930.80 | 2025-04-25 09:15:00 | 2863.90 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-05-26 11:45:00 | 2924.00 | 2025-05-27 14:15:00 | 2973.70 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-26 13:45:00 | 2931.60 | 2025-05-27 14:15:00 | 2973.70 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-05-27 09:15:00 | 2927.90 | 2025-05-27 14:15:00 | 2973.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-05-27 09:45:00 | 2930.90 | 2025-05-27 14:15:00 | 2973.70 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-29 09:15:00 | 3061.80 | 2025-06-04 09:15:00 | 3367.98 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-18 11:30:00 | 3268.70 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-06-18 13:45:00 | 3274.10 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-06-18 15:15:00 | 3273.50 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-06-19 10:30:00 | 3273.00 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-06-20 12:15:00 | 3277.10 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-06-20 14:00:00 | 3280.00 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-20 15:00:00 | 3267.30 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-30 09:15:00 | 3384.00 | 2025-07-01 15:15:00 | 3352.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-07-11 10:45:00 | 3555.00 | 2025-07-14 11:15:00 | 3487.30 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-07-11 11:30:00 | 3556.60 | 2025-07-14 11:15:00 | 3487.30 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-07-11 12:30:00 | 3557.70 | 2025-07-14 11:15:00 | 3487.30 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-11 14:30:00 | 3551.60 | 2025-07-14 11:15:00 | 3487.30 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-07-15 11:30:00 | 3550.40 | 2025-07-18 11:15:00 | 3547.90 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-07-18 11:15:00 | 3537.00 | 2025-07-18 11:15:00 | 3547.90 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-07-22 11:30:00 | 3598.00 | 2025-07-23 09:15:00 | 3552.60 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-07-23 11:30:00 | 3589.10 | 2025-07-24 14:15:00 | 3567.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-30 10:00:00 | 3515.50 | 2025-07-30 12:15:00 | 3560.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-13 09:15:00 | 3773.10 | 2025-08-19 10:15:00 | 3749.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-08-19 09:45:00 | 3771.80 | 2025-08-19 10:15:00 | 3749.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-15 09:30:00 | 4060.40 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-09-15 11:30:00 | 4059.30 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-09-15 12:15:00 | 4059.90 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-09-16 09:15:00 | 4060.00 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-09-16 13:00:00 | 4078.80 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-09-25 11:30:00 | 4001.00 | 2025-10-06 09:15:00 | 3954.20 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2025-10-16 09:15:00 | 3975.40 | 2025-10-16 11:15:00 | 3946.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-11-03 09:15:00 | 4373.30 | 2025-11-04 09:15:00 | 4327.60 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-11-07 11:15:00 | 4256.00 | 2025-11-10 14:15:00 | 4344.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-11-10 11:45:00 | 4268.80 | 2025-11-10 14:15:00 | 4344.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-11-21 12:45:00 | 4353.00 | 2025-11-24 14:15:00 | 4298.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-24 09:45:00 | 4355.00 | 2025-11-24 14:15:00 | 4298.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-03 13:30:00 | 4521.60 | 2025-12-04 13:15:00 | 4468.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-04 09:30:00 | 4545.90 | 2025-12-04 13:15:00 | 4468.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-09 09:15:00 | 4419.30 | 2025-12-09 14:15:00 | 4479.80 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-12 09:15:00 | 4540.40 | 2025-12-15 10:15:00 | 4515.30 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-12-12 09:45:00 | 4552.50 | 2025-12-15 10:15:00 | 4515.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-12-12 11:00:00 | 4543.60 | 2025-12-15 10:15:00 | 4515.30 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-15 12:00:00 | 4541.20 | 2025-12-16 09:15:00 | 4500.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-17 12:45:00 | 4503.10 | 2025-12-22 11:15:00 | 4523.90 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-12-18 09:15:00 | 4495.90 | 2025-12-22 11:15:00 | 4523.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-30 09:15:00 | 4369.00 | 2025-12-31 09:15:00 | 4455.80 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-12-30 13:15:00 | 4386.80 | 2025-12-31 09:15:00 | 4455.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-12-30 15:00:00 | 4390.10 | 2025-12-31 09:15:00 | 4455.80 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-01-02 14:45:00 | 4465.40 | 2026-01-05 11:15:00 | 4425.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-08 12:30:00 | 4137.30 | 2026-01-12 10:15:00 | 3937.94 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2026-01-09 09:30:00 | 4140.40 | 2026-01-12 10:15:00 | 3940.41 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2026-01-09 10:00:00 | 4145.20 | 2026-01-12 11:15:00 | 3930.43 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2026-01-09 10:30:00 | 4147.80 | 2026-01-12 11:15:00 | 3933.38 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-01-08 12:30:00 | 4137.30 | 2026-01-13 14:15:00 | 3986.50 | STOP_HIT | 0.50 | 3.64% |
| SELL | retest2 | 2026-01-09 09:30:00 | 4140.40 | 2026-01-13 14:15:00 | 3986.50 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2026-01-09 10:00:00 | 4145.20 | 2026-01-13 14:15:00 | 3986.50 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2026-01-09 10:30:00 | 4147.80 | 2026-01-13 14:15:00 | 3986.50 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2026-01-16 09:15:00 | 4015.60 | 2026-01-16 10:15:00 | 4064.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-19 13:45:00 | 4026.10 | 2026-01-20 09:15:00 | 4025.20 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2026-01-19 14:45:00 | 4037.90 | 2026-01-20 09:15:00 | 4025.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-01-20 09:15:00 | 4030.20 | 2026-01-20 09:15:00 | 4025.20 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-01-29 11:15:00 | 4020.00 | 2026-02-02 10:15:00 | 3996.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-01-29 12:45:00 | 4031.00 | 2026-02-02 10:15:00 | 3996.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-01 14:45:00 | 4018.30 | 2026-02-02 10:15:00 | 3996.60 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-02-01 15:15:00 | 4038.10 | 2026-02-02 10:15:00 | 3996.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-04 11:45:00 | 4196.10 | 2026-02-18 09:15:00 | 4615.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 13:00:00 | 4199.00 | 2026-02-18 09:15:00 | 4618.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 14:00:00 | 4202.10 | 2026-02-18 09:15:00 | 4622.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 15:00:00 | 4211.20 | 2026-02-18 09:15:00 | 4632.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-05 10:15:00 | 4244.60 | 2026-02-18 09:15:00 | 4669.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-17 11:15:00 | 4593.80 | 2026-03-18 11:15:00 | 4695.20 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-03-17 11:45:00 | 4600.00 | 2026-03-18 11:15:00 | 4695.20 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-03-17 12:15:00 | 4597.70 | 2026-03-18 11:15:00 | 4695.20 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-03-17 14:15:00 | 4597.00 | 2026-03-18 11:15:00 | 4695.20 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-03-20 12:15:00 | 4579.50 | 2026-03-24 12:15:00 | 4628.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-03-23 09:15:00 | 4512.80 | 2026-03-24 12:15:00 | 4628.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-03-27 11:30:00 | 4618.60 | 2026-03-30 09:15:00 | 4582.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-04-08 09:15:00 | 4781.30 | 2026-04-21 09:15:00 | 5259.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-28 14:15:00 | 5231.20 | 2026-05-05 11:15:00 | 5244.00 | STOP_HIT | 1.00 | 0.24% |
