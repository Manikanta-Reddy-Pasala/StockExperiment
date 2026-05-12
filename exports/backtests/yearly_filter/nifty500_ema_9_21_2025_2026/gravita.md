# Gravita India Ltd. (GRAVITA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1760.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 44 |
| ALERT2 | 44 |
| ALERT2_SKIP | 21 |
| ALERT3 | 114 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 25 / 28
- **Target hits / Stop hits / Partials:** 2 / 46 / 5
- **Avg / median % per leg:** 0.63% / -0.13%
- **Sum % (uncompounded):** 33.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 2 | 10.0% | 1 | 19 | 0 | -0.94% | -18.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 2 | 10.0% | 1 | 19 | 0 | -0.94% | -18.8% |
| SELL (all) | 33 | 23 | 69.7% | 1 | 27 | 5 | 1.58% | 52.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 23 | 69.7% | 1 | 27 | 5 | 1.58% | 52.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 25 | 47.2% | 2 | 46 | 5 | 0.63% | 33.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1888.00 | 1852.31 | 1851.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1901.10 | 1874.38 | 1862.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1880.40 | 1884.45 | 1873.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 1880.40 | 1884.45 | 1873.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1882.80 | 1884.12 | 1874.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 1873.00 | 1884.12 | 1874.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1913.80 | 1915.43 | 1903.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 1913.80 | 1915.43 | 1903.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2077.20 | 2084.10 | 2036.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 2058.00 | 2084.10 | 2036.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 2058.70 | 2084.00 | 2056.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 2058.70 | 2084.00 | 2056.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 2051.00 | 2077.40 | 2055.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 2081.00 | 2077.40 | 2055.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2092.60 | 2080.44 | 2059.22 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 2045.00 | 2058.61 | 2058.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 1978.80 | 2040.74 | 2050.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 1956.50 | 1955.08 | 1982.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 14:30:00 | 1953.00 | 1955.08 | 1982.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1926.90 | 1923.62 | 1942.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 1932.30 | 1923.62 | 1942.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1895.00 | 1915.38 | 1932.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:15:00 | 1886.30 | 1910.59 | 1928.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:45:00 | 1885.50 | 1906.15 | 1924.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:30:00 | 1885.30 | 1899.70 | 1918.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 14:15:00 | 1886.20 | 1899.70 | 1918.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1858.40 | 1846.30 | 1863.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 1858.40 | 1846.30 | 1863.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1886.40 | 1857.03 | 1864.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 1862.20 | 1857.83 | 1863.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:45:00 | 1859.70 | 1858.60 | 1863.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1857.50 | 1856.32 | 1860.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 1868.50 | 1852.11 | 1851.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 1868.50 | 1852.11 | 1851.10 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 10:15:00 | 1850.00 | 1857.00 | 1857.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 11:15:00 | 1844.10 | 1854.42 | 1856.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 1685.20 | 1683.43 | 1703.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 1685.20 | 1683.43 | 1703.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1685.20 | 1683.43 | 1703.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 1691.90 | 1683.43 | 1703.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1667.90 | 1664.52 | 1679.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1669.50 | 1664.52 | 1679.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1682.00 | 1668.65 | 1676.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1665.20 | 1668.65 | 1676.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1687.60 | 1674.53 | 1678.29 | SL hit (close>static) qty=1.00 sl=1685.40 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 1685.00 | 1680.90 | 1680.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 15:15:00 | 1687.90 | 1682.96 | 1681.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 1859.00 | 1860.34 | 1831.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 1859.00 | 1860.34 | 1831.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1857.20 | 1859.67 | 1847.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 1859.40 | 1853.71 | 1847.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 1872.30 | 1856.37 | 1849.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:45:00 | 1859.40 | 1857.16 | 1851.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 11:00:00 | 1859.90 | 1857.59 | 1853.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1854.80 | 1856.56 | 1853.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 1853.60 | 1856.56 | 1853.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1845.10 | 1854.26 | 1853.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 1851.00 | 1854.26 | 1853.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1845.00 | 1852.41 | 1852.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1851.30 | 1852.41 | 1852.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:00:00 | 1851.90 | 1853.94 | 1853.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 1849.40 | 1852.72 | 1852.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 1849.40 | 1852.72 | 1852.76 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 1855.70 | 1853.32 | 1853.03 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 1844.00 | 1851.14 | 1852.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 1833.90 | 1847.69 | 1850.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 1810.60 | 1809.29 | 1822.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:30:00 | 1812.00 | 1809.29 | 1822.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1780.00 | 1796.31 | 1808.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 1777.10 | 1796.31 | 1808.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 1773.50 | 1791.81 | 1805.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 1768.00 | 1779.72 | 1793.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 1729.10 | 1720.83 | 1720.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1729.10 | 1720.83 | 1720.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 1809.00 | 1741.15 | 1730.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 1841.10 | 1850.46 | 1817.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 12:15:00 | 1816.00 | 1836.59 | 1818.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 1816.00 | 1836.59 | 1818.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 1816.00 | 1836.59 | 1818.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1807.00 | 1830.67 | 1817.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:30:00 | 1811.40 | 1830.67 | 1817.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1817.00 | 1827.94 | 1817.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 1817.00 | 1827.94 | 1817.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1823.70 | 1827.09 | 1817.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 1832.00 | 1827.09 | 1817.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 1834.50 | 1861.50 | 1863.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1834.50 | 1861.50 | 1863.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1803.00 | 1849.80 | 1858.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 12:15:00 | 1780.40 | 1778.87 | 1792.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 13:15:00 | 1782.70 | 1779.64 | 1791.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1782.70 | 1779.64 | 1791.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 1782.70 | 1779.64 | 1791.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1788.20 | 1781.35 | 1791.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 1788.20 | 1781.35 | 1791.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1788.30 | 1782.74 | 1791.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 1802.00 | 1782.74 | 1791.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1796.90 | 1785.57 | 1791.83 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 1837.50 | 1796.26 | 1794.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 1849.00 | 1806.81 | 1799.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 10:15:00 | 1799.10 | 1805.49 | 1799.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 10:15:00 | 1799.10 | 1805.49 | 1799.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1799.10 | 1805.49 | 1799.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 1795.90 | 1805.49 | 1799.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1798.80 | 1804.15 | 1799.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:45:00 | 1800.50 | 1804.15 | 1799.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1799.30 | 1803.18 | 1799.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:30:00 | 1798.10 | 1803.18 | 1799.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 1795.50 | 1801.65 | 1799.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:45:00 | 1794.30 | 1801.65 | 1799.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 1772.80 | 1795.88 | 1796.96 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 1800.00 | 1796.88 | 1796.55 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 1786.20 | 1795.30 | 1795.97 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 1806.10 | 1793.77 | 1792.82 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 1786.50 | 1791.59 | 1791.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 1779.50 | 1789.17 | 1790.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 12:15:00 | 1789.40 | 1786.92 | 1789.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 12:15:00 | 1789.40 | 1786.92 | 1789.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1789.40 | 1786.92 | 1789.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:30:00 | 1788.80 | 1786.92 | 1789.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1793.80 | 1788.30 | 1789.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 1793.80 | 1788.30 | 1789.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1784.20 | 1787.48 | 1789.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 15:15:00 | 1780.30 | 1787.48 | 1789.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1750.80 | 1738.51 | 1737.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 1750.80 | 1738.51 | 1737.55 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1703.00 | 1735.39 | 1737.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1679.30 | 1703.34 | 1719.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1654.00 | 1639.90 | 1659.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1654.00 | 1639.90 | 1659.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1654.00 | 1639.90 | 1659.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:15:00 | 1659.70 | 1639.90 | 1659.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1656.90 | 1643.30 | 1659.70 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1698.50 | 1668.48 | 1665.31 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1658.10 | 1680.07 | 1681.09 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 12:15:00 | 1697.50 | 1682.73 | 1681.38 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 1677.90 | 1686.65 | 1686.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 1659.40 | 1667.40 | 1673.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 1665.90 | 1661.88 | 1667.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 1680.80 | 1661.88 | 1667.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1675.30 | 1664.56 | 1668.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 1683.50 | 1664.56 | 1668.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1675.00 | 1666.65 | 1668.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 1669.00 | 1668.24 | 1669.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:30:00 | 1671.50 | 1664.24 | 1665.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:00:00 | 1669.80 | 1665.35 | 1666.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:00:00 | 1670.40 | 1665.93 | 1666.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1663.80 | 1665.51 | 1665.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 1657.10 | 1665.51 | 1665.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 1666.10 | 1663.98 | 1663.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 13:15:00 | 1666.10 | 1663.98 | 1663.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1688.60 | 1669.84 | 1666.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 1674.00 | 1675.69 | 1670.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 1674.00 | 1675.69 | 1670.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1665.60 | 1674.81 | 1671.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1665.60 | 1674.81 | 1671.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1670.00 | 1673.85 | 1671.70 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 1662.60 | 1669.64 | 1670.04 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 1683.00 | 1671.62 | 1670.38 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 1655.90 | 1669.04 | 1670.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 1651.10 | 1665.45 | 1668.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1663.00 | 1662.20 | 1665.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1663.00 | 1662.20 | 1665.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1663.00 | 1662.20 | 1665.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1663.00 | 1662.20 | 1665.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1654.10 | 1660.58 | 1664.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:30:00 | 1653.30 | 1656.30 | 1661.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1570.63 | 1606.98 | 1628.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 1584.00 | 1575.28 | 1597.88 | SL hit (close>ema200) qty=0.50 sl=1575.28 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1580.00 | 1558.31 | 1557.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 1585.90 | 1563.83 | 1560.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1613.60 | 1625.14 | 1606.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:00:00 | 1613.60 | 1625.14 | 1606.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1604.70 | 1621.05 | 1606.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 1605.10 | 1621.05 | 1606.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1604.40 | 1617.72 | 1606.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:30:00 | 1602.80 | 1617.72 | 1606.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 1614.10 | 1617.00 | 1606.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:15:00 | 1619.80 | 1617.00 | 1606.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:15:00 | 1619.70 | 1616.07 | 1608.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1599.00 | 1612.79 | 1608.76 | SL hit (close<static) qty=1.00 sl=1601.50 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 1590.80 | 1606.04 | 1606.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 1586.80 | 1599.16 | 1602.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 14:15:00 | 1600.00 | 1589.29 | 1594.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 1600.00 | 1589.29 | 1594.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1600.00 | 1589.29 | 1594.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1600.00 | 1589.29 | 1594.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 1605.00 | 1592.43 | 1595.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 1614.60 | 1592.43 | 1595.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1594.00 | 1593.21 | 1595.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 1592.90 | 1593.21 | 1595.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1574.00 | 1589.37 | 1593.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:45:00 | 1573.20 | 1584.48 | 1590.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 1572.30 | 1579.56 | 1585.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 1601.50 | 1568.70 | 1571.32 | SL hit (close>static) qty=1.00 sl=1594.30 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1595.70 | 1574.10 | 1573.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 1628.40 | 1587.65 | 1579.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1593.20 | 1600.76 | 1589.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 1593.20 | 1600.76 | 1589.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1681.00 | 1615.09 | 1598.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:15:00 | 1698.00 | 1629.09 | 1606.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 1698.00 | 1642.87 | 1614.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 1630.00 | 1649.34 | 1651.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1630.00 | 1649.34 | 1651.60 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 1668.40 | 1651.36 | 1649.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 1677.80 | 1656.65 | 1652.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 1688.00 | 1691.77 | 1673.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 12:45:00 | 1686.00 | 1691.77 | 1673.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 1667.60 | 1686.65 | 1674.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:45:00 | 1663.30 | 1686.65 | 1674.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1670.00 | 1683.32 | 1673.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1728.00 | 1683.32 | 1673.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 1702.60 | 1735.82 | 1737.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 1702.60 | 1735.82 | 1737.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 1700.90 | 1728.84 | 1733.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 1702.80 | 1701.63 | 1714.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:45:00 | 1700.70 | 1701.63 | 1714.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1711.80 | 1703.67 | 1714.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1691.10 | 1705.13 | 1713.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1746.50 | 1711.31 | 1715.07 | SL hit (close>static) qty=1.00 sl=1727.70 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1746.50 | 1718.34 | 1717.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1767.30 | 1738.63 | 1731.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 1747.00 | 1749.45 | 1740.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 14:00:00 | 1747.00 | 1749.45 | 1740.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1739.10 | 1747.38 | 1739.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 1739.10 | 1747.38 | 1739.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 1736.30 | 1745.16 | 1739.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1752.10 | 1745.16 | 1739.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 1731.00 | 1741.76 | 1738.98 | SL hit (close<static) qty=1.00 sl=1733.70 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 1728.90 | 1737.04 | 1737.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 15:15:00 | 1711.50 | 1727.69 | 1732.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 09:15:00 | 1729.80 | 1728.11 | 1732.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 1729.80 | 1728.11 | 1732.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1729.80 | 1728.11 | 1732.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:30:00 | 1731.40 | 1728.11 | 1732.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1736.00 | 1729.69 | 1732.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 1737.30 | 1729.69 | 1732.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1752.00 | 1734.15 | 1734.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 1757.10 | 1734.15 | 1734.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 1749.00 | 1737.12 | 1735.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 15:15:00 | 1758.00 | 1750.77 | 1744.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1728.60 | 1746.34 | 1743.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1728.60 | 1746.34 | 1743.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1728.60 | 1746.34 | 1743.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1728.60 | 1746.34 | 1743.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1742.70 | 1745.61 | 1743.18 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 1736.20 | 1740.77 | 1741.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 1725.30 | 1737.07 | 1739.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 1714.80 | 1711.70 | 1719.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:30:00 | 1715.00 | 1711.70 | 1719.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 1707.00 | 1705.25 | 1711.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 1706.50 | 1705.25 | 1711.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1705.90 | 1705.38 | 1711.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1690.40 | 1704.55 | 1709.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1715.60 | 1706.76 | 1710.20 | SL hit (close>static) qty=1.00 sl=1714.90 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 1738.90 | 1713.12 | 1712.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 13:15:00 | 1783.90 | 1727.27 | 1718.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 1845.00 | 1847.86 | 1818.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:30:00 | 1846.80 | 1847.15 | 1821.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1825.80 | 1839.48 | 1827.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 1825.80 | 1839.48 | 1827.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1824.90 | 1836.56 | 1826.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 1824.00 | 1836.56 | 1826.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1820.40 | 1832.39 | 1826.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:45:00 | 1834.70 | 1831.97 | 1826.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 1832.20 | 1832.02 | 1827.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 1792.90 | 1820.51 | 1823.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 1792.90 | 1820.51 | 1823.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 1782.30 | 1812.87 | 1819.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 1795.20 | 1791.26 | 1802.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1795.20 | 1791.26 | 1802.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1795.20 | 1791.26 | 1802.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 1795.00 | 1791.26 | 1802.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1776.80 | 1781.84 | 1792.39 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 1814.10 | 1791.00 | 1790.91 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 1785.00 | 1795.70 | 1797.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1768.30 | 1790.22 | 1794.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 1813.50 | 1794.88 | 1796.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 1813.50 | 1794.88 | 1796.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1813.50 | 1794.88 | 1796.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 1813.50 | 1794.88 | 1796.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 1814.10 | 1798.72 | 1797.85 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1796.00 | 1801.16 | 1801.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1791.20 | 1799.17 | 1800.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1794.90 | 1784.54 | 1790.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 1794.90 | 1784.54 | 1790.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1794.90 | 1784.54 | 1790.58 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 1819.50 | 1796.63 | 1795.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 1841.20 | 1809.38 | 1801.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1854.60 | 1856.37 | 1838.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:15:00 | 1851.40 | 1856.37 | 1838.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1825.60 | 1850.22 | 1837.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1825.60 | 1850.22 | 1837.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1831.80 | 1846.53 | 1836.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:30:00 | 1834.90 | 1840.55 | 1835.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 1841.30 | 1841.04 | 1836.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:45:00 | 1841.50 | 1838.70 | 1836.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 1837.10 | 1838.70 | 1836.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1838.00 | 1838.56 | 1837.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 1835.60 | 1838.56 | 1837.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1835.00 | 1837.85 | 1836.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1818.10 | 1837.85 | 1836.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 1801.30 | 1830.54 | 1833.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1801.30 | 1830.54 | 1833.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 10:15:00 | 1792.50 | 1812.18 | 1820.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 1808.90 | 1806.23 | 1814.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 1808.90 | 1806.23 | 1814.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1811.00 | 1807.19 | 1814.32 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1846.90 | 1821.64 | 1819.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 13:15:00 | 1854.60 | 1844.08 | 1835.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 1843.30 | 1843.93 | 1836.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 1843.30 | 1843.93 | 1836.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1839.60 | 1842.45 | 1836.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 1839.60 | 1842.45 | 1836.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1840.60 | 1842.08 | 1837.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 1841.40 | 1842.08 | 1837.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1841.50 | 1841.96 | 1837.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 1834.10 | 1841.96 | 1837.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1863.00 | 1870.19 | 1861.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1872.90 | 1870.19 | 1861.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 1843.70 | 1864.89 | 1859.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:00:00 | 1843.70 | 1864.89 | 1859.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 1823.60 | 1856.64 | 1856.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 1823.60 | 1856.64 | 1856.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 1851.10 | 1856.40 | 1856.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1838.70 | 1851.84 | 1854.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 1861.00 | 1853.67 | 1854.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 1861.00 | 1853.67 | 1854.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1861.00 | 1853.67 | 1854.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 1861.00 | 1853.67 | 1854.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 1850.10 | 1852.96 | 1854.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:15:00 | 1863.10 | 1852.96 | 1854.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 12:15:00 | 1874.60 | 1857.28 | 1856.33 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 11:15:00 | 1850.40 | 1856.93 | 1857.27 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 1870.00 | 1857.95 | 1857.42 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 1846.00 | 1855.56 | 1856.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 11:15:00 | 1842.30 | 1851.76 | 1854.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 1851.80 | 1849.42 | 1852.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 14:15:00 | 1851.80 | 1849.42 | 1852.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1851.80 | 1849.42 | 1852.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1851.80 | 1849.42 | 1852.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1865.00 | 1852.54 | 1853.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1851.00 | 1852.54 | 1853.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1856.10 | 1853.25 | 1853.87 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1860.10 | 1854.62 | 1854.43 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 1843.90 | 1854.67 | 1854.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 10:15:00 | 1840.00 | 1851.74 | 1853.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 15:15:00 | 1644.00 | 1632.86 | 1656.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 1634.10 | 1632.86 | 1656.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1588.00 | 1583.95 | 1598.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:45:00 | 1574.30 | 1580.68 | 1595.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:45:00 | 1571.50 | 1554.05 | 1562.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:00:00 | 1574.00 | 1536.58 | 1546.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 12:15:00 | 1495.58 | 1515.87 | 1530.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 12:15:00 | 1492.92 | 1515.87 | 1530.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 12:15:00 | 1495.30 | 1515.87 | 1530.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1505.20 | 1495.08 | 1514.04 | SL hit (close>ema200) qty=0.50 sl=1495.08 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 1557.60 | 1526.86 | 1524.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 11:15:00 | 1564.90 | 1542.52 | 1533.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1602.00 | 1606.04 | 1586.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:45:00 | 1592.20 | 1606.04 | 1586.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1605.60 | 1607.38 | 1590.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:45:00 | 1589.60 | 1607.38 | 1590.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1555.00 | 1596.91 | 1587.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1559.30 | 1596.91 | 1587.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1558.00 | 1589.13 | 1584.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 1566.90 | 1589.13 | 1584.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1571.00 | 1580.98 | 1581.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1533.50 | 1571.49 | 1576.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1626.40 | 1572.26 | 1573.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 1626.40 | 1572.26 | 1573.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1626.40 | 1572.26 | 1573.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1626.40 | 1572.26 | 1573.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1621.70 | 1582.15 | 1578.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 1674.90 | 1607.88 | 1590.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 1660.00 | 1661.77 | 1649.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 1635.00 | 1661.77 | 1649.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1634.20 | 1656.25 | 1648.48 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 1635.10 | 1644.30 | 1644.60 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 1648.40 | 1645.12 | 1644.94 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 1635.40 | 1643.18 | 1644.07 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1658.30 | 1646.20 | 1645.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1719.20 | 1675.42 | 1661.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1681.90 | 1686.93 | 1672.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 1681.90 | 1686.93 | 1672.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1678.10 | 1689.18 | 1677.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1674.00 | 1689.18 | 1677.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1690.00 | 1689.34 | 1678.86 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 1672.40 | 1679.48 | 1679.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 1669.70 | 1677.52 | 1678.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 1670.00 | 1666.75 | 1671.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 13:15:00 | 1670.00 | 1666.75 | 1671.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 1670.00 | 1666.75 | 1671.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:45:00 | 1669.00 | 1666.75 | 1671.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1651.80 | 1663.76 | 1669.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:45:00 | 1662.40 | 1663.76 | 1669.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1671.60 | 1654.18 | 1660.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1671.60 | 1654.18 | 1660.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1666.00 | 1656.54 | 1660.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1646.40 | 1656.54 | 1660.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1642.00 | 1599.91 | 1595.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1642.00 | 1599.91 | 1595.54 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1581.50 | 1603.71 | 1604.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 1515.50 | 1529.20 | 1546.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1535.70 | 1530.50 | 1545.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 1535.70 | 1530.50 | 1545.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1544.00 | 1535.20 | 1545.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 1545.00 | 1535.20 | 1545.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1526.00 | 1533.36 | 1543.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:30:00 | 1546.00 | 1533.36 | 1543.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1542.00 | 1535.98 | 1542.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:15:00 | 1537.00 | 1536.98 | 1542.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 1460.15 | 1491.26 | 1499.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-16 13:15:00 | 1383.30 | 1413.55 | 1443.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1450.00 | 1431.38 | 1430.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1454.10 | 1435.92 | 1433.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1426.70 | 1436.32 | 1434.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1426.70 | 1436.32 | 1434.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1426.70 | 1436.32 | 1434.16 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1414.10 | 1431.88 | 1432.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1410.90 | 1423.15 | 1427.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1342.00 | 1334.70 | 1364.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:45:00 | 1347.80 | 1334.70 | 1364.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1363.10 | 1341.77 | 1362.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 1361.30 | 1341.77 | 1362.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1398.50 | 1353.12 | 1366.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1392.30 | 1353.12 | 1366.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1398.70 | 1362.23 | 1369.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1398.70 | 1362.23 | 1369.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1446.40 | 1387.06 | 1379.36 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1367.60 | 1401.95 | 1402.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 1352.30 | 1379.87 | 1391.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 15:15:00 | 1305.00 | 1301.01 | 1320.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 09:15:00 | 1322.10 | 1301.01 | 1320.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1312.70 | 1303.35 | 1320.05 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1370.40 | 1331.85 | 1330.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 1378.70 | 1346.86 | 1337.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1523.20 | 1539.42 | 1510.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1523.20 | 1539.42 | 1510.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1523.20 | 1539.42 | 1510.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1530.90 | 1539.42 | 1510.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 09:15:00 | 1683.99 | 1652.26 | 1632.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 1633.60 | 1648.78 | 1648.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 1629.00 | 1641.57 | 1645.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1587.90 | 1577.91 | 1593.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1587.90 | 1577.91 | 1593.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1587.90 | 1577.91 | 1593.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 1596.20 | 1577.91 | 1593.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1610.90 | 1585.89 | 1593.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 1610.90 | 1585.89 | 1593.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1604.80 | 1589.67 | 1594.27 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1614.90 | 1598.61 | 1597.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1622.00 | 1604.26 | 1600.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1605.40 | 1636.75 | 1627.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 1605.40 | 1636.75 | 1627.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1605.40 | 1636.75 | 1627.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1605.40 | 1636.75 | 1627.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1606.20 | 1630.64 | 1625.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 1602.30 | 1630.64 | 1625.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1624.40 | 1627.26 | 1624.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 1624.40 | 1627.26 | 1624.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1626.90 | 1627.19 | 1625.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1690.30 | 1627.19 | 1625.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-29 11:15:00 | 1886.30 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-05-29 11:45:00 | 1885.50 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-05-29 13:30:00 | 1885.30 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2025-05-29 14:15:00 | 1886.20 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-06-03 10:45:00 | 1862.20 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-03 11:45:00 | 1859.70 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-06-04 09:15:00 | 1857.50 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1665.20 | 2025-06-23 10:15:00 | 1687.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-02 11:15:00 | 1859.40 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-02 11:45:00 | 1872.30 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-02 14:45:00 | 1859.40 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-03 11:00:00 | 1859.90 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-04 09:15:00 | 1851.30 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-07-04 12:00:00 | 1851.90 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-07-10 10:15:00 | 1777.10 | 2025-07-23 09:15:00 | 1729.10 | STOP_HIT | 1.00 | 2.70% |
| SELL | retest2 | 2025-07-10 10:45:00 | 1773.50 | 2025-07-23 09:15:00 | 1729.10 | STOP_HIT | 1.00 | 2.50% |
| SELL | retest2 | 2025-07-11 09:15:00 | 1768.00 | 2025-07-23 09:15:00 | 1729.10 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-07-29 09:15:00 | 1832.00 | 2025-08-01 11:15:00 | 1834.50 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-08-14 15:15:00 | 1780.30 | 2025-08-25 10:15:00 | 1750.80 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-09-12 12:15:00 | 1669.00 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-09-15 10:30:00 | 1671.50 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-09-15 12:00:00 | 1669.80 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-09-16 10:00:00 | 1670.40 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-09-16 11:15:00 | 1657.10 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-24 14:30:00 | 1653.30 | 2025-09-26 09:15:00 | 1570.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:30:00 | 1653.30 | 2025-09-29 09:15:00 | 1584.00 | STOP_HIT | 0.50 | 4.19% |
| BUY | retest2 | 2025-10-13 13:15:00 | 1619.80 | 2025-10-14 10:15:00 | 1599.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-13 15:15:00 | 1619.70 | 2025-10-14 10:15:00 | 1599.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-16 13:45:00 | 1573.20 | 2025-10-21 13:15:00 | 1601.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-10-17 11:45:00 | 1572.30 | 2025-10-21 13:15:00 | 1601.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-10-24 11:15:00 | 1698.00 | 2025-10-29 09:15:00 | 1630.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-10-24 12:00:00 | 1698.00 | 2025-10-29 09:15:00 | 1630.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-11-03 09:15:00 | 1728.00 | 2025-11-06 12:15:00 | 1702.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1691.10 | 2025-11-10 10:15:00 | 1746.50 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-11-13 09:15:00 | 1752.10 | 2025-11-13 10:15:00 | 1731.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1690.40 | 2025-11-24 09:15:00 | 1715.60 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-11-24 11:45:00 | 1698.50 | 2025-11-24 12:15:00 | 1738.90 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-11-28 11:45:00 | 1834.70 | 2025-12-01 09:15:00 | 1792.90 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-11-28 13:00:00 | 1832.20 | 2025-12-01 09:15:00 | 1792.90 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-12-16 13:30:00 | 1834.90 | 2025-12-18 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-12-16 14:45:00 | 1841.30 | 2025-12-18 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-12-17 11:45:00 | 1841.50 | 2025-12-18 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-12-17 14:00:00 | 1837.10 | 2025-12-18 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-01-20 10:45:00 | 1574.30 | 2026-01-27 12:15:00 | 1495.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 09:45:00 | 1571.50 | 2026-01-27 12:15:00 | 1492.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:00:00 | 1574.00 | 2026-01-27 12:15:00 | 1495.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 10:45:00 | 1574.30 | 2026-01-28 09:15:00 | 1505.20 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2026-01-22 09:45:00 | 1571.50 | 2026-01-28 09:15:00 | 1505.20 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2026-01-23 10:00:00 | 1574.00 | 2026-01-28 09:15:00 | 1505.20 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1646.40 | 2026-02-26 09:15:00 | 1642.00 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1537.00 | 2026-03-13 09:15:00 | 1460.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1537.00 | 2026-03-16 13:15:00 | 1383.30 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1530.90 | 2026-04-20 09:15:00 | 1683.99 | TARGET_HIT | 1.00 | 10.00% |
