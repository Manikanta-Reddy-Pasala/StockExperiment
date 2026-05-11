# ICICI Lombard General Insurance Company Ltd. (ICICIGI)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1820.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 91 |
| ALERT1 | 54 |
| ALERT2 | 54 |
| ALERT2_SKIP | 29 |
| ALERT3 | 160 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 61 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 42
- **Target hits / Stop hits / Partials:** 0 / 63 / 4
- **Avg / median % per leg:** 0.19% / -0.42%
- **Sum % (uncompounded):** 12.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 10 | 32.3% | 0 | 31 | 0 | 0.10% | 3.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 10 | 32.3% | 0 | 31 | 0 | 0.10% | 3.0% |
| SELL (all) | 36 | 15 | 41.7% | 0 | 32 | 4 | 0.27% | 9.9% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.02% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 14 | 41.2% | 0 | 30 | 4 | 0.29% | 9.8% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.02% | 0.0% |
| retest2 (combined) | 65 | 24 | 36.9% | 0 | 61 | 4 | 0.20% | 12.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1828.40 | 1801.27 | 1800.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1833.00 | 1807.61 | 1803.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 1843.60 | 1851.54 | 1836.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:45:00 | 1844.20 | 1851.54 | 1836.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 1840.10 | 1850.02 | 1839.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 1839.50 | 1850.02 | 1839.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 1849.00 | 1849.82 | 1840.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1874.80 | 1850.89 | 1842.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:15:00 | 1857.90 | 1863.23 | 1859.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:00:00 | 1856.20 | 1860.97 | 1858.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1868.60 | 1877.05 | 1877.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 1868.60 | 1877.05 | 1877.43 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1893.60 | 1878.30 | 1876.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 1901.10 | 1882.86 | 1878.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 09:15:00 | 1877.50 | 1886.39 | 1882.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 1877.50 | 1886.39 | 1882.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1877.50 | 1886.39 | 1882.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 1877.50 | 1886.39 | 1882.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1852.80 | 1879.67 | 1879.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 1852.80 | 1879.67 | 1879.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 1843.80 | 1872.50 | 1876.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 1839.40 | 1862.28 | 1870.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 12:15:00 | 1854.20 | 1844.00 | 1850.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 1854.20 | 1844.00 | 1850.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1854.20 | 1844.00 | 1850.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 1854.20 | 1844.00 | 1850.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1859.50 | 1847.10 | 1851.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 1859.50 | 1847.10 | 1851.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1855.00 | 1848.14 | 1850.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 1855.00 | 1848.14 | 1850.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1849.00 | 1848.32 | 1850.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 1854.40 | 1848.32 | 1850.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1860.00 | 1850.65 | 1851.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 1860.00 | 1850.65 | 1851.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1845.60 | 1849.64 | 1850.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:30:00 | 1860.00 | 1849.64 | 1850.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1850.00 | 1849.71 | 1850.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 1850.00 | 1849.71 | 1850.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1850.70 | 1849.91 | 1850.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1850.70 | 1849.91 | 1850.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1850.10 | 1849.95 | 1850.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1859.30 | 1849.95 | 1850.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 1858.10 | 1851.58 | 1851.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 1872.20 | 1856.65 | 1853.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 10:15:00 | 1869.40 | 1870.23 | 1863.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 11:00:00 | 1869.40 | 1870.23 | 1863.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1862.70 | 1868.73 | 1863.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 1860.90 | 1868.73 | 1863.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1861.50 | 1867.28 | 1862.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:00:00 | 1861.50 | 1867.28 | 1862.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1863.40 | 1866.50 | 1863.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 1871.30 | 1864.65 | 1862.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 1856.60 | 1862.53 | 1862.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 1856.60 | 1862.53 | 1862.68 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 14:15:00 | 1864.20 | 1862.86 | 1862.82 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 1860.00 | 1862.29 | 1862.56 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1875.00 | 1863.83 | 1863.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1881.00 | 1871.44 | 1867.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 1877.10 | 1877.17 | 1872.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 15:00:00 | 1877.10 | 1877.17 | 1872.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1892.60 | 1880.56 | 1874.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:15:00 | 1899.80 | 1880.56 | 1874.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 1944.00 | 1963.18 | 1965.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1944.00 | 1963.18 | 1965.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 1931.50 | 1950.29 | 1958.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 1926.10 | 1919.14 | 1935.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 10:00:00 | 1926.10 | 1919.14 | 1935.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 1925.10 | 1920.34 | 1934.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 1938.40 | 1920.34 | 1934.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1938.70 | 1923.92 | 1931.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1938.70 | 1923.92 | 1931.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1943.30 | 1927.79 | 1932.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1933.00 | 1927.79 | 1932.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 10:30:00 | 1931.30 | 1929.58 | 1932.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1943.30 | 1935.58 | 1934.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1943.30 | 1935.58 | 1934.90 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 1926.40 | 1934.16 | 1934.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 13:15:00 | 1920.00 | 1931.33 | 1933.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 11:15:00 | 1933.80 | 1926.96 | 1929.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 11:15:00 | 1933.80 | 1926.96 | 1929.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1933.80 | 1926.96 | 1929.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:45:00 | 1931.50 | 1926.96 | 1929.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1932.00 | 1927.97 | 1930.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:15:00 | 1932.20 | 1927.97 | 1930.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 1930.10 | 1928.40 | 1930.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:30:00 | 1935.00 | 1928.40 | 1930.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 1933.40 | 1929.40 | 1930.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 1933.40 | 1929.40 | 1930.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1929.60 | 1929.44 | 1930.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1934.70 | 1929.44 | 1930.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1933.90 | 1930.33 | 1930.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 1941.60 | 1930.33 | 1930.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1931.00 | 1930.46 | 1930.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 1928.00 | 1930.46 | 1930.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:00:00 | 1928.00 | 1929.88 | 1930.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 1942.80 | 1931.57 | 1930.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 1942.80 | 1931.57 | 1930.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 10:15:00 | 1962.30 | 1937.72 | 1933.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 2010.00 | 2010.68 | 2000.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 09:15:00 | 2009.50 | 2010.68 | 2000.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2040.10 | 2050.99 | 2039.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 2040.10 | 2050.99 | 2039.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2032.20 | 2047.23 | 2039.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 2032.20 | 2047.23 | 2039.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 2040.30 | 2045.85 | 2039.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:45:00 | 2042.60 | 2045.08 | 2039.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:30:00 | 2042.60 | 2043.40 | 2039.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 15:00:00 | 2042.60 | 2043.40 | 2039.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:45:00 | 2042.80 | 2043.99 | 2040.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2045.20 | 2044.23 | 2040.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 2045.20 | 2044.23 | 2040.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 2041.40 | 2043.67 | 2040.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 2041.40 | 2043.67 | 2040.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 2036.90 | 2042.31 | 2040.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 2036.90 | 2042.31 | 2040.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 2035.10 | 2040.87 | 2040.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 2035.10 | 2040.87 | 2040.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-03 14:15:00 | 2033.50 | 2039.40 | 2039.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 2033.50 | 2039.40 | 2039.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 2031.70 | 2037.15 | 2038.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 2037.80 | 2036.72 | 2037.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 2037.80 | 2036.72 | 2037.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2037.80 | 2036.72 | 2037.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 2037.80 | 2036.72 | 2037.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 2035.00 | 2036.38 | 2037.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 2035.00 | 2036.38 | 2037.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 2049.00 | 2038.90 | 2038.63 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 2031.70 | 2037.72 | 2038.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 2026.70 | 2034.68 | 2036.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 11:15:00 | 2031.30 | 2030.05 | 2033.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 11:15:00 | 2031.30 | 2030.05 | 2033.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2031.30 | 2030.05 | 2033.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 2031.30 | 2030.05 | 2033.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 2039.30 | 2031.90 | 2033.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 2039.30 | 2031.90 | 2033.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 2034.70 | 2032.46 | 2034.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 2030.70 | 2034.06 | 2034.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 2024.30 | 2014.86 | 2013.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 2024.30 | 2014.86 | 2013.79 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 1995.20 | 2010.47 | 2011.96 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 2014.00 | 2012.90 | 2012.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 2029.40 | 2016.20 | 2014.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 2012.90 | 2015.54 | 2014.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 10:15:00 | 2012.90 | 2015.54 | 2014.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2012.90 | 2015.54 | 2014.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 2012.90 | 2015.54 | 2014.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 11:15:00 | 2000.00 | 2012.43 | 2012.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 12:15:00 | 1996.70 | 2009.28 | 2011.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 15:15:00 | 2007.80 | 2006.57 | 2009.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 15:15:00 | 2007.80 | 2006.57 | 2009.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 2007.80 | 2006.57 | 2009.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 2013.00 | 2006.57 | 2009.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2026.00 | 2010.46 | 2010.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 2029.60 | 2010.46 | 2010.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 2042.00 | 2016.76 | 2013.74 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 1976.70 | 2010.72 | 2011.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 1969.50 | 2002.48 | 2008.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 1925.60 | 1925.32 | 1945.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 11:00:00 | 1925.60 | 1925.32 | 1945.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1951.30 | 1935.06 | 1943.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 1951.30 | 1935.06 | 1943.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1954.20 | 1938.89 | 1944.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1958.10 | 1938.89 | 1944.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 1965.00 | 1949.25 | 1948.51 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 1940.80 | 1950.04 | 1950.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1914.00 | 1936.05 | 1942.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 1912.20 | 1908.67 | 1920.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 13:00:00 | 1912.20 | 1908.67 | 1920.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1921.20 | 1911.18 | 1920.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 1921.20 | 1911.18 | 1920.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1921.10 | 1913.16 | 1920.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 1923.90 | 1913.16 | 1920.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 1916.00 | 1913.73 | 1920.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:45:00 | 1912.30 | 1917.18 | 1920.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 13:30:00 | 1915.00 | 1910.43 | 1913.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:15:00 | 1915.20 | 1911.84 | 1914.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 15:15:00 | 1923.00 | 1914.40 | 1913.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 1923.00 | 1914.40 | 1913.70 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1902.30 | 1911.98 | 1912.66 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 1919.40 | 1913.72 | 1913.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 1923.80 | 1915.74 | 1914.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1909.20 | 1919.93 | 1917.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 1909.20 | 1919.93 | 1917.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1909.20 | 1919.93 | 1917.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1909.20 | 1919.93 | 1917.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1904.80 | 1916.91 | 1916.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1904.80 | 1916.91 | 1916.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1888.00 | 1911.12 | 1913.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1877.60 | 1904.42 | 1910.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1890.00 | 1880.26 | 1890.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 13:15:00 | 1890.00 | 1880.26 | 1890.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1890.00 | 1880.26 | 1890.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1890.00 | 1880.26 | 1890.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1896.20 | 1883.45 | 1891.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1896.20 | 1883.45 | 1891.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1895.00 | 1885.76 | 1891.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 1878.80 | 1885.76 | 1891.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 1901.10 | 1888.72 | 1890.93 | SL hit (close>static) qty=1.00 sl=1900.50 alert=retest2 |

### Cycle 29 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 1897.30 | 1893.17 | 1892.63 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 1867.50 | 1888.04 | 1890.34 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 13:15:00 | 1900.00 | 1887.32 | 1886.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 1903.90 | 1890.64 | 1888.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 1889.90 | 1892.79 | 1889.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 1889.90 | 1892.79 | 1889.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1889.90 | 1892.79 | 1889.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 1889.90 | 1892.79 | 1889.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1889.20 | 1892.07 | 1889.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 1889.20 | 1892.07 | 1889.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1878.20 | 1889.30 | 1888.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 1878.20 | 1889.30 | 1888.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1895.10 | 1890.46 | 1889.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 1899.60 | 1890.46 | 1889.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 14:30:00 | 1901.20 | 1892.83 | 1890.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 1899.70 | 1892.23 | 1890.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:00:00 | 1897.90 | 1893.36 | 1891.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1894.70 | 1893.63 | 1891.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:15:00 | 1892.80 | 1893.63 | 1891.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1890.10 | 1892.92 | 1891.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 1885.40 | 1892.92 | 1891.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 1896.20 | 1893.58 | 1891.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:30:00 | 1887.30 | 1893.58 | 1891.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1907.70 | 1899.12 | 1895.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 11:00:00 | 1918.50 | 1905.36 | 1900.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 1916.60 | 1913.01 | 1906.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 1898.40 | 1904.00 | 1904.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 1898.40 | 1904.00 | 1904.42 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1973.90 | 1916.81 | 1910.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 1978.00 | 1929.05 | 1916.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 1945.20 | 1949.92 | 1932.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 1945.20 | 1949.92 | 1932.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1928.20 | 1944.79 | 1932.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 1928.20 | 1944.79 | 1932.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1935.00 | 1942.83 | 1933.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:30:00 | 1942.40 | 1944.29 | 1934.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:00:00 | 1937.80 | 1962.67 | 1961.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 1937.10 | 1957.56 | 1958.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 1937.10 | 1957.56 | 1958.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 1916.00 | 1949.25 | 1955.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 1925.20 | 1924.20 | 1937.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 1925.20 | 1924.20 | 1937.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1926.70 | 1925.93 | 1936.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 1934.30 | 1925.93 | 1936.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1833.50 | 1839.25 | 1851.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:00:00 | 1819.30 | 1835.26 | 1848.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 1819.40 | 1831.87 | 1843.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:15:00 | 1820.00 | 1830.38 | 1841.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1873.90 | 1832.92 | 1833.75 | SL hit (close>static) qty=1.00 sl=1863.40 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 1859.70 | 1838.28 | 1836.11 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 1833.80 | 1844.19 | 1844.93 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1848.10 | 1841.79 | 1840.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 1860.00 | 1848.08 | 1844.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1846.90 | 1849.84 | 1846.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1846.90 | 1849.84 | 1846.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1846.90 | 1849.84 | 1846.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1842.20 | 1849.84 | 1846.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1851.50 | 1850.17 | 1846.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:45:00 | 1860.60 | 1852.94 | 1848.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 1868.00 | 1891.46 | 1892.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 1868.00 | 1891.46 | 1892.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 10:15:00 | 1865.00 | 1877.71 | 1882.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 15:15:00 | 1871.60 | 1867.73 | 1874.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:15:00 | 1877.00 | 1867.73 | 1874.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1884.80 | 1871.14 | 1875.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:30:00 | 1879.00 | 1871.14 | 1875.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1884.10 | 1873.73 | 1876.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1886.10 | 1873.73 | 1876.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 1886.40 | 1877.11 | 1877.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:00:00 | 1886.40 | 1877.11 | 1877.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 1886.60 | 1879.01 | 1878.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 10:15:00 | 1891.70 | 1883.81 | 1880.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1884.80 | 1889.23 | 1885.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1884.80 | 1889.23 | 1885.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1884.80 | 1889.23 | 1885.48 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 1876.40 | 1882.15 | 1882.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 1864.70 | 1877.23 | 1880.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 11:15:00 | 1880.00 | 1877.59 | 1879.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 1880.00 | 1877.59 | 1879.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1880.00 | 1877.59 | 1879.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:30:00 | 1879.00 | 1877.59 | 1879.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 1880.00 | 1878.07 | 1879.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:45:00 | 1880.00 | 1878.07 | 1879.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1880.50 | 1878.56 | 1879.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 1880.50 | 1878.56 | 1879.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1880.30 | 1878.90 | 1879.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 1882.90 | 1878.90 | 1879.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1886.00 | 1880.32 | 1880.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 1899.20 | 1880.32 | 1880.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 09:15:00 | 1900.10 | 1884.28 | 1882.23 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1873.80 | 1888.63 | 1889.96 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1897.80 | 1890.38 | 1890.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1906.10 | 1894.54 | 1892.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 1916.00 | 1924.71 | 1917.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 13:15:00 | 1916.00 | 1924.71 | 1917.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1916.00 | 1924.71 | 1917.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 1916.00 | 1924.71 | 1917.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1903.50 | 1920.47 | 1916.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1903.50 | 1920.47 | 1916.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1903.10 | 1917.00 | 1914.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1904.60 | 1917.00 | 1914.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1884.50 | 1908.02 | 1911.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1876.40 | 1894.54 | 1903.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 1885.20 | 1880.89 | 1891.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 13:00:00 | 1885.20 | 1880.89 | 1891.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1890.00 | 1882.18 | 1889.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1894.50 | 1882.18 | 1889.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1891.10 | 1883.96 | 1889.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 1885.20 | 1885.68 | 1889.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 2000.00 | 1880.69 | 1869.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 2000.00 | 1880.69 | 1869.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 2006.70 | 1939.24 | 1901.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1997.00 | 2002.40 | 1972.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 1997.00 | 2002.40 | 1972.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 2004.40 | 2007.49 | 2000.48 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 1984.20 | 1996.96 | 1997.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 1978.20 | 1993.21 | 1995.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 1990.90 | 1986.23 | 1990.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 1990.90 | 1986.23 | 1990.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1990.90 | 1986.23 | 1990.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1990.90 | 1986.23 | 1990.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1995.60 | 1988.11 | 1991.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 1995.60 | 1988.11 | 1991.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1993.40 | 1989.16 | 1991.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:30:00 | 1992.80 | 1989.16 | 1991.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 1995.00 | 1990.33 | 1991.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 2003.50 | 1990.33 | 1991.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 2010.00 | 1994.26 | 1993.36 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 1987.20 | 1992.28 | 1992.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 15:15:00 | 1981.60 | 1989.09 | 1991.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1991.30 | 1989.53 | 1991.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1991.30 | 1989.53 | 1991.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1991.30 | 1989.53 | 1991.26 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 2006.10 | 1993.72 | 1992.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 12:15:00 | 2016.00 | 1998.18 | 1995.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 2014.20 | 2021.20 | 2012.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 2014.20 | 2021.20 | 2012.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2014.20 | 2021.20 | 2012.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 2014.20 | 2021.20 | 2012.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 2015.10 | 2019.98 | 2013.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 2013.30 | 2019.98 | 2013.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 2013.10 | 2018.60 | 2013.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:45:00 | 2015.00 | 2018.60 | 2013.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 2012.60 | 2017.40 | 2012.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:30:00 | 2014.00 | 2017.40 | 2012.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 2008.90 | 2015.70 | 2012.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 2005.70 | 2015.70 | 2012.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 2015.00 | 2015.56 | 2012.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:30:00 | 2008.70 | 2015.56 | 2012.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 2015.00 | 2015.45 | 2013.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 2009.20 | 2015.45 | 2013.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 2013.50 | 2015.06 | 2013.07 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1992.60 | 2009.27 | 2011.03 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 2032.60 | 2011.34 | 2009.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 10:15:00 | 2036.20 | 2016.31 | 2012.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 2014.60 | 2030.17 | 2023.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 2014.60 | 2030.17 | 2023.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2014.60 | 2030.17 | 2023.20 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 2003.80 | 2017.64 | 2018.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 2001.50 | 2012.26 | 2015.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 2004.60 | 2000.01 | 2007.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 2004.60 | 2000.01 | 2007.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 2004.60 | 2000.01 | 2007.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 2008.90 | 2000.01 | 2007.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 2018.60 | 2003.73 | 2008.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 2020.00 | 2003.73 | 2008.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 2020.10 | 2007.00 | 2009.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 2026.00 | 2007.00 | 2009.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 2024.00 | 2012.58 | 2011.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 2035.90 | 2018.78 | 2014.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 15:15:00 | 2021.40 | 2022.01 | 2017.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:15:00 | 2007.00 | 2022.01 | 2017.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1998.60 | 2017.32 | 2015.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 1998.60 | 2017.32 | 2015.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1993.10 | 2012.48 | 2013.63 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 2016.20 | 2011.29 | 2010.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 2020.80 | 2013.19 | 2011.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 2032.90 | 2035.56 | 2027.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 2032.90 | 2035.56 | 2027.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 2031.20 | 2034.60 | 2028.16 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 2004.20 | 2022.30 | 2023.63 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 2028.60 | 2022.90 | 2022.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 2037.20 | 2025.76 | 2023.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 2024.00 | 2029.04 | 2026.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 2024.00 | 2029.04 | 2026.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 2024.00 | 2029.04 | 2026.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 2024.00 | 2029.04 | 2026.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2030.50 | 2029.33 | 2027.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 2019.60 | 2029.33 | 2027.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2022.40 | 2027.95 | 2026.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 2031.10 | 2029.38 | 2027.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 2016.60 | 2033.56 | 2035.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 2016.60 | 2033.56 | 2035.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 2002.30 | 2023.20 | 2029.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 2008.00 | 2001.90 | 2012.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 2008.00 | 2001.90 | 2012.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 2008.00 | 2001.90 | 2012.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 2008.50 | 2001.90 | 2012.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 2001.00 | 2001.72 | 2011.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 2018.20 | 2001.72 | 2011.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2020.50 | 2001.18 | 2006.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 2020.50 | 2001.18 | 2006.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 2006.10 | 2002.17 | 2006.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 2004.00 | 2003.33 | 2006.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:15:00 | 2002.50 | 2003.33 | 2006.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 2004.10 | 2006.07 | 2006.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 1990.00 | 1983.26 | 1982.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 1990.00 | 1983.26 | 1982.75 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1956.40 | 1977.89 | 1980.36 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 1987.30 | 1980.39 | 1980.37 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1955.50 | 1976.61 | 1978.73 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1981.50 | 1976.37 | 1976.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 1984.20 | 1977.93 | 1977.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 1971.00 | 1978.89 | 1978.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 11:15:00 | 1971.00 | 1978.89 | 1978.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1971.00 | 1978.89 | 1978.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 1971.00 | 1978.89 | 1978.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1978.70 | 1978.85 | 1978.27 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 1974.30 | 1977.79 | 1977.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 15:15:00 | 1970.20 | 1976.27 | 1977.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 1954.00 | 1953.51 | 1961.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 1954.00 | 1953.51 | 1961.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1954.00 | 1953.51 | 1961.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 1942.10 | 1953.19 | 1960.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 14:15:00 | 1953.50 | 1938.82 | 1938.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 1953.50 | 1938.82 | 1938.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 10:15:00 | 1962.50 | 1948.10 | 1943.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 13:15:00 | 1947.80 | 1950.59 | 1945.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:00:00 | 1947.80 | 1950.59 | 1945.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1952.90 | 1951.06 | 1946.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 1944.30 | 1951.06 | 1946.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1946.70 | 1950.18 | 1946.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 1953.90 | 1950.18 | 1946.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 1934.60 | 1945.13 | 1944.93 | SL hit (close<static) qty=1.00 sl=1940.70 alert=retest2 |

### Cycle 66 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 1940.50 | 1944.21 | 1944.52 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 1953.90 | 1945.98 | 1944.90 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 11:15:00 | 1936.60 | 1944.89 | 1945.75 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 1951.90 | 1946.81 | 1946.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 1954.00 | 1948.92 | 1947.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 1956.10 | 1956.46 | 1953.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 12:15:00 | 1956.10 | 1956.46 | 1953.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1956.10 | 1956.46 | 1953.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:45:00 | 1955.70 | 1956.46 | 1953.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1959.40 | 1957.14 | 1954.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 1955.50 | 1957.14 | 1954.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1958.80 | 1957.93 | 1955.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:45:00 | 1951.70 | 1957.93 | 1955.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1945.00 | 1955.34 | 1954.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1945.00 | 1955.34 | 1954.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 1943.30 | 1952.93 | 1953.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 1940.20 | 1948.51 | 1951.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 14:15:00 | 1950.40 | 1948.89 | 1951.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 1950.40 | 1948.89 | 1951.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1953.00 | 1949.71 | 1951.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 1944.90 | 1949.71 | 1951.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1941.80 | 1948.13 | 1950.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 1938.80 | 1946.26 | 1949.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1933.50 | 1943.99 | 1946.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:30:00 | 1937.50 | 1941.80 | 1944.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 1938.20 | 1941.80 | 1944.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 1940.30 | 1941.50 | 1944.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:30:00 | 1947.70 | 1941.50 | 1944.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1950.00 | 1943.20 | 1945.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 1950.00 | 1943.20 | 1945.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1955.10 | 1945.58 | 1945.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 1955.10 | 1945.58 | 1945.96 | SL hit (close>static) qty=1.00 sl=1953.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 1955.00 | 1947.47 | 1946.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1979.90 | 1960.98 | 1956.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 11:15:00 | 1970.00 | 1971.43 | 1965.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 11:30:00 | 1969.60 | 1971.43 | 1965.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1964.60 | 1970.07 | 1965.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 1964.60 | 1970.07 | 1965.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1967.10 | 1969.47 | 1965.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1967.10 | 1969.47 | 1965.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1980.70 | 1971.72 | 1967.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1984.60 | 1973.18 | 1968.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 1984.30 | 1975.22 | 1969.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:30:00 | 1988.00 | 1980.72 | 1972.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 15:15:00 | 1967.20 | 1981.28 | 1982.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 1967.20 | 1981.28 | 1982.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 1946.50 | 1974.32 | 1979.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1896.80 | 1895.94 | 1915.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 1896.80 | 1895.94 | 1915.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1907.00 | 1899.85 | 1912.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1912.00 | 1899.85 | 1912.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1913.00 | 1902.48 | 1912.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1907.80 | 1902.48 | 1912.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1905.60 | 1903.10 | 1911.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1890.90 | 1901.78 | 1910.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 15:15:00 | 1885.00 | 1874.96 | 1874.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 15:15:00 | 1885.00 | 1874.96 | 1874.83 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 1871.70 | 1874.31 | 1874.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1860.90 | 1869.70 | 1872.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 10:15:00 | 1823.90 | 1823.83 | 1834.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:00:00 | 1823.90 | 1823.83 | 1834.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1813.40 | 1802.72 | 1812.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1825.90 | 1802.72 | 1812.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1822.50 | 1806.67 | 1812.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1823.40 | 1806.67 | 1812.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1834.00 | 1812.14 | 1814.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1834.00 | 1812.14 | 1814.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 1820.00 | 1816.51 | 1816.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 1824.90 | 1819.18 | 1818.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 1819.50 | 1820.33 | 1818.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 1819.50 | 1820.33 | 1818.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1819.50 | 1820.33 | 1818.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 1812.50 | 1820.33 | 1818.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1821.00 | 1820.47 | 1819.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 1818.60 | 1820.47 | 1819.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1812.00 | 1819.26 | 1818.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 1812.00 | 1819.26 | 1818.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 1814.90 | 1818.39 | 1818.43 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1831.80 | 1817.83 | 1817.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 13:15:00 | 1841.00 | 1825.81 | 1821.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 1826.00 | 1830.16 | 1825.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 1826.00 | 1830.16 | 1825.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1826.00 | 1830.16 | 1825.02 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 13:15:00 | 1816.00 | 1821.42 | 1821.89 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1830.50 | 1823.24 | 1822.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 1831.00 | 1824.79 | 1823.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1849.30 | 1852.33 | 1842.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:30:00 | 1846.70 | 1852.33 | 1842.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 1843.90 | 1850.64 | 1843.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:00:00 | 1843.90 | 1850.64 | 1843.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1840.40 | 1848.59 | 1842.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:30:00 | 1835.50 | 1848.59 | 1842.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1851.20 | 1849.11 | 1843.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 14:15:00 | 1853.10 | 1849.11 | 1843.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 1853.10 | 1849.91 | 1844.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 1830.50 | 1845.54 | 1843.84 | SL hit (close<static) qty=1.00 sl=1840.00 alert=retest2 |

### Cycle 80 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 1908.60 | 1936.50 | 1940.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 1898.80 | 1924.93 | 1934.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1939.30 | 1921.44 | 1929.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 1939.30 | 1921.44 | 1929.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1939.30 | 1921.44 | 1929.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 1939.30 | 1921.44 | 1929.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1928.30 | 1922.81 | 1929.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:00:00 | 1924.20 | 1923.09 | 1929.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1955.00 | 1934.14 | 1932.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1955.00 | 1934.14 | 1932.55 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1908.50 | 1929.78 | 1932.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 1904.30 | 1920.89 | 1927.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 10:15:00 | 1911.10 | 1907.67 | 1917.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 11:00:00 | 1911.10 | 1907.67 | 1917.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1871.40 | 1857.97 | 1871.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1871.40 | 1857.97 | 1871.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1879.40 | 1862.26 | 1872.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1868.80 | 1862.26 | 1872.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 1859.70 | 1862.61 | 1871.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 13:15:00 | 1886.50 | 1871.71 | 1873.09 | SL hit (close>static) qty=1.00 sl=1880.90 alert=retest2 |

### Cycle 83 — BUY (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 15:15:00 | 1878.50 | 1874.36 | 1874.13 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1859.80 | 1871.45 | 1872.83 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 1888.40 | 1872.99 | 1872.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 1892.70 | 1878.26 | 1875.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 14:15:00 | 1888.80 | 1889.13 | 1882.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-10 15:00:00 | 1888.80 | 1889.13 | 1882.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1872.40 | 1885.28 | 1881.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 1874.60 | 1885.28 | 1881.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 1884.20 | 1885.07 | 1882.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 11:15:00 | 1892.00 | 1885.07 | 1882.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 15:15:00 | 1870.00 | 1881.78 | 1882.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 1870.00 | 1881.78 | 1882.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1849.20 | 1875.26 | 1879.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 15:15:00 | 1860.00 | 1859.23 | 1867.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1849.20 | 1859.23 | 1867.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:30:00 | 1852.20 | 1854.61 | 1863.22 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1850.40 | 1834.19 | 1843.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 1850.40 | 1834.19 | 1843.82 | SL hit (close>ema400) qty=1.00 sl=1843.82 alert=retest1 |

### Cycle 87 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 1773.80 | 1757.19 | 1755.97 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 1740.00 | 1755.74 | 1756.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1699.30 | 1738.42 | 1747.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1729.30 | 1719.36 | 1730.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1729.30 | 1719.36 | 1730.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1729.30 | 1719.36 | 1730.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1725.70 | 1719.36 | 1730.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 1724.20 | 1720.33 | 1729.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 10:15:00 | 1639.41 | 1686.37 | 1707.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 11:15:00 | 1637.99 | 1675.92 | 1701.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 1690.20 | 1674.58 | 1693.73 | SL hit (close>ema200) qty=0.50 sl=1674.58 alert=retest2 |

### Cycle 89 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 1718.80 | 1701.19 | 1699.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 1723.70 | 1711.22 | 1705.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1745.80 | 1748.39 | 1734.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 1745.80 | 1748.39 | 1734.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1745.80 | 1748.39 | 1734.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:30:00 | 1734.10 | 1748.39 | 1734.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1768.30 | 1779.44 | 1768.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 1788.00 | 1780.04 | 1772.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 1852.20 | 1864.74 | 1865.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 1852.20 | 1864.74 | 1865.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 1846.90 | 1861.18 | 1863.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 12:15:00 | 1830.90 | 1826.79 | 1836.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 13:00:00 | 1830.90 | 1826.79 | 1836.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 1814.00 | 1824.23 | 1834.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:30:00 | 1818.50 | 1824.23 | 1834.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1780.00 | 1772.98 | 1781.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 1797.80 | 1778.96 | 1783.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1797.30 | 1782.63 | 1784.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:15:00 | 1794.30 | 1782.63 | 1784.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 13:15:00 | 1772.60 | 1764.38 | 1764.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 1772.60 | 1764.38 | 1764.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1775.90 | 1766.69 | 1765.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 1835.00 | 1839.63 | 1822.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:00:00 | 1835.00 | 1839.63 | 1822.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1821.00 | 1835.91 | 1822.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 1810.40 | 1835.91 | 1822.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1822.10 | 1833.15 | 1822.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 1820.70 | 1833.15 | 1822.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1826.40 | 1831.80 | 1822.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:15:00 | 1820.00 | 1831.80 | 1822.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1820.00 | 1829.44 | 1822.20 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 09:15:00 | 1874.80 | 2025-05-22 09:15:00 | 1868.60 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-05-16 15:15:00 | 1857.90 | 2025-05-22 09:15:00 | 1868.60 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-05-19 10:00:00 | 1856.20 | 2025-05-22 09:15:00 | 1868.60 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-06-03 09:15:00 | 1871.30 | 2025-06-03 13:15:00 | 1856.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-06 10:15:00 | 1899.80 | 2025-06-11 13:15:00 | 1944.00 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1933.00 | 2025-06-16 13:15:00 | 1943.30 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-06-16 10:30:00 | 1931.30 | 2025-06-16 13:15:00 | 1943.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-06-19 11:15:00 | 1928.00 | 2025-06-20 09:15:00 | 1942.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-06-19 15:00:00 | 1928.00 | 2025-06-20 09:15:00 | 1942.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-07-02 12:45:00 | 2042.60 | 2025-07-03 14:15:00 | 2033.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-02 14:30:00 | 2042.60 | 2025-07-03 14:15:00 | 2033.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-02 15:00:00 | 2042.60 | 2025-07-03 14:15:00 | 2033.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-03 09:45:00 | 2042.80 | 2025-07-03 14:15:00 | 2033.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-09 09:15:00 | 2030.70 | 2025-07-14 10:15:00 | 2024.30 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1912.30 | 2025-07-30 15:15:00 | 1923.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-29 13:30:00 | 1915.00 | 2025-07-30 15:15:00 | 1923.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-07-29 15:15:00 | 1915.20 | 2025-07-30 15:15:00 | 1923.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-08-05 09:15:00 | 1878.80 | 2025-08-05 12:15:00 | 1901.10 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-08-08 13:15:00 | 1899.60 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-08-08 14:30:00 | 1901.20 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-08-11 09:15:00 | 1899.70 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-08-11 10:00:00 | 1897.90 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-08-13 11:00:00 | 1918.50 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-08-14 09:15:00 | 1916.60 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-19 11:30:00 | 1942.40 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-22 11:00:00 | 1937.80 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-09-02 11:00:00 | 1819.30 | 2025-09-04 09:15:00 | 1873.90 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-09-02 14:15:00 | 1819.40 | 2025-09-04 09:15:00 | 1873.90 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-09-02 15:15:00 | 1820.00 | 2025-09-04 09:15:00 | 1873.90 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-09-11 11:45:00 | 1860.60 | 2025-09-17 13:15:00 | 1868.00 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-10-10 12:15:00 | 1885.20 | 2025-10-15 09:15:00 | 2000.00 | STOP_HIT | 1.00 | -6.09% |
| BUY | retest2 | 2025-11-19 10:30:00 | 2031.10 | 2025-11-21 10:15:00 | 2016.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-11-26 11:30:00 | 2004.00 | 2025-12-01 15:15:00 | 1990.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-11-26 12:15:00 | 2002.50 | 2025-12-01 15:15:00 | 1990.00 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-11-27 09:15:00 | 2004.10 | 2025-12-01 15:15:00 | 1990.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-12-09 14:15:00 | 1942.10 | 2025-12-15 14:15:00 | 1953.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-17 09:15:00 | 1953.90 | 2025-12-17 11:15:00 | 1934.60 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-29 11:00:00 | 1938.80 | 2025-12-30 14:15:00 | 1955.10 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-30 09:15:00 | 1933.50 | 2025-12-30 14:15:00 | 1955.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-30 11:30:00 | 1937.50 | 2025-12-30 14:15:00 | 1955.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-30 12:00:00 | 1938.20 | 2025-12-30 14:15:00 | 1955.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1984.60 | 2026-01-07 15:15:00 | 1967.20 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-06 09:45:00 | 1984.30 | 2026-01-07 15:15:00 | 1967.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-06 10:30:00 | 1988.00 | 2026-01-07 15:15:00 | 1967.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1890.90 | 2026-01-19 15:15:00 | 1885.00 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-02-04 14:15:00 | 1853.10 | 2026-02-05 10:15:00 | 1830.50 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-02-04 15:00:00 | 1853.10 | 2026-02-05 10:15:00 | 1830.50 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-02-05 13:15:00 | 1865.20 | 2026-02-24 11:15:00 | 1908.60 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2026-02-06 11:00:00 | 1853.20 | 2026-02-24 11:15:00 | 1908.60 | STOP_HIT | 1.00 | 2.99% |
| BUY | retest2 | 2026-02-09 12:00:00 | 1874.70 | 2026-02-24 11:15:00 | 1908.60 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2026-02-10 09:45:00 | 1874.10 | 2026-02-24 11:15:00 | 1908.60 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2026-02-25 12:00:00 | 1924.20 | 2026-02-26 09:15:00 | 1955.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1868.80 | 2026-03-06 13:15:00 | 1886.50 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-03-06 09:45:00 | 1859.70 | 2026-03-06 13:15:00 | 1886.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-03-11 11:15:00 | 1892.00 | 2026-03-11 15:15:00 | 1870.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest1 | 2026-03-13 09:15:00 | 1849.20 | 2026-03-16 14:15:00 | 1850.40 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest1 | 2026-03-13 11:30:00 | 1852.20 | 2026-03-16 14:15:00 | 1850.40 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2026-03-18 09:30:00 | 1824.90 | 2026-03-23 09:15:00 | 1733.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 14:45:00 | 1830.40 | 2026-03-23 09:15:00 | 1738.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 09:30:00 | 1824.90 | 2026-03-24 11:15:00 | 1731.10 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2026-03-18 14:45:00 | 1830.40 | 2026-03-24 11:15:00 | 1731.10 | STOP_HIT | 0.50 | 5.43% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1725.70 | 2026-04-02 10:15:00 | 1639.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1724.20 | 2026-04-02 11:15:00 | 1637.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1725.70 | 2026-04-02 14:15:00 | 1690.20 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1724.20 | 2026-04-02 14:15:00 | 1690.20 | STOP_HIT | 0.50 | 1.97% |
| BUY | retest2 | 2026-04-13 15:15:00 | 1788.00 | 2026-04-21 11:15:00 | 1852.20 | STOP_HIT | 1.00 | 3.59% |
| SELL | retest2 | 2026-04-29 11:15:00 | 1794.30 | 2026-05-05 13:15:00 | 1772.60 | STOP_HIT | 1.00 | 1.21% |
