# ACC Ltd. (ACC)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1393.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 60 |
| ALERT1 | 43 |
| ALERT2 | 41 |
| ALERT2_SKIP | 18 |
| ALERT3 | 114 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 81 |
| PARTIAL | 10 |
| TARGET_HIT | 0 |
| STOP_HIT | 82 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 92 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 55
- **Target hits / Stop hits / Partials:** 0 / 82 / 10
- **Avg / median % per leg:** 0.60% / -0.28%
- **Sum % (uncompounded):** 55.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 5 | 17.2% | 0 | 29 | 0 | -0.46% | -13.2% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.06% | 0.1% |
| BUY @ 3rd Alert (retest2) | 28 | 4 | 14.3% | 0 | 28 | 0 | -0.48% | -13.3% |
| SELL (all) | 63 | 32 | 50.8% | 0 | 53 | 10 | 1.09% | 68.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 63 | 32 | 50.8% | 0 | 53 | 10 | 1.09% | 68.5% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.06% | 0.1% |
| retest2 (combined) | 91 | 36 | 39.6% | 0 | 81 | 10 | 0.61% | 55.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1854.00 | 1833.14 | 1832.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1864.70 | 1853.62 | 1844.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 1853.20 | 1857.44 | 1849.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 1853.20 | 1857.44 | 1849.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1920.00 | 1934.75 | 1924.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 1920.00 | 1934.75 | 1924.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1920.50 | 1931.90 | 1924.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 1924.60 | 1931.90 | 1924.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1939.60 | 1933.44 | 1925.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 1941.70 | 1932.18 | 1926.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:15:00 | 1942.00 | 1933.84 | 1927.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 1942.20 | 1935.51 | 1929.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 1941.90 | 1935.67 | 1930.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1928.80 | 1934.29 | 1930.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 1928.80 | 1934.29 | 1930.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1932.70 | 1933.98 | 1930.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 1929.50 | 1933.98 | 1930.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1935.40 | 1934.26 | 1930.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:00:00 | 1937.50 | 1934.91 | 1931.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:30:00 | 1939.50 | 1937.07 | 1932.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:30:00 | 1937.00 | 1951.98 | 1950.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 1937.70 | 1951.98 | 1950.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 1926.30 | 1933.89 | 1939.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 1926.90 | 1924.59 | 1931.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 1926.90 | 1924.59 | 1931.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1882.80 | 1884.24 | 1894.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:45:00 | 1872.10 | 1880.22 | 1889.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 13:30:00 | 1875.00 | 1879.36 | 1888.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 1875.00 | 1879.36 | 1888.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1872.30 | 1879.30 | 1886.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1878.00 | 1874.66 | 1880.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 1878.00 | 1874.66 | 1880.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1885.70 | 1876.92 | 1880.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 1887.00 | 1876.92 | 1880.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1891.10 | 1879.76 | 1881.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:00:00 | 1891.10 | 1879.76 | 1881.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 1890.90 | 1883.88 | 1883.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 1890.90 | 1883.88 | 1883.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 1890.90 | 1883.88 | 1883.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 1890.90 | 1883.88 | 1883.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1890.90 | 1883.88 | 1883.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 1897.20 | 1888.42 | 1885.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 1913.30 | 1913.45 | 1906.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 15:00:00 | 1913.30 | 1913.45 | 1906.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1898.80 | 1912.24 | 1909.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1899.10 | 1912.24 | 1909.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1904.60 | 1910.71 | 1908.92 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 1890.00 | 1904.32 | 1906.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1886.60 | 1900.77 | 1904.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1854.80 | 1851.30 | 1865.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 1854.80 | 1851.30 | 1865.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1862.20 | 1853.48 | 1864.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 1862.00 | 1853.48 | 1864.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1871.50 | 1857.09 | 1865.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1871.50 | 1857.09 | 1865.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1874.30 | 1860.53 | 1866.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1874.30 | 1860.53 | 1866.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1870.10 | 1862.44 | 1866.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1870.70 | 1862.44 | 1866.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1863.00 | 1863.14 | 1866.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:15:00 | 1872.80 | 1863.14 | 1866.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1868.80 | 1864.27 | 1866.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:30:00 | 1872.90 | 1864.27 | 1866.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 1863.70 | 1864.16 | 1866.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 1866.50 | 1864.16 | 1866.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 1865.70 | 1864.47 | 1866.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 1865.00 | 1864.47 | 1866.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 1859.70 | 1863.51 | 1865.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 1857.00 | 1863.51 | 1865.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 1852.80 | 1861.39 | 1863.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1858.80 | 1831.28 | 1831.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1858.80 | 1831.28 | 1831.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1858.80 | 1831.28 | 1831.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 1869.80 | 1838.98 | 1834.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1846.20 | 1846.32 | 1839.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1846.20 | 1846.32 | 1839.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1914.80 | 1918.23 | 1909.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 1911.10 | 1918.23 | 1909.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1951.50 | 1961.37 | 1953.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 1951.50 | 1961.37 | 1953.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1947.00 | 1958.50 | 1953.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 1945.80 | 1958.50 | 1953.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1957.20 | 1960.03 | 1956.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1957.20 | 1960.03 | 1956.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1949.90 | 1958.00 | 1955.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 1949.90 | 1958.00 | 1955.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1960.30 | 1958.46 | 1956.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:30:00 | 1965.50 | 1960.47 | 1957.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1971.40 | 1986.30 | 1987.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 10:15:00 | 1971.40 | 1986.30 | 1987.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 11:15:00 | 1970.00 | 1983.04 | 1985.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1986.00 | 1980.25 | 1982.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1986.00 | 1980.25 | 1982.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1986.00 | 1980.25 | 1982.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 1986.50 | 1980.25 | 1982.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1991.00 | 1982.40 | 1983.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1991.00 | 1982.40 | 1983.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1986.80 | 1983.28 | 1983.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 1983.90 | 1982.61 | 1983.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 1985.00 | 1979.98 | 1981.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 1993.00 | 1983.77 | 1983.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 1993.00 | 1983.77 | 1983.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1993.00 | 1983.77 | 1983.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 14:15:00 | 1996.50 | 1986.32 | 1984.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1980.30 | 1985.27 | 1984.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1980.30 | 1985.27 | 1984.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1980.30 | 1985.27 | 1984.16 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 1979.70 | 1983.31 | 1983.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1973.20 | 1979.16 | 1981.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 15:15:00 | 1973.30 | 1973.02 | 1976.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 1975.40 | 1973.02 | 1976.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1974.60 | 1973.34 | 1976.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 10:30:00 | 1966.30 | 1971.27 | 1975.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 1969.00 | 1975.12 | 1975.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1970.10 | 1974.08 | 1975.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 1871.59 | 1913.12 | 1933.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 1867.98 | 1903.32 | 1927.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 1870.55 | 1903.32 | 1927.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1824.90 | 1813.94 | 1825.66 | SL hit (close>ema200) qty=0.50 sl=1813.94 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1824.90 | 1813.94 | 1825.66 | SL hit (close>ema200) qty=0.50 sl=1813.94 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1824.90 | 1813.94 | 1825.66 | SL hit (close>ema200) qty=0.50 sl=1813.94 alert=retest2 |

### Cycle 9 — BUY (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 09:15:00 | 1828.00 | 1804.21 | 1802.31 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 1802.70 | 1808.74 | 1808.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 1799.70 | 1806.93 | 1808.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 1790.50 | 1790.49 | 1795.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1790.50 | 1790.49 | 1795.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1790.50 | 1790.49 | 1795.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 1783.10 | 1789.89 | 1792.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 12:15:00 | 1785.00 | 1789.89 | 1792.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 1785.70 | 1789.87 | 1792.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 1785.20 | 1789.46 | 1791.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1791.30 | 1789.14 | 1791.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 1785.90 | 1788.72 | 1790.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 1785.80 | 1787.16 | 1789.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:45:00 | 1784.60 | 1787.11 | 1789.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 14:15:00 | 1785.10 | 1787.11 | 1789.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 1848.20 | 1820.12 | 1805.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 1859.50 | 1860.79 | 1850.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:45:00 | 1859.60 | 1860.79 | 1850.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1848.00 | 1857.96 | 1853.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:45:00 | 1848.90 | 1857.96 | 1853.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1850.00 | 1856.37 | 1852.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1845.30 | 1856.37 | 1852.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1833.90 | 1849.85 | 1850.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 1828.70 | 1845.62 | 1848.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1805.60 | 1804.89 | 1814.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 1805.60 | 1804.89 | 1814.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1804.30 | 1803.54 | 1809.22 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1817.00 | 1809.43 | 1809.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1824.20 | 1816.09 | 1812.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1814.90 | 1821.35 | 1816.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1814.90 | 1821.35 | 1816.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1814.90 | 1821.35 | 1816.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1814.90 | 1821.35 | 1816.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1824.00 | 1821.88 | 1817.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 1819.50 | 1821.88 | 1817.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1815.00 | 1820.50 | 1817.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1825.20 | 1820.50 | 1817.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 1832.70 | 1823.22 | 1818.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 12:15:00 | 1827.50 | 1833.56 | 1833.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 12:15:00 | 1827.50 | 1833.56 | 1833.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 1827.50 | 1833.56 | 1833.98 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 1844.40 | 1834.15 | 1833.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 1850.00 | 1841.80 | 1838.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 1847.80 | 1849.32 | 1846.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 14:15:00 | 1847.80 | 1849.32 | 1846.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1847.80 | 1849.32 | 1846.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 1846.10 | 1849.32 | 1846.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1850.40 | 1849.54 | 1846.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1852.00 | 1849.54 | 1846.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1840.00 | 1847.98 | 1847.06 | SL hit (close<static) qty=1.00 sl=1846.70 alert=retest2 |

### Cycle 16 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 1837.20 | 1845.82 | 1846.16 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1854.40 | 1847.10 | 1846.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 1860.00 | 1851.68 | 1849.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 14:15:00 | 1866.00 | 1866.69 | 1861.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 15:00:00 | 1866.00 | 1866.69 | 1861.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1863.10 | 1866.02 | 1861.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 1865.40 | 1866.02 | 1861.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1861.90 | 1865.20 | 1861.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:45:00 | 1862.40 | 1865.20 | 1861.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1860.80 | 1864.32 | 1861.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 1860.80 | 1864.32 | 1861.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1858.90 | 1863.23 | 1861.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:30:00 | 1857.10 | 1863.23 | 1861.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 1854.80 | 1859.24 | 1859.78 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 1864.00 | 1860.11 | 1860.01 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 1850.00 | 1858.08 | 1859.10 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 1869.40 | 1860.76 | 1859.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 1875.50 | 1865.27 | 1862.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1886.00 | 1889.99 | 1879.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:45:00 | 1884.90 | 1889.99 | 1879.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1884.00 | 1888.80 | 1879.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 1891.20 | 1888.80 | 1879.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 1893.60 | 1889.76 | 1880.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 1890.80 | 1888.29 | 1881.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:15:00 | 1887.10 | 1886.94 | 1881.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1879.40 | 1885.02 | 1881.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 1877.50 | 1885.02 | 1881.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1879.50 | 1883.92 | 1881.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 1868.30 | 1883.92 | 1881.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1876.60 | 1882.46 | 1881.02 | SL hit (close<static) qty=1.00 sl=1878.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1876.60 | 1882.46 | 1881.02 | SL hit (close<static) qty=1.00 sl=1878.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1876.60 | 1882.46 | 1881.02 | SL hit (close<static) qty=1.00 sl=1878.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1876.60 | 1882.46 | 1881.02 | SL hit (close<static) qty=1.00 sl=1878.30 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1871.70 | 1880.30 | 1880.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1871.70 | 1880.30 | 1880.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1873.60 | 1878.96 | 1879.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 1864.50 | 1874.62 | 1877.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 1837.00 | 1834.67 | 1844.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 1837.00 | 1834.67 | 1844.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1825.50 | 1826.10 | 1836.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 1831.90 | 1826.10 | 1836.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1840.50 | 1828.50 | 1832.35 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1844.70 | 1835.00 | 1834.63 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 15:15:00 | 1829.90 | 1833.67 | 1834.16 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1844.30 | 1835.80 | 1835.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 1856.10 | 1843.83 | 1841.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1844.30 | 1850.43 | 1846.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 1844.30 | 1850.43 | 1846.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1844.30 | 1850.43 | 1846.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 1844.90 | 1850.43 | 1846.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1851.40 | 1850.62 | 1846.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 1857.00 | 1853.08 | 1848.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 1863.00 | 1867.65 | 1868.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1863.00 | 1867.65 | 1868.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 1855.30 | 1862.74 | 1864.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1862.90 | 1862.06 | 1864.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 1862.90 | 1862.06 | 1864.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1862.90 | 1862.06 | 1864.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 1856.90 | 1862.36 | 1863.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 1854.90 | 1860.55 | 1862.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 1853.90 | 1859.22 | 1861.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 1859.00 | 1844.49 | 1843.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 1859.00 | 1844.49 | 1843.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 1859.00 | 1844.49 | 1843.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 1859.00 | 1844.49 | 1843.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 1863.10 | 1851.44 | 1848.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 1850.80 | 1852.40 | 1849.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 10:15:00 | 1850.80 | 1852.40 | 1849.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1850.80 | 1852.40 | 1849.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 1848.00 | 1852.40 | 1849.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1860.50 | 1853.77 | 1850.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 1865.50 | 1855.23 | 1851.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 1862.70 | 1869.90 | 1865.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 1864.00 | 1865.87 | 1864.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1862.00 | 1864.70 | 1863.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1863.00 | 1864.36 | 1863.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 1877.90 | 1864.30 | 1863.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 1839.90 | 1864.79 | 1869.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1835.00 | 1831.61 | 1840.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:45:00 | 1836.70 | 1831.61 | 1840.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1846.00 | 1834.48 | 1840.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1846.00 | 1834.48 | 1840.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1843.20 | 1836.23 | 1840.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 1850.00 | 1836.23 | 1840.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1839.50 | 1836.88 | 1840.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1836.70 | 1838.69 | 1841.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 1850.00 | 1842.85 | 1842.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1850.00 | 1842.85 | 1842.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1860.50 | 1847.41 | 1845.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 1852.40 | 1852.61 | 1849.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 1852.40 | 1852.61 | 1849.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1852.40 | 1852.61 | 1849.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:45:00 | 1850.80 | 1852.61 | 1849.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1854.70 | 1853.86 | 1850.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 1852.10 | 1853.86 | 1850.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1849.50 | 1852.98 | 1850.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 1849.50 | 1852.98 | 1850.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1847.70 | 1851.93 | 1850.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1847.70 | 1851.93 | 1850.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1845.10 | 1849.29 | 1849.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1839.00 | 1846.93 | 1848.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1844.10 | 1843.75 | 1846.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1844.10 | 1843.75 | 1846.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1844.10 | 1843.75 | 1846.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:45:00 | 1840.60 | 1844.24 | 1845.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 1841.10 | 1843.22 | 1844.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 1840.40 | 1839.53 | 1842.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:00:00 | 1840.00 | 1838.68 | 1841.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1852.20 | 1841.30 | 1841.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 1850.00 | 1841.30 | 1841.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1858.00 | 1844.64 | 1843.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1858.00 | 1844.64 | 1843.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1858.00 | 1844.64 | 1843.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1858.00 | 1844.64 | 1843.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1858.00 | 1844.64 | 1843.01 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1839.10 | 1844.36 | 1844.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 1828.00 | 1838.52 | 1841.50 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 1916.20 | 1845.47 | 1841.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 1930.00 | 1862.38 | 1849.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 11:15:00 | 1864.90 | 1865.67 | 1854.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 1864.90 | 1865.67 | 1854.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1874.90 | 1867.70 | 1859.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 1880.10 | 1871.96 | 1865.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1856.10 | 1865.22 | 1866.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1856.10 | 1865.22 | 1866.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 1850.00 | 1862.17 | 1864.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 1854.60 | 1852.12 | 1855.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1854.60 | 1852.12 | 1855.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1854.60 | 1852.12 | 1855.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 1854.60 | 1852.12 | 1855.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1853.90 | 1851.48 | 1854.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1853.90 | 1851.48 | 1854.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1855.00 | 1852.18 | 1854.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1845.90 | 1852.18 | 1854.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1839.00 | 1849.54 | 1852.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 1837.00 | 1847.62 | 1851.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 1838.00 | 1844.67 | 1849.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:45:00 | 1838.30 | 1843.64 | 1848.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:00:00 | 1837.20 | 1842.35 | 1847.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1791.60 | 1789.70 | 1798.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1786.10 | 1789.10 | 1796.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1786.90 | 1781.94 | 1785.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 1784.80 | 1783.01 | 1785.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 1786.50 | 1783.01 | 1785.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1785.40 | 1783.49 | 1785.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 1783.80 | 1782.79 | 1785.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 10:15:00 | 1783.40 | 1779.84 | 1782.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 1787.40 | 1769.39 | 1765.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1761.90 | 1769.05 | 1766.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1761.90 | 1769.05 | 1766.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1761.90 | 1769.05 | 1766.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1760.70 | 1769.05 | 1766.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1763.10 | 1767.86 | 1765.93 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 1757.10 | 1764.20 | 1764.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 1755.00 | 1762.36 | 1763.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 1736.40 | 1728.76 | 1732.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 1736.40 | 1728.76 | 1732.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 1736.40 | 1728.76 | 1732.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 1734.40 | 1728.76 | 1732.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1744.40 | 1731.89 | 1734.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 1744.40 | 1731.89 | 1734.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1730.00 | 1731.44 | 1733.28 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1738.10 | 1733.76 | 1733.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 1742.00 | 1735.86 | 1734.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 1738.20 | 1738.34 | 1736.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1748.90 | 1738.34 | 1736.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1742.20 | 1739.91 | 1737.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:30:00 | 1741.90 | 1739.91 | 1737.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1750.00 | 1759.96 | 1753.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 1750.00 | 1759.96 | 1753.87 | SL hit (close<ema400) qty=1.00 sl=1753.87 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 1750.10 | 1759.96 | 1753.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1752.60 | 1758.49 | 1753.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 1758.00 | 1755.50 | 1754.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 15:15:00 | 1758.70 | 1755.50 | 1754.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 1742.00 | 1753.31 | 1753.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 1742.00 | 1753.31 | 1753.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 1742.00 | 1753.31 | 1753.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1734.50 | 1747.32 | 1750.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 1704.40 | 1703.10 | 1715.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 1704.40 | 1703.10 | 1715.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1706.70 | 1705.47 | 1713.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1702.30 | 1705.30 | 1712.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 1719.00 | 1708.04 | 1713.33 | SL hit (close>static) qty=1.00 sl=1717.00 alert=retest2 |

### Cycle 39 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1725.60 | 1714.94 | 1714.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 1728.60 | 1717.67 | 1715.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 10:15:00 | 1737.30 | 1738.12 | 1730.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:30:00 | 1739.00 | 1738.12 | 1730.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1730.00 | 1736.07 | 1732.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1736.90 | 1736.07 | 1732.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1738.30 | 1736.51 | 1732.89 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 1720.90 | 1730.59 | 1730.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1706.80 | 1725.83 | 1728.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1706.80 | 1704.52 | 1714.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 1706.80 | 1704.52 | 1714.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 1714.20 | 1706.16 | 1713.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 1714.20 | 1706.16 | 1713.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1731.00 | 1711.13 | 1715.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1727.10 | 1715.00 | 1716.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1729.00 | 1719.40 | 1718.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 1733.60 | 1722.24 | 1719.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1719.60 | 1724.83 | 1722.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1719.60 | 1724.83 | 1722.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1719.60 | 1724.83 | 1722.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:15:00 | 1716.00 | 1724.83 | 1722.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1719.00 | 1723.66 | 1721.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:45:00 | 1716.90 | 1723.66 | 1721.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1698.20 | 1716.79 | 1718.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1684.90 | 1710.41 | 1715.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1698.00 | 1696.94 | 1707.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 1686.00 | 1687.38 | 1695.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1686.00 | 1687.38 | 1695.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 14:15:00 | 1680.00 | 1689.66 | 1694.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 15:00:00 | 1684.50 | 1688.63 | 1693.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:00:00 | 1682.00 | 1680.29 | 1686.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:45:00 | 1679.70 | 1680.11 | 1685.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1635.70 | 1668.09 | 1676.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:00:00 | 1617.90 | 1643.94 | 1658.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:15:00 | 1600.27 | 1630.80 | 1648.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:30:00 | 1616.00 | 1630.80 | 1648.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 1596.00 | 1626.96 | 1645.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 1597.90 | 1626.96 | 1645.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 1595.71 | 1626.96 | 1645.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 1630.80 | 1626.29 | 1641.72 | SL hit (close>ema200) qty=0.50 sl=1626.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 1630.80 | 1626.29 | 1641.72 | SL hit (close>ema200) qty=0.50 sl=1626.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 1630.80 | 1626.29 | 1641.72 | SL hit (close>ema200) qty=0.50 sl=1626.29 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 1630.80 | 1626.29 | 1641.72 | SL hit (close>ema200) qty=0.50 sl=1626.29 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 1624.30 | 1628.43 | 1641.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 1681.10 | 1650.27 | 1648.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 1681.10 | 1650.27 | 1648.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 1681.10 | 1650.27 | 1648.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1681.10 | 1650.27 | 1648.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 1687.00 | 1671.64 | 1662.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1677.00 | 1681.35 | 1671.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1677.00 | 1681.35 | 1671.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1670.80 | 1678.08 | 1671.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 1670.80 | 1678.08 | 1671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 1673.10 | 1677.09 | 1671.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 1670.00 | 1677.09 | 1671.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 1671.60 | 1675.99 | 1671.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 1671.60 | 1675.99 | 1671.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1677.50 | 1676.29 | 1672.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 1676.00 | 1676.29 | 1672.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 1674.90 | 1676.01 | 1672.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 1668.50 | 1676.01 | 1672.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1656.10 | 1672.03 | 1670.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1656.10 | 1672.03 | 1670.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 1660.40 | 1669.70 | 1670.03 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1676.00 | 1668.66 | 1668.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1690.10 | 1672.95 | 1670.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1694.10 | 1698.07 | 1689.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 1694.10 | 1698.07 | 1689.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1688.50 | 1695.08 | 1690.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1688.60 | 1695.08 | 1690.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1690.70 | 1694.20 | 1690.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:15:00 | 1687.60 | 1694.20 | 1690.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1688.00 | 1692.96 | 1690.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 1688.00 | 1692.96 | 1690.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1690.00 | 1692.37 | 1690.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 1695.00 | 1692.37 | 1690.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1681.30 | 1691.01 | 1690.63 | SL hit (close<static) qty=1.00 sl=1684.90 alert=retest2 |

### Cycle 46 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 1679.80 | 1688.77 | 1689.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 1676.00 | 1684.65 | 1687.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 1644.20 | 1638.51 | 1648.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 1644.20 | 1638.51 | 1648.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1638.50 | 1638.51 | 1647.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1640.00 | 1638.51 | 1647.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1644.40 | 1639.11 | 1646.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1633.00 | 1637.23 | 1643.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 1634.10 | 1637.74 | 1640.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 1624.60 | 1621.70 | 1621.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 1624.60 | 1621.70 | 1621.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1624.60 | 1621.70 | 1621.32 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 1617.30 | 1621.29 | 1621.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 1615.60 | 1620.15 | 1620.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1522.50 | 1519.08 | 1536.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 1526.30 | 1519.08 | 1536.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1531.00 | 1522.41 | 1534.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 1531.00 | 1522.41 | 1534.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1477.70 | 1472.74 | 1480.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 1474.30 | 1473.47 | 1479.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:30:00 | 1474.90 | 1473.98 | 1479.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 13:00:00 | 1474.30 | 1474.04 | 1479.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1400.58 | 1419.90 | 1440.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1401.15 | 1419.90 | 1440.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1400.58 | 1419.90 | 1440.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1378.60 | 1376.98 | 1390.92 | SL hit (close>ema200) qty=0.50 sl=1376.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1378.60 | 1376.98 | 1390.92 | SL hit (close>ema200) qty=0.50 sl=1376.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1378.60 | 1376.98 | 1390.92 | SL hit (close>ema200) qty=0.50 sl=1376.98 alert=retest2 |

### Cycle 49 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 1404.60 | 1394.39 | 1394.08 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1373.20 | 1393.41 | 1394.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 1366.60 | 1388.05 | 1391.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1370.80 | 1367.32 | 1377.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:00:00 | 1370.80 | 1367.32 | 1377.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1369.80 | 1368.12 | 1376.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1345.00 | 1375.21 | 1377.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1379.70 | 1345.08 | 1348.65 | SL hit (close>static) qty=1.00 sl=1376.40 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1377.50 | 1351.57 | 1351.28 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1317.60 | 1348.54 | 1352.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1315.90 | 1332.31 | 1342.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1310.50 | 1284.62 | 1305.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1310.50 | 1284.62 | 1305.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1310.50 | 1284.62 | 1305.05 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 1329.00 | 1314.78 | 1313.58 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1304.10 | 1311.50 | 1312.22 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1319.80 | 1313.71 | 1313.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 1328.00 | 1316.56 | 1314.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1339.50 | 1343.06 | 1332.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 1339.50 | 1343.06 | 1332.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1337.50 | 1341.43 | 1333.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 1335.00 | 1341.43 | 1333.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1389.60 | 1411.58 | 1401.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:30:00 | 1410.40 | 1406.73 | 1402.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1427.10 | 1406.73 | 1402.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1426.60 | 1433.93 | 1434.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1426.60 | 1433.93 | 1434.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1426.60 | 1433.93 | 1434.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1410.80 | 1424.91 | 1429.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1428.70 | 1415.06 | 1420.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1428.70 | 1415.06 | 1420.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1428.70 | 1415.06 | 1420.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1432.80 | 1415.06 | 1420.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1433.00 | 1418.65 | 1421.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1433.00 | 1418.65 | 1421.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1454.90 | 1427.35 | 1425.03 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1410.20 | 1434.70 | 1435.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 1409.00 | 1429.56 | 1433.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 1425.60 | 1421.79 | 1426.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 10:15:00 | 1425.60 | 1421.79 | 1426.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1425.60 | 1421.79 | 1426.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 1425.60 | 1421.79 | 1426.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1425.00 | 1422.43 | 1426.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 1432.90 | 1422.43 | 1426.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 1411.00 | 1420.14 | 1424.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 1406.40 | 1420.14 | 1424.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:15:00 | 1396.50 | 1418.31 | 1423.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1422.40 | 1408.96 | 1407.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1422.40 | 1408.96 | 1407.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1422.40 | 1408.96 | 1407.37 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 1400.10 | 1411.33 | 1411.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 1395.00 | 1408.07 | 1410.17 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 13:00:00 | 1941.70 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-05-21 14:15:00 | 1942.00 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-21 15:00:00 | 1942.20 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-05-22 09:45:00 | 1941.90 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-22 14:00:00 | 1937.50 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-05-22 14:30:00 | 1939.50 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-05-27 10:30:00 | 1937.00 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-05-27 11:15:00 | 1937.70 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-06-03 12:45:00 | 1872.10 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-03 13:30:00 | 1875.00 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-03 14:15:00 | 1875.00 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-04 09:15:00 | 1872.30 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-17 15:15:00 | 1857.00 | 2025-06-24 10:15:00 | 1858.80 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-06-18 11:15:00 | 1852.80 | 2025-06-24 10:15:00 | 1858.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-08 12:30:00 | 1965.50 | 2025-07-14 10:15:00 | 1971.40 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-07-15 12:30:00 | 1983.90 | 2025-07-16 13:15:00 | 1993.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-16 12:15:00 | 1985.00 | 2025-07-16 13:15:00 | 1993.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-21 10:30:00 | 1966.30 | 2025-07-25 09:15:00 | 1871.59 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1969.00 | 2025-07-25 10:15:00 | 1867.98 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1970.10 | 2025-07-25 10:15:00 | 1870.55 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-07-21 10:30:00 | 1966.30 | 2025-07-31 12:15:00 | 1824.90 | STOP_HIT | 0.50 | 7.19% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1969.00 | 2025-07-31 12:15:00 | 1824.90 | STOP_HIT | 0.50 | 7.32% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1970.10 | 2025-07-31 12:15:00 | 1824.90 | STOP_HIT | 0.50 | 7.37% |
| SELL | retest2 | 2025-08-13 11:45:00 | 1783.10 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-08-13 12:15:00 | 1785.00 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-08-13 13:15:00 | 1785.70 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-08-13 15:15:00 | 1785.20 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-08-14 11:15:00 | 1785.90 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-08-14 12:45:00 | 1785.80 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-08-14 13:45:00 | 1784.60 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-08-14 14:15:00 | 1785.10 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1825.20 | 2025-09-05 12:15:00 | 1827.50 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-09-03 09:45:00 | 1832.70 | 2025-09-05 12:15:00 | 1827.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1852.00 | 2025-09-11 12:15:00 | 1840.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-09-23 09:15:00 | 1891.20 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-09-23 10:00:00 | 1893.60 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-23 10:30:00 | 1890.80 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-23 13:15:00 | 1887.10 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-08 14:30:00 | 1857.00 | 2025-10-14 11:15:00 | 1863.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-10-16 13:15:00 | 1856.90 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-10-17 10:15:00 | 1854.90 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-10-17 11:00:00 | 1853.90 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-28 15:15:00 | 1865.50 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-30 12:15:00 | 1862.70 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-30 15:00:00 | 1864.00 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1862.00 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-31 12:15:00 | 1877.90 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1836.70 | 2025-11-10 12:15:00 | 1850.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-18 09:45:00 | 1840.60 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-18 11:30:00 | 1841.10 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-19 09:30:00 | 1840.40 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-11-19 13:00:00 | 1840.00 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-11-26 14:45:00 | 1880.10 | 2025-11-28 11:15:00 | 1856.10 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-03 10:30:00 | 1837.00 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.34% |
| SELL | retest2 | 2025-12-03 12:45:00 | 1838.00 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.39% |
| SELL | retest2 | 2025-12-03 13:45:00 | 1838.30 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.41% |
| SELL | retest2 | 2025-12-03 15:00:00 | 1837.20 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.35% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1786.10 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-12-12 10:15:00 | 1786.90 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-12-12 10:45:00 | 1784.80 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2025-12-12 11:15:00 | 1786.50 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-12-12 12:30:00 | 1783.80 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-12-15 10:15:00 | 1783.40 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest1 | 2026-01-02 09:15:00 | 1748.90 | 2026-01-06 10:15:00 | 1750.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-01-07 14:45:00 | 1758.00 | 2026-01-08 09:15:00 | 1742.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-07 15:15:00 | 1758.70 | 2026-01-08 09:15:00 | 1742.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1702.30 | 2026-01-13 11:15:00 | 1719.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-28 14:15:00 | 1680.00 | 2026-02-02 10:15:00 | 1600.27 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-28 15:00:00 | 1684.50 | 2026-02-02 11:15:00 | 1596.00 | PARTIAL | 0.50 | 5.25% |
| SELL | retest2 | 2026-01-29 14:00:00 | 1682.00 | 2026-02-02 11:15:00 | 1597.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1679.70 | 2026-02-02 11:15:00 | 1595.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 14:15:00 | 1680.00 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2026-01-28 15:00:00 | 1684.50 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-01-29 14:00:00 | 1682.00 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1679.70 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2026-02-01 15:00:00 | 1617.90 | 2026-02-03 11:15:00 | 1681.10 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2026-02-02 10:30:00 | 1616.00 | 2026-02-03 11:15:00 | 1681.10 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2026-02-02 14:30:00 | 1624.30 | 2026-02-03 11:15:00 | 1681.10 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2026-02-11 13:15:00 | 1695.00 | 2026-02-12 10:15:00 | 1681.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1633.00 | 2026-02-25 11:15:00 | 1624.60 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2026-02-19 11:45:00 | 1634.10 | 2026-02-25 11:15:00 | 1624.60 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1474.30 | 2026-03-13 12:15:00 | 1400.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:30:00 | 1474.90 | 2026-03-13 12:15:00 | 1401.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 1474.30 | 2026-03-13 12:15:00 | 1400.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1474.30 | 2026-03-17 13:15:00 | 1378.60 | STOP_HIT | 0.50 | 6.49% |
| SELL | retest2 | 2026-03-11 11:30:00 | 1474.90 | 2026-03-17 13:15:00 | 1378.60 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2026-03-11 13:00:00 | 1474.30 | 2026-03-17 13:15:00 | 1378.60 | STOP_HIT | 0.50 | 6.49% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1345.00 | 2026-03-25 09:15:00 | 1379.70 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-04-13 14:30:00 | 1410.40 | 2026-04-23 11:15:00 | 1426.60 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1427.10 | 2026-04-23 11:15:00 | 1426.60 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2026-05-04 13:15:00 | 1406.40 | 2026-05-07 09:15:00 | 1422.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-05-04 14:15:00 | 1396.50 | 2026-05-07 09:15:00 | 1422.40 | STOP_HIT | 1.00 | -1.85% |
