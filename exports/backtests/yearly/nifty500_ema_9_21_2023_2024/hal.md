# Hindustan Aeronautics Ltd. (HAL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4790.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 208 |
| ALERT1 | 148 |
| ALERT2 | 148 |
| ALERT2_SKIP | 67 |
| ALERT3 | 399 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 181 |
| PARTIAL | 17 |
| TARGET_HIT | 10 |
| STOP_HIT | 180 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 207 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 153
- **Target hits / Stop hits / Partials:** 10 / 180 / 17
- **Avg / median % per leg:** -0.19% / -1.07%
- **Sum % (uncompounded):** -38.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 106 | 21 | 19.8% | 7 | 99 | 0 | -0.98% | -104.2% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 8 | 0 | -0.18% | -1.4% |
| BUY @ 3rd Alert (retest2) | 98 | 18 | 18.4% | 7 | 91 | 0 | -1.05% | -102.7% |
| SELL (all) | 101 | 33 | 32.7% | 3 | 81 | 17 | 0.65% | 65.3% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.23% | 8.5% |
| SELL @ 3rd Alert (retest2) | 99 | 31 | 31.3% | 3 | 80 | 16 | 0.57% | 56.8% |
| retest1 (combined) | 10 | 5 | 50.0% | 0 | 9 | 1 | 0.70% | 7.0% |
| retest2 (combined) | 197 | 49 | 24.9% | 10 | 171 | 16 | -0.23% | -45.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 12:15:00 | 1529.93 | 1540.86 | 1541.99 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 14:15:00 | 1550.47 | 1542.80 | 1541.93 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 12:15:00 | 1529.00 | 1540.12 | 1541.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 13:15:00 | 1528.00 | 1537.70 | 1539.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 1507.30 | 1506.03 | 1515.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-25 15:00:00 | 1507.30 | 1506.03 | 1515.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 1513.28 | 1508.51 | 1514.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 10:45:00 | 1509.00 | 1509.23 | 1514.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 11:15:00 | 1510.00 | 1509.23 | 1514.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 11:45:00 | 1509.88 | 1508.14 | 1513.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 13:45:00 | 1508.35 | 1508.83 | 1512.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 15:15:00 | 1513.00 | 1509.84 | 1512.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 09:15:00 | 1509.40 | 1509.84 | 1512.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 1509.97 | 1509.87 | 1512.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 13:30:00 | 1503.50 | 1509.23 | 1511.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 09:15:00 | 1530.97 | 1513.34 | 1512.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 09:15:00 | 1530.97 | 1513.34 | 1512.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 10:15:00 | 1552.88 | 1521.25 | 1516.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 10:15:00 | 1533.50 | 1537.85 | 1529.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 11:00:00 | 1533.50 | 1537.85 | 1529.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 1524.22 | 1535.12 | 1528.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 12:00:00 | 1524.22 | 1535.12 | 1528.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 1522.18 | 1532.53 | 1528.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 15:00:00 | 1560.35 | 1536.33 | 1530.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-06 15:15:00 | 1716.38 | 1680.32 | 1643.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 09:15:00 | 1893.50 | 1909.58 | 1911.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 1883.75 | 1902.47 | 1907.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 12:15:00 | 1835.00 | 1832.50 | 1854.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 13:00:00 | 1835.00 | 1832.50 | 1854.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 1849.88 | 1836.54 | 1852.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:45:00 | 1846.82 | 1836.54 | 1852.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 1855.00 | 1840.23 | 1852.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 1869.12 | 1840.23 | 1852.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 1861.05 | 1844.39 | 1853.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 11:15:00 | 1854.95 | 1846.95 | 1853.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 09:45:00 | 1848.60 | 1842.86 | 1848.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 15:15:00 | 1864.97 | 1850.02 | 1849.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 15:15:00 | 1864.97 | 1850.02 | 1849.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 13:15:00 | 1886.50 | 1862.78 | 1856.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 10:15:00 | 1882.03 | 1884.73 | 1876.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 11:00:00 | 1882.03 | 1884.73 | 1876.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 11:15:00 | 1877.18 | 1883.22 | 1876.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 12:00:00 | 1877.18 | 1883.22 | 1876.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 12:15:00 | 1867.90 | 1880.16 | 1876.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 13:00:00 | 1867.90 | 1880.16 | 1876.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 1870.90 | 1878.30 | 1875.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:15:00 | 1861.32 | 1878.30 | 1875.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 15:15:00 | 1862.50 | 1872.62 | 1873.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 11:15:00 | 1853.97 | 1865.44 | 1869.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 14:15:00 | 1867.50 | 1862.72 | 1866.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 14:15:00 | 1867.50 | 1862.72 | 1866.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 14:15:00 | 1867.50 | 1862.72 | 1866.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 15:00:00 | 1867.50 | 1862.72 | 1866.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 1871.00 | 1864.37 | 1867.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:15:00 | 1872.97 | 1864.37 | 1867.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 1870.18 | 1865.53 | 1867.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 11:30:00 | 1862.97 | 1864.29 | 1866.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 09:15:00 | 1881.30 | 1867.72 | 1867.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 09:15:00 | 1881.30 | 1867.72 | 1867.13 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 1855.50 | 1865.64 | 1866.40 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 13:15:00 | 1875.03 | 1867.52 | 1867.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 14:15:00 | 1880.35 | 1870.08 | 1868.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 11:15:00 | 1877.50 | 1877.65 | 1873.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-10 12:00:00 | 1877.50 | 1877.65 | 1873.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 12:15:00 | 1876.45 | 1877.41 | 1873.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 12:30:00 | 1876.00 | 1877.41 | 1873.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 1887.40 | 1879.37 | 1874.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 14:45:00 | 1876.57 | 1879.37 | 1874.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 1935.50 | 1939.19 | 1915.93 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 10:15:00 | 1898.00 | 1918.53 | 1920.09 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 10:15:00 | 1931.00 | 1916.34 | 1915.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 09:15:00 | 1938.03 | 1923.22 | 1919.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 09:15:00 | 1920.65 | 1926.37 | 1923.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 1920.65 | 1926.37 | 1923.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 1920.65 | 1926.37 | 1923.41 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 13:15:00 | 1916.50 | 1921.12 | 1921.55 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 09:15:00 | 1932.00 | 1922.18 | 1921.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 09:15:00 | 1953.72 | 1934.28 | 1929.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 11:15:00 | 1937.90 | 1938.26 | 1932.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 11:15:00 | 1937.90 | 1938.26 | 1932.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 1937.90 | 1938.26 | 1932.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:45:00 | 1934.80 | 1938.26 | 1932.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 1939.50 | 1941.19 | 1936.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:00:00 | 1939.50 | 1941.19 | 1936.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 1931.03 | 1939.16 | 1935.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:30:00 | 1932.05 | 1939.16 | 1935.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 1922.97 | 1935.92 | 1934.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 12:00:00 | 1922.97 | 1935.92 | 1934.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2023-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 13:15:00 | 1917.62 | 1930.61 | 1932.31 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 09:15:00 | 1955.10 | 1932.33 | 1930.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 12:15:00 | 1962.97 | 1943.41 | 1936.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 09:15:00 | 1957.50 | 1969.48 | 1959.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 09:15:00 | 1957.50 | 1969.48 | 1959.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 1957.50 | 1969.48 | 1959.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:00:00 | 1957.50 | 1969.48 | 1959.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 1949.50 | 1965.48 | 1958.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:30:00 | 1949.75 | 1965.48 | 1958.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 13:15:00 | 1933.57 | 1951.70 | 1953.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 09:15:00 | 1918.25 | 1939.98 | 1947.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 15:15:00 | 1886.50 | 1886.16 | 1901.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-04 09:15:00 | 1897.38 | 1886.16 | 1901.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 1894.80 | 1882.91 | 1887.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 11:15:00 | 1878.00 | 1882.70 | 1886.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 12:00:00 | 1879.00 | 1881.96 | 1885.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 10:15:00 | 1901.95 | 1887.81 | 1887.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 10:15:00 | 1901.95 | 1887.81 | 1887.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 15:15:00 | 1915.50 | 1900.23 | 1893.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 13:15:00 | 1899.62 | 1904.80 | 1899.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 13:15:00 | 1899.62 | 1904.80 | 1899.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 1899.62 | 1904.80 | 1899.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 13:45:00 | 1900.03 | 1904.80 | 1899.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 1896.80 | 1903.20 | 1899.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 15:00:00 | 1896.80 | 1903.20 | 1899.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 1900.00 | 1902.56 | 1899.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 09:15:00 | 1900.43 | 1902.56 | 1899.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 14:15:00 | 1885.55 | 1906.02 | 1903.67 | SL hit (close<static) qty=1.00 sl=1892.05 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 15:15:00 | 1885.43 | 1901.90 | 1902.01 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 09:15:00 | 1907.45 | 1903.01 | 1902.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 10:15:00 | 1933.32 | 1909.07 | 1905.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-16 09:15:00 | 1921.57 | 1925.99 | 1917.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-16 10:00:00 | 1921.57 | 1925.99 | 1917.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 1924.35 | 1925.71 | 1918.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 11:30:00 | 1917.43 | 1925.71 | 1918.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 1936.95 | 1927.96 | 1920.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 15:00:00 | 1942.97 | 1931.85 | 1923.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 14:30:00 | 1939.50 | 1940.88 | 1933.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 09:15:00 | 1918.50 | 1935.83 | 1932.64 | SL hit (close<static) qty=1.00 sl=1919.03 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 11:15:00 | 1907.28 | 1928.19 | 1929.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 1896.95 | 1921.94 | 1926.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 14:15:00 | 1907.47 | 1906.34 | 1913.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 15:00:00 | 1907.47 | 1906.34 | 1913.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 1913.50 | 1907.77 | 1913.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 1934.53 | 1907.77 | 1913.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 1938.07 | 1913.83 | 1915.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 10:00:00 | 1938.07 | 1913.83 | 1915.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 10:15:00 | 1937.95 | 1918.65 | 1917.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 09:15:00 | 1959.38 | 1938.86 | 1929.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 10:15:00 | 1992.70 | 1996.35 | 1970.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 11:00:00 | 1992.70 | 1996.35 | 1970.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 1983.50 | 1995.08 | 1980.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 1991.20 | 1995.08 | 1980.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 1984.97 | 1993.06 | 1980.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:15:00 | 1966.65 | 1993.06 | 1980.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 1956.30 | 1985.71 | 1978.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 1961.05 | 1985.71 | 1978.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 11:15:00 | 1957.05 | 1979.98 | 1976.49 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 1961.00 | 1972.84 | 1973.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 14:15:00 | 1953.65 | 1969.00 | 1971.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 1969.00 | 1965.88 | 1969.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 1969.00 | 1965.88 | 1969.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 1969.00 | 1965.88 | 1969.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 09:30:00 | 1949.78 | 1960.19 | 1965.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 14:45:00 | 1948.65 | 1951.71 | 1958.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 12:15:00 | 1965.47 | 1961.81 | 1961.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 12:15:00 | 1965.47 | 1961.81 | 1961.44 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 14:15:00 | 1951.97 | 1959.88 | 1960.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 1950.00 | 1955.75 | 1958.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 15:15:00 | 1956.00 | 1953.76 | 1956.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 15:15:00 | 1956.00 | 1953.76 | 1956.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 1956.00 | 1953.76 | 1956.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:15:00 | 1967.50 | 1953.76 | 1956.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 1982.00 | 1959.40 | 1958.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 10:15:00 | 1996.80 | 1975.68 | 1968.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 11:15:00 | 1981.00 | 1983.21 | 1977.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 12:00:00 | 1981.00 | 1983.21 | 1977.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 1976.30 | 1981.82 | 1977.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 12:45:00 | 1977.00 | 1981.82 | 1977.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 1978.75 | 1981.21 | 1977.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 09:15:00 | 1986.72 | 1980.07 | 1977.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 10:15:00 | 1995.00 | 2024.90 | 2027.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 1995.00 | 2024.90 | 2027.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 1971.32 | 1997.49 | 2011.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 13:15:00 | 1982.03 | 1979.89 | 1994.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 14:00:00 | 1982.03 | 1979.89 | 1994.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 1982.38 | 1982.00 | 1992.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 11:15:00 | 1976.60 | 1981.80 | 1991.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:45:00 | 1977.10 | 1981.35 | 1988.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 09:30:00 | 1977.00 | 1981.53 | 1986.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 15:00:00 | 1973.03 | 1979.97 | 1984.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 2005.53 | 1984.29 | 1985.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-18 09:15:00 | 2005.53 | 1984.29 | 1985.41 | SL hit (close>static) qty=1.00 sl=1994.75 alert=retest2 |

### Cycle 28 — BUY (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 10:15:00 | 2003.65 | 1988.16 | 1987.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-20 09:15:00 | 2025.78 | 1996.85 | 1991.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 14:15:00 | 1995.97 | 2000.17 | 1995.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 14:15:00 | 1995.97 | 2000.17 | 1995.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 1995.97 | 2000.17 | 1995.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 15:00:00 | 1995.97 | 2000.17 | 1995.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 1999.00 | 1999.93 | 1996.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-21 09:15:00 | 4015.00 | 1999.93 | 1996.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-22 09:15:00 | 1949.62 | 3213.73 | 2853.00 | SL hit (close<static) qty=1.00 sl=1993.50 alert=retest2 |

### Cycle 29 — SELL (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 12:15:00 | 1939.47 | 2591.23 | 2625.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 13:15:00 | 1934.30 | 2459.84 | 2562.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 10:15:00 | 1919.00 | 1909.42 | 1946.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 11:00:00 | 1919.00 | 1909.42 | 1946.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 1923.25 | 1919.51 | 1940.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 14:45:00 | 1939.00 | 1919.51 | 1940.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1938.80 | 1925.93 | 1939.53 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 15:15:00 | 1958.85 | 1945.86 | 1944.90 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 1923.25 | 1941.34 | 1942.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 1904.15 | 1931.50 | 1938.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 1946.00 | 1924.94 | 1930.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 1946.00 | 1924.94 | 1930.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 1946.00 | 1924.94 | 1930.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 09:30:00 | 1968.05 | 1924.94 | 1930.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 13:15:00 | 1940.55 | 1934.69 | 1934.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 09:15:00 | 1958.95 | 1941.94 | 1937.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 1936.85 | 1949.08 | 1944.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 1936.85 | 1949.08 | 1944.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 1936.85 | 1949.08 | 1944.89 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 11:15:00 | 1928.90 | 1941.92 | 1942.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 1919.70 | 1937.48 | 1940.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1950.40 | 1933.34 | 1936.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1950.40 | 1933.34 | 1936.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1950.40 | 1933.34 | 1936.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:45:00 | 1954.25 | 1933.34 | 1936.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 1952.00 | 1937.07 | 1937.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:45:00 | 1956.25 | 1937.07 | 1937.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 1962.70 | 1942.20 | 1940.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 12:15:00 | 1964.95 | 1946.75 | 1942.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 14:15:00 | 1960.95 | 1962.81 | 1955.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 15:00:00 | 1960.95 | 1962.81 | 1955.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1972.90 | 1967.03 | 1962.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 11:15:00 | 1978.00 | 1968.80 | 1963.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 12:15:00 | 1978.30 | 1970.20 | 1964.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 13:00:00 | 1977.55 | 1971.67 | 1965.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 14:15:00 | 1964.10 | 1966.43 | 1966.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 14:15:00 | 1964.10 | 1966.43 | 1966.59 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 12:15:00 | 1967.90 | 1966.68 | 1966.55 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 1960.75 | 1965.49 | 1966.02 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 09:15:00 | 1981.75 | 1968.24 | 1967.08 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 13:15:00 | 1962.40 | 1966.26 | 1966.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 1954.60 | 1963.16 | 1964.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 13:15:00 | 1804.90 | 1802.59 | 1833.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 14:15:00 | 1815.75 | 1805.22 | 1831.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 1815.75 | 1805.22 | 1831.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:45:00 | 1828.80 | 1805.22 | 1831.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1835.50 | 1813.96 | 1831.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:45:00 | 1840.20 | 1813.96 | 1831.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 1840.00 | 1819.17 | 1832.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:30:00 | 1848.00 | 1819.17 | 1832.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 1850.05 | 1825.35 | 1833.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:00:00 | 1850.05 | 1825.35 | 1833.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 1853.90 | 1838.27 | 1838.12 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 12:15:00 | 1835.20 | 1838.46 | 1838.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 14:15:00 | 1833.00 | 1836.92 | 1837.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-01 11:15:00 | 1827.85 | 1827.29 | 1830.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-01 12:00:00 | 1827.85 | 1827.29 | 1830.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 1823.50 | 1826.53 | 1830.10 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 11:15:00 | 1837.95 | 1830.14 | 1830.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 13:15:00 | 1853.65 | 1837.29 | 1833.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 15:15:00 | 2031.00 | 2031.77 | 2010.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 09:15:00 | 2073.90 | 2031.77 | 2010.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 2032.45 | 2057.15 | 2033.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-10 13:15:00 | 2032.45 | 2057.15 | 2033.62 | SL hit (close<ema400) qty=1.00 sl=2033.62 alert=retest1 |

### Cycle 43 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 2103.00 | 2132.80 | 2133.66 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 11:15:00 | 2151.05 | 2130.91 | 2130.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 10:15:00 | 2172.10 | 2148.89 | 2140.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 09:15:00 | 2668.70 | 2670.49 | 2603.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 09:15:00 | 2746.50 | 2682.51 | 2639.32 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 10:00:00 | 2742.75 | 2694.56 | 2648.73 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:30:00 | 2773.70 | 2734.77 | 2693.38 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 2740.00 | 2767.48 | 2734.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:00:00 | 2740.00 | 2767.48 | 2734.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 2749.10 | 2763.81 | 2736.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-12 13:15:00 | 2735.00 | 2752.90 | 2737.62 | SL hit (close<ema400) qty=1.00 sl=2737.62 alert=retest1 |

### Cycle 45 — SELL (started 2023-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 11:15:00 | 2745.35 | 2757.62 | 2757.91 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 14:15:00 | 2764.00 | 2758.35 | 2758.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 09:15:00 | 2775.55 | 2762.37 | 2760.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 2782.50 | 2797.65 | 2784.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 2782.50 | 2797.65 | 2784.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 2782.50 | 2797.65 | 2784.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 2782.50 | 2797.65 | 2784.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 2787.65 | 2795.65 | 2784.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 11:45:00 | 2801.25 | 2797.65 | 2786.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 13:30:00 | 2802.80 | 2797.88 | 2788.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:30:00 | 2817.20 | 2796.69 | 2790.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 11:45:00 | 2798.00 | 2798.40 | 2792.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 2780.95 | 2794.91 | 2791.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 2780.95 | 2794.91 | 2791.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 2702.45 | 2776.42 | 2782.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 2702.45 | 2776.42 | 2782.95 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 11:15:00 | 2768.25 | 2732.17 | 2729.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 12:15:00 | 2796.00 | 2744.93 | 2735.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 2796.00 | 2797.60 | 2773.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:30:00 | 2795.75 | 2797.60 | 2773.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 2798.35 | 2803.58 | 2793.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:00:00 | 2798.35 | 2803.58 | 2793.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 12:15:00 | 2798.10 | 2803.11 | 2796.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 12:45:00 | 2797.10 | 2803.11 | 2796.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 13:15:00 | 2788.10 | 2800.11 | 2795.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 14:00:00 | 2788.10 | 2800.11 | 2795.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 2803.45 | 2800.78 | 2796.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 2844.50 | 2801.62 | 2796.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:00:00 | 2814.85 | 2822.80 | 2813.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 10:15:00 | 2773.95 | 2813.03 | 2810.19 | SL hit (close<static) qty=1.00 sl=2784.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 2775.65 | 2805.55 | 2807.05 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 11:15:00 | 2829.95 | 2806.59 | 2805.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 12:15:00 | 2846.05 | 2814.48 | 2809.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 2990.00 | 2996.77 | 2963.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 14:30:00 | 2992.15 | 2996.77 | 2963.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 3007.00 | 3015.11 | 2995.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 09:15:00 | 3032.00 | 3011.10 | 3001.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 14:15:00 | 3023.25 | 3021.01 | 3011.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 15:15:00 | 3022.00 | 3020.53 | 3011.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 12:15:00 | 2999.50 | 3008.66 | 3008.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 12:15:00 | 2999.50 | 3008.66 | 3008.80 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 12:15:00 | 3018.00 | 3008.85 | 3008.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 13:15:00 | 3019.75 | 3011.03 | 3009.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 11:15:00 | 3018.50 | 3024.33 | 3017.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 11:15:00 | 3018.50 | 3024.33 | 3017.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 3018.50 | 3024.33 | 3017.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 12:00:00 | 3018.50 | 3024.33 | 3017.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 2969.60 | 3013.38 | 3013.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 2969.60 | 3013.38 | 3013.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 13:15:00 | 2988.00 | 3008.30 | 3010.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 2917.15 | 2962.75 | 2982.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 2967.00 | 2957.85 | 2976.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 12:00:00 | 2967.00 | 2957.85 | 2976.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 14:15:00 | 2964.20 | 2956.44 | 2970.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 14:45:00 | 2958.30 | 2956.44 | 2970.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 2974.05 | 2959.96 | 2971.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:15:00 | 2995.95 | 2959.96 | 2971.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 2993.95 | 2966.76 | 2973.11 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 11:15:00 | 3007.00 | 2979.46 | 2978.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 12:15:00 | 3020.40 | 2987.64 | 2981.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 3005.10 | 3010.10 | 3001.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 15:00:00 | 3005.10 | 3010.10 | 3001.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 3001.95 | 3008.47 | 3001.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 3004.95 | 3008.47 | 3001.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 2984.40 | 3003.66 | 2999.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 2984.40 | 3003.66 | 2999.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 2976.65 | 2998.26 | 2997.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 2956.85 | 2998.26 | 2997.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 2937.65 | 2986.13 | 2992.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 12:15:00 | 2924.70 | 2973.85 | 2985.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 2932.75 | 2924.62 | 2946.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 14:00:00 | 2932.75 | 2924.62 | 2946.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 2921.90 | 2926.57 | 2942.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:15:00 | 2909.60 | 2926.57 | 2942.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 12:15:00 | 2973.65 | 2926.54 | 2927.08 | SL hit (close>static) qty=1.00 sl=2958.30 alert=retest2 |

### Cycle 56 — BUY (started 2024-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 13:15:00 | 2968.75 | 2934.98 | 2930.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 15:15:00 | 2984.00 | 2950.78 | 2939.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 09:15:00 | 3000.40 | 3008.38 | 2984.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 11:15:00 | 2996.05 | 3003.07 | 2985.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 2996.05 | 3003.07 | 2985.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 12:30:00 | 3001.15 | 3000.31 | 2985.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 13:45:00 | 3005.40 | 3001.04 | 2987.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 15:15:00 | 3009.00 | 2999.94 | 2988.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 14:45:00 | 3003.70 | 3005.27 | 2999.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 2995.00 | 3003.22 | 2998.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 3017.20 | 3003.22 | 2998.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-02 13:15:00 | 2988.65 | 3005.66 | 3002.99 | SL hit (close<static) qty=1.00 sl=2990.75 alert=retest2 |

### Cycle 57 — SELL (started 2024-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 14:15:00 | 2971.45 | 2998.82 | 3000.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 2936.00 | 2973.92 | 2985.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 11:15:00 | 2958.10 | 2953.17 | 2970.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-06 11:45:00 | 2954.00 | 2953.17 | 2970.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 13:15:00 | 2959.15 | 2955.22 | 2968.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 14:00:00 | 2959.15 | 2955.22 | 2968.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 15:15:00 | 2964.90 | 2957.31 | 2967.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:15:00 | 2963.00 | 2957.31 | 2967.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 2947.70 | 2955.38 | 2965.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 10:30:00 | 2932.65 | 2952.92 | 2963.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:15:00 | 2930.60 | 2952.92 | 2963.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 09:15:00 | 3025.10 | 2965.08 | 2963.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 09:15:00 | 3025.10 | 2965.08 | 2963.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 11:15:00 | 3042.00 | 2989.10 | 2975.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 3020.00 | 3052.00 | 3018.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 3020.00 | 3052.00 | 3018.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 3020.00 | 3052.00 | 3018.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 3020.00 | 3052.00 | 3018.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 2944.20 | 3030.44 | 3011.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 2944.20 | 3030.44 | 3011.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 2952.60 | 3014.87 | 3006.17 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 13:15:00 | 2962.00 | 2994.68 | 2997.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 2912.00 | 2970.49 | 2985.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 2923.80 | 2899.35 | 2931.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 10:15:00 | 2923.80 | 2899.35 | 2931.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 2923.80 | 2899.35 | 2931.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:45:00 | 2912.90 | 2899.35 | 2931.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 2922.00 | 2904.50 | 2925.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 14:00:00 | 2922.00 | 2904.50 | 2925.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 2919.70 | 2907.54 | 2925.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 14:30:00 | 2916.45 | 2907.54 | 2925.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 2915.35 | 2909.10 | 2924.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 2885.90 | 2909.10 | 2924.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 14:15:00 | 2951.65 | 2920.36 | 2922.91 | SL hit (close>static) qty=1.00 sl=2928.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 2960.00 | 2928.29 | 2926.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 3000.45 | 2942.72 | 2933.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 3056.25 | 3056.56 | 3021.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 10:00:00 | 3056.25 | 3056.56 | 3021.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 3039.00 | 3050.40 | 3024.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 11:30:00 | 3028.80 | 3050.40 | 3024.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 3023.00 | 3042.14 | 3025.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 14:00:00 | 3023.00 | 3042.14 | 3025.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 3022.90 | 3038.29 | 3025.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 15:00:00 | 3022.90 | 3038.29 | 3025.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 3028.00 | 3036.24 | 3025.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:15:00 | 2989.65 | 3036.24 | 3025.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 2997.50 | 3028.49 | 3022.84 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 11:15:00 | 3004.70 | 3018.45 | 3018.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 2985.10 | 3000.89 | 3007.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 13:15:00 | 2977.90 | 2974.28 | 2986.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 13:15:00 | 2977.90 | 2974.28 | 2986.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 2977.90 | 2974.28 | 2986.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:00:00 | 2977.90 | 2974.28 | 2986.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 3005.70 | 2980.57 | 2988.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 3005.70 | 2980.57 | 2988.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 3008.00 | 2986.05 | 2990.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 3024.90 | 2986.05 | 2990.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 3043.90 | 2997.62 | 2995.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 10:15:00 | 3053.45 | 3008.79 | 3000.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 14:15:00 | 3049.50 | 3050.24 | 3034.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 15:00:00 | 3049.50 | 3050.24 | 3034.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 3116.20 | 3064.03 | 3043.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 09:15:00 | 3155.00 | 3089.20 | 3067.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 11:00:00 | 3127.80 | 3106.33 | 3079.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 12:00:00 | 3123.60 | 3109.78 | 3083.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-29 10:15:00 | 3037.05 | 3074.38 | 3076.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 10:15:00 | 3037.05 | 3074.38 | 3076.02 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 3099.00 | 3076.16 | 3075.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 3115.40 | 3084.01 | 3078.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 3167.35 | 3219.39 | 3198.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 3167.35 | 3219.39 | 3198.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 3167.35 | 3219.39 | 3198.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 3167.35 | 3219.39 | 3198.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 3191.05 | 3213.72 | 3197.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 14:15:00 | 3216.65 | 3204.27 | 3196.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 15:15:00 | 3274.90 | 3296.30 | 3298.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 15:15:00 | 3274.90 | 3296.30 | 3298.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 3129.00 | 3262.84 | 3283.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 3125.25 | 3108.12 | 3170.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:00:00 | 3125.25 | 3108.12 | 3170.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 3172.00 | 3131.84 | 3163.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 3172.00 | 3131.84 | 3163.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 3168.00 | 3139.07 | 3163.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 3156.60 | 3139.07 | 3163.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 3075.00 | 3126.26 | 3155.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 3056.30 | 3126.26 | 3155.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 13:45:00 | 3073.00 | 3087.14 | 3124.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 10:45:00 | 3060.50 | 3101.49 | 3112.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:15:00 | 2919.35 | 3029.03 | 3067.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 3069.15 | 3014.03 | 3039.54 | SL hit (close>ema200) qty=0.50 sl=3014.03 alert=retest2 |

### Cycle 66 — BUY (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 13:15:00 | 3103.00 | 3059.87 | 3055.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 3127.40 | 3073.38 | 3062.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 3253.10 | 3254.30 | 3198.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 10:00:00 | 3253.10 | 3254.30 | 3198.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 3537.10 | 3542.52 | 3520.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:30:00 | 3516.70 | 3542.52 | 3520.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 3550.50 | 3561.64 | 3548.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 13:30:00 | 3553.85 | 3561.64 | 3548.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 3562.45 | 3561.81 | 3549.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 09:45:00 | 3579.85 | 3565.16 | 3553.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 12:45:00 | 3573.85 | 3564.79 | 3556.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 13:15:00 | 3534.65 | 3558.76 | 3554.15 | SL hit (close<static) qty=1.00 sl=3549.65 alert=retest2 |

### Cycle 67 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 3936.10 | 3973.10 | 3973.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 10:15:00 | 3922.35 | 3945.47 | 3956.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 3774.90 | 3759.56 | 3810.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 3774.90 | 3759.56 | 3810.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 3809.75 | 3773.52 | 3808.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:45:00 | 3813.40 | 3773.52 | 3808.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 3864.95 | 3791.81 | 3813.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:00:00 | 3864.95 | 3791.81 | 3813.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 3839.00 | 3801.25 | 3815.82 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 09:15:00 | 3902.20 | 3838.07 | 3830.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 4009.00 | 3916.11 | 3883.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 5108.95 | 5122.56 | 5014.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 10:00:00 | 5108.95 | 5122.56 | 5014.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 5075.00 | 5121.74 | 5068.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:30:00 | 5093.70 | 5121.74 | 5068.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 5049.40 | 5107.27 | 5066.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:00:00 | 5049.40 | 5107.27 | 5066.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 4988.00 | 5083.42 | 5059.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 4988.00 | 5083.42 | 5059.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 5014.45 | 5045.82 | 5045.99 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 5130.00 | 5057.25 | 5050.82 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 10:15:00 | 5021.00 | 5053.10 | 5054.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 5005.00 | 5043.48 | 5050.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 4975.35 | 4959.11 | 4992.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 4975.35 | 4959.11 | 4992.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 4975.35 | 4959.11 | 4992.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 4975.35 | 4959.11 | 4992.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 5009.00 | 4970.35 | 4991.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 5186.30 | 4970.35 | 4991.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 5302.95 | 5036.87 | 5020.19 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 4543.75 | 5049.48 | 5070.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 4218.95 | 4883.38 | 4992.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 4364.75 | 4359.09 | 4569.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 4364.75 | 4359.09 | 4569.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 4719.25 | 4428.71 | 4564.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 4719.25 | 4428.71 | 4564.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 4777.25 | 4498.42 | 4583.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 4790.25 | 4498.42 | 4583.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 4668.40 | 4568.35 | 4597.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:45:00 | 4679.60 | 4568.35 | 4597.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 4692.85 | 4621.51 | 4617.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 4748.00 | 4674.85 | 4645.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 11:15:00 | 4889.35 | 4890.38 | 4858.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 11:15:00 | 4889.35 | 4890.38 | 4858.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 4889.35 | 4890.38 | 4858.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 4904.00 | 4890.38 | 4858.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 5394.40 | 5208.54 | 5087.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 5239.65 | 5299.10 | 5301.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 5202.25 | 5257.24 | 5279.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 5249.85 | 5227.36 | 5256.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 5249.85 | 5227.36 | 5256.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 5249.85 | 5227.36 | 5256.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 5249.85 | 5227.36 | 5256.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 5273.30 | 5236.55 | 5257.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 5287.10 | 5236.55 | 5257.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 5293.50 | 5247.94 | 5260.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:00:00 | 5293.50 | 5247.94 | 5260.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 14:15:00 | 5329.70 | 5277.39 | 5272.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 5388.00 | 5307.93 | 5287.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 09:15:00 | 5332.40 | 5347.89 | 5323.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 5332.40 | 5347.89 | 5323.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 5332.40 | 5347.89 | 5323.24 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 5263.00 | 5308.77 | 5314.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 5239.15 | 5294.85 | 5308.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 5288.30 | 5276.36 | 5293.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 5288.30 | 5276.36 | 5293.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 5288.30 | 5276.36 | 5293.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 5288.30 | 5276.36 | 5293.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 5274.00 | 5275.89 | 5291.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 5341.00 | 5275.89 | 5291.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 5298.00 | 5280.31 | 5292.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:15:00 | 5273.75 | 5280.31 | 5292.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 5261.00 | 5281.86 | 5290.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 5271.00 | 5271.83 | 5283.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:00:00 | 5271.80 | 5270.73 | 5280.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 5267.90 | 5270.16 | 5279.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-01 14:15:00 | 5399.20 | 5296.08 | 5288.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 14:15:00 | 5399.20 | 5296.08 | 5288.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 13:15:00 | 5420.50 | 5378.08 | 5348.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 15:15:00 | 5544.00 | 5545.27 | 5499.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:15:00 | 5618.45 | 5545.27 | 5499.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 5539.90 | 5587.91 | 5556.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 5539.90 | 5587.91 | 5556.92 | SL hit (close<ema400) qty=1.00 sl=5556.92 alert=retest1 |

### Cycle 79 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 5470.95 | 5540.38 | 5544.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 5449.50 | 5522.21 | 5536.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 15:15:00 | 5495.00 | 5490.80 | 5512.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 09:15:00 | 5533.00 | 5490.80 | 5512.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 5486.75 | 5489.99 | 5509.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 5477.50 | 5489.99 | 5509.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 13:15:00 | 5554.05 | 5512.58 | 5514.84 | SL hit (close>static) qty=1.00 sl=5541.90 alert=retest2 |

### Cycle 80 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 5542.95 | 5518.66 | 5517.40 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 5486.25 | 5515.14 | 5517.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 12:15:00 | 5410.90 | 5464.16 | 5484.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 4912.80 | 4896.22 | 5037.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 4912.80 | 4896.22 | 5037.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 5017.00 | 4930.93 | 5019.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 5017.00 | 4930.93 | 5019.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 5026.00 | 4949.94 | 5019.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 5008.50 | 4949.94 | 5019.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 5005.15 | 4960.98 | 5018.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 4712.10 | 4999.09 | 5020.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 15:15:00 | 4908.00 | 4893.05 | 4892.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 15:15:00 | 4908.00 | 4893.05 | 4892.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 4958.00 | 4906.04 | 4898.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 4962.20 | 4999.03 | 4963.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 11:15:00 | 4962.20 | 4999.03 | 4963.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 4962.20 | 4999.03 | 4963.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:00:00 | 4962.20 | 4999.03 | 4963.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 4972.00 | 4993.62 | 4963.96 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 14:15:00 | 4916.55 | 4956.45 | 4958.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 4897.65 | 4938.86 | 4949.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 4661.65 | 4625.81 | 4693.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 4661.65 | 4625.81 | 4693.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4661.65 | 4625.81 | 4693.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 4585.20 | 4614.28 | 4666.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:00:00 | 4590.00 | 4580.83 | 4635.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 14:15:00 | 4746.40 | 4657.40 | 4655.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 4746.40 | 4657.40 | 4655.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 4750.65 | 4727.86 | 4712.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 15:15:00 | 4724.00 | 4727.10 | 4715.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 15:15:00 | 4724.00 | 4727.10 | 4715.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 4724.00 | 4727.10 | 4715.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 4721.05 | 4725.89 | 4715.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 4724.95 | 4725.70 | 4716.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 4712.35 | 4725.70 | 4716.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 4741.65 | 4733.59 | 4721.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 4741.65 | 4733.59 | 4721.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 4717.90 | 4730.45 | 4721.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 4717.90 | 4730.45 | 4721.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 4699.30 | 4724.22 | 4719.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 4699.30 | 4724.22 | 4719.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 4694.00 | 4718.18 | 4717.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 4634.00 | 4718.18 | 4717.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 4646.00 | 4703.74 | 4710.74 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 4732.50 | 4697.57 | 4696.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 4751.25 | 4708.31 | 4701.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 4752.70 | 4779.61 | 4756.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 4752.70 | 4779.61 | 4756.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 4752.70 | 4779.61 | 4756.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:30:00 | 4751.50 | 4779.61 | 4756.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 4727.75 | 4769.24 | 4754.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 4717.20 | 4769.24 | 4754.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 14:15:00 | 4737.90 | 4745.82 | 4746.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 10:15:00 | 4726.10 | 4741.22 | 4743.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 15:15:00 | 4740.00 | 4735.91 | 4739.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 15:15:00 | 4740.00 | 4735.91 | 4739.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 4740.00 | 4735.91 | 4739.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:15:00 | 4744.35 | 4735.91 | 4739.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 4734.75 | 4735.68 | 4739.31 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 4793.00 | 4748.60 | 4744.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 09:15:00 | 4850.05 | 4776.34 | 4759.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 09:15:00 | 4807.45 | 4812.21 | 4790.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 09:30:00 | 4825.05 | 4812.21 | 4790.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 4776.60 | 4805.09 | 4789.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 4776.60 | 4805.09 | 4789.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 4790.15 | 4802.10 | 4789.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:30:00 | 4785.80 | 4802.10 | 4789.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 4791.35 | 4799.95 | 4789.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:30:00 | 4786.25 | 4799.95 | 4789.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 4802.00 | 4800.36 | 4790.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:30:00 | 4794.00 | 4800.36 | 4790.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 4794.00 | 4799.16 | 4792.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 4767.00 | 4799.16 | 4792.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 4744.00 | 4788.13 | 4787.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 4744.00 | 4788.13 | 4787.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 10:15:00 | 4754.55 | 4781.41 | 4784.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 4719.05 | 4754.88 | 4769.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 4680.00 | 4632.98 | 4667.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 4680.00 | 4632.98 | 4667.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 4680.00 | 4632.98 | 4667.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 4680.00 | 4632.98 | 4667.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 4667.60 | 4639.91 | 4667.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:15:00 | 4662.55 | 4639.91 | 4667.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 15:15:00 | 4688.45 | 4663.79 | 4671.29 | SL hit (close>static) qty=1.00 sl=4682.50 alert=retest2 |

### Cycle 90 — BUY (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 14:15:00 | 4686.80 | 4676.90 | 4675.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 4821.50 | 4707.23 | 4689.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 13:15:00 | 4858.40 | 4858.92 | 4805.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 14:00:00 | 4858.40 | 4858.92 | 4805.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 4807.00 | 4849.81 | 4814.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:00:00 | 4807.00 | 4849.81 | 4814.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 4814.70 | 4842.79 | 4814.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:30:00 | 4798.00 | 4842.79 | 4814.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 4820.00 | 4838.23 | 4814.98 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 4737.00 | 4800.00 | 4804.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 4700.95 | 4752.06 | 4776.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 4652.00 | 4650.35 | 4703.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 4652.00 | 4650.35 | 4703.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 4684.60 | 4660.98 | 4699.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 4668.25 | 4682.63 | 4695.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:00:00 | 4666.30 | 4679.36 | 4692.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:45:00 | 4668.50 | 4637.36 | 4647.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:15:00 | 4434.84 | 4548.32 | 4592.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:15:00 | 4435.07 | 4548.32 | 4592.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 12:15:00 | 4432.98 | 4528.36 | 4579.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-19 11:15:00 | 4201.43 | 4346.40 | 4433.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 92 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 4443.55 | 4372.18 | 4366.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 4451.75 | 4388.09 | 4374.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 4403.00 | 4410.59 | 4391.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 10:15:00 | 4401.25 | 4408.72 | 4392.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 4401.25 | 4408.72 | 4392.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:45:00 | 4400.00 | 4408.72 | 4392.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 4380.60 | 4403.10 | 4391.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:30:00 | 4365.55 | 4403.10 | 4391.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 4385.05 | 4399.49 | 4390.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:00:00 | 4385.05 | 4399.49 | 4390.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 4415.65 | 4402.72 | 4392.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 4428.00 | 4402.23 | 4394.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 4371.00 | 4390.49 | 4392.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 4371.00 | 4390.49 | 4392.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 11:15:00 | 4350.40 | 4379.34 | 4386.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 4368.55 | 4363.29 | 4376.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 14:45:00 | 4362.45 | 4363.29 | 4376.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 4388.00 | 4368.23 | 4377.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 4414.20 | 4368.23 | 4377.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 4414.70 | 4377.53 | 4380.70 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 4425.00 | 4391.10 | 4386.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 14:15:00 | 4480.30 | 4419.20 | 4401.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 4402.60 | 4424.81 | 4407.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 4402.60 | 4424.81 | 4407.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 4402.60 | 4424.81 | 4407.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 4396.00 | 4424.81 | 4407.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 4413.85 | 4422.62 | 4408.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 4401.95 | 4422.62 | 4408.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 4427.50 | 4423.59 | 4409.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:30:00 | 4407.50 | 4423.59 | 4409.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 4411.10 | 4422.60 | 4411.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:30:00 | 4436.60 | 4422.60 | 4411.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 4420.75 | 4422.23 | 4412.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:30:00 | 4403.10 | 4422.23 | 4412.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 4420.05 | 4421.79 | 4413.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 4429.75 | 4421.79 | 4413.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 13:00:00 | 4427.50 | 4421.43 | 4415.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 4399.40 | 4417.54 | 4415.80 | SL hit (close<static) qty=1.00 sl=4412.90 alert=retest2 |

### Cycle 95 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 4389.00 | 4411.83 | 4413.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 4318.05 | 4393.08 | 4404.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 4316.20 | 4313.34 | 4351.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:45:00 | 4314.50 | 4313.34 | 4351.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 4224.00 | 4274.20 | 4314.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 4194.90 | 4274.20 | 4314.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:30:00 | 4194.40 | 4245.11 | 4293.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 4326.00 | 4231.77 | 4258.39 | SL hit (close>static) qty=1.00 sl=4318.65 alert=retest2 |

### Cycle 96 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 4353.95 | 4287.86 | 4280.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 4374.00 | 4316.63 | 4295.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 4453.00 | 4462.13 | 4420.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:00:00 | 4453.00 | 4462.13 | 4420.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 4542.20 | 4614.48 | 4579.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 4542.20 | 4614.48 | 4579.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 4540.15 | 4599.61 | 4575.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:30:00 | 4556.10 | 4580.99 | 4570.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 14:15:00 | 4519.40 | 4559.07 | 4562.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 4519.40 | 4559.07 | 4562.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 15:15:00 | 4511.00 | 4549.46 | 4557.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 4543.60 | 4530.47 | 4545.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 4543.60 | 4530.47 | 4545.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 4543.60 | 4530.47 | 4545.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 4543.60 | 4530.47 | 4545.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 4513.00 | 4526.98 | 4542.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 4496.95 | 4528.38 | 4535.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 4272.10 | 4351.43 | 4423.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 15:15:00 | 4182.00 | 4170.16 | 4219.55 | SL hit (close>ema200) qty=0.50 sl=4170.16 alert=retest2 |

### Cycle 98 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 4282.45 | 4193.20 | 4192.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 4290.00 | 4226.58 | 4208.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 4238.25 | 4255.53 | 4230.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 4238.25 | 4255.53 | 4230.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 4230.00 | 4247.33 | 4231.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 4218.00 | 4247.33 | 4231.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 4235.35 | 4244.93 | 4231.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 11:45:00 | 4265.80 | 4248.60 | 4235.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 12:15:00 | 4261.85 | 4248.60 | 4235.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 4276.00 | 4251.98 | 4241.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 09:30:00 | 4268.90 | 4255.90 | 4245.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 4208.95 | 4246.51 | 4242.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 4208.95 | 4246.51 | 4242.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 4219.60 | 4241.13 | 4240.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-04 13:15:00 | 4212.00 | 4234.48 | 4237.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 4212.00 | 4234.48 | 4237.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 4151.30 | 4209.85 | 4224.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 4238.35 | 4196.58 | 4211.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 4238.35 | 4196.58 | 4211.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 4238.35 | 4196.58 | 4211.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:45:00 | 4238.15 | 4196.58 | 4211.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 4278.65 | 4213.00 | 4217.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 4278.65 | 4213.00 | 4217.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 4255.00 | 4221.40 | 4221.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 4338.00 | 4244.72 | 4231.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 4416.05 | 4430.98 | 4385.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 12:00:00 | 4416.05 | 4430.98 | 4385.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 4403.10 | 4427.12 | 4395.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 4420.30 | 4427.12 | 4395.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 4380.00 | 4417.70 | 4394.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 4385.45 | 4417.70 | 4394.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 4411.05 | 4416.37 | 4395.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:15:00 | 4441.30 | 4416.37 | 4395.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 11:15:00 | 4326.85 | 4396.46 | 4402.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 11:15:00 | 4326.85 | 4396.46 | 4402.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 4300.95 | 4377.36 | 4393.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 4138.90 | 4093.87 | 4173.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 13:00:00 | 4138.90 | 4093.87 | 4173.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 4134.00 | 4101.90 | 4169.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:15:00 | 4087.05 | 4101.90 | 4169.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 4192.90 | 4118.43 | 4160.28 | SL hit (close>static) qty=1.00 sl=4185.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 4113.60 | 4070.65 | 4066.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 4278.00 | 4118.24 | 4089.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 15:15:00 | 4459.00 | 4468.77 | 4420.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 09:15:00 | 4486.70 | 4468.77 | 4420.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 4438.00 | 4459.61 | 4428.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 4423.80 | 4459.61 | 4428.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 4489.85 | 4469.43 | 4444.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 4532.95 | 4489.20 | 4466.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 4580.50 | 4506.57 | 4487.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 4559.70 | 4517.28 | 4503.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 13:15:00 | 4631.30 | 4663.90 | 4666.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 4631.30 | 4663.90 | 4666.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 4624.15 | 4655.95 | 4662.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 4257.35 | 4239.89 | 4298.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 4257.35 | 4239.89 | 4298.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 4266.00 | 4231.26 | 4240.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 4266.00 | 4231.26 | 4240.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 4231.80 | 4231.37 | 4239.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:45:00 | 4235.85 | 4231.37 | 4239.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 4240.00 | 4233.10 | 4239.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 4233.40 | 4233.10 | 4239.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 4214.70 | 4229.42 | 4237.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:45:00 | 4231.25 | 4229.42 | 4237.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 4196.00 | 4162.08 | 4181.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 4196.00 | 4162.08 | 4181.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 4177.25 | 4165.12 | 4181.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 4167.10 | 4168.29 | 4181.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 14:45:00 | 4170.80 | 4162.84 | 4171.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 4218.70 | 4174.84 | 4175.85 | SL hit (close>static) qty=1.00 sl=4196.35 alert=retest2 |

### Cycle 104 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 4220.00 | 4183.87 | 4179.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 10:15:00 | 4243.90 | 4223.97 | 4206.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 4219.45 | 4228.28 | 4213.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 4219.45 | 4228.28 | 4213.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 4207.10 | 4224.05 | 4212.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 4207.10 | 4224.05 | 4212.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 4203.00 | 4219.84 | 4211.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 4179.00 | 4219.84 | 4211.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 4182.65 | 4212.40 | 4208.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 4166.20 | 4212.40 | 4208.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 4109.85 | 4191.89 | 4199.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 4094.40 | 4156.01 | 4180.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 4148.45 | 4138.82 | 4164.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 4148.45 | 4138.82 | 4164.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 4148.45 | 4138.82 | 4164.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 4165.85 | 4138.82 | 4164.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 4170.80 | 4145.21 | 4165.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 4170.80 | 4145.21 | 4165.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 4170.00 | 4150.17 | 4165.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 4185.00 | 4150.17 | 4165.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 4181.00 | 4161.62 | 4168.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 4191.00 | 4161.62 | 4168.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 4164.45 | 4162.47 | 4167.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 4069.90 | 4128.00 | 4145.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 4096.00 | 4123.20 | 4141.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:30:00 | 4097.35 | 4115.96 | 4131.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 3891.20 | 3977.75 | 4036.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 3892.48 | 3977.75 | 4036.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 3866.40 | 3951.92 | 4019.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 3864.85 | 3860.87 | 3941.02 | SL hit (close>ema200) qty=0.50 sl=3860.87 alert=retest2 |

### Cycle 106 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 3936.00 | 3882.51 | 3880.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 4052.35 | 3928.23 | 3903.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 4161.30 | 4167.93 | 4097.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 4161.30 | 4167.93 | 4097.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 4069.05 | 4148.15 | 4095.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 4069.05 | 4148.15 | 4095.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 4115.55 | 4141.63 | 4097.11 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 3926.75 | 4065.42 | 4075.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 3876.25 | 4027.58 | 4057.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 3938.25 | 3930.31 | 3983.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:45:00 | 3944.00 | 3930.31 | 3983.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 3884.00 | 3918.93 | 3952.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 3844.65 | 3889.41 | 3924.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 3652.42 | 3688.11 | 3782.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 3650.55 | 3640.09 | 3733.06 | SL hit (close>ema200) qty=0.50 sl=3640.09 alert=retest2 |

### Cycle 108 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 3795.00 | 3717.63 | 3712.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 3903.65 | 3786.02 | 3750.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 3939.00 | 3941.42 | 3871.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:45:00 | 3947.95 | 3941.42 | 3871.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 3703.75 | 3893.88 | 3856.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 3703.75 | 3893.88 | 3856.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 3807.35 | 3876.58 | 3851.79 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 3566.05 | 3786.25 | 3814.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 3533.10 | 3699.42 | 3767.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 3672.95 | 3614.70 | 3690.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 3672.95 | 3614.70 | 3690.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 3672.95 | 3614.70 | 3690.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 3672.95 | 3614.70 | 3690.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 3691.20 | 3630.00 | 3691.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 3691.20 | 3630.00 | 3691.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 3701.00 | 3644.20 | 3691.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:45:00 | 3699.90 | 3644.20 | 3691.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 3713.05 | 3657.97 | 3693.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:00:00 | 3713.05 | 3657.97 | 3693.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 3705.30 | 3667.44 | 3694.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:45:00 | 3715.65 | 3667.44 | 3694.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 3816.75 | 3716.58 | 3712.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 3835.10 | 3740.28 | 3723.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 3777.65 | 3793.26 | 3766.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 3777.65 | 3793.26 | 3766.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 3775.95 | 3788.29 | 3768.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 3775.95 | 3788.29 | 3768.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 3754.10 | 3781.45 | 3767.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 3758.85 | 3781.45 | 3767.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 3767.50 | 3778.66 | 3767.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 3813.80 | 3778.33 | 3768.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:00:00 | 3786.15 | 3779.89 | 3769.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 3793.95 | 3787.91 | 3774.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:45:00 | 3796.85 | 3792.86 | 3783.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 3800.40 | 3794.37 | 3785.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 11:30:00 | 3810.25 | 3794.87 | 3786.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 3687.70 | 3766.67 | 3775.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 3687.70 | 3766.67 | 3775.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 3656.00 | 3744.54 | 3764.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 3631.00 | 3620.76 | 3673.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 3623.00 | 3620.76 | 3673.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 3674.90 | 3621.23 | 3655.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 3677.15 | 3621.23 | 3655.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 3722.00 | 3641.38 | 3661.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 3722.00 | 3641.38 | 3661.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 3706.10 | 3654.33 | 3665.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 3706.10 | 3654.33 | 3665.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 3657.55 | 3666.16 | 3669.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:45:00 | 3675.00 | 3666.16 | 3669.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 3608.60 | 3654.62 | 3663.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 3574.55 | 3654.62 | 3663.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 3395.82 | 3540.00 | 3591.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 3392.55 | 3370.17 | 3435.30 | SL hit (close>ema200) qty=0.50 sl=3370.17 alert=retest2 |

### Cycle 112 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 3319.10 | 3205.65 | 3197.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 3359.90 | 3289.95 | 3248.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 3411.00 | 3411.97 | 3369.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 3427.15 | 3411.97 | 3369.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 3468.30 | 3423.24 | 3378.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:30:00 | 3481.00 | 3434.69 | 3387.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 11:00:00 | 3480.50 | 3434.69 | 3387.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 3521.55 | 3449.17 | 3413.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 11:15:00 | 3406.25 | 3432.13 | 3433.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 3406.25 | 3432.13 | 3433.21 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 3464.50 | 3437.10 | 3433.57 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 3397.50 | 3431.46 | 3432.62 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 3436.00 | 3432.41 | 3432.32 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 3417.20 | 3429.37 | 3430.95 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 3440.70 | 3432.54 | 3432.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 3551.00 | 3457.59 | 3443.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 11:15:00 | 4057.00 | 4071.98 | 3980.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 11:45:00 | 4069.35 | 4071.98 | 3980.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 4041.50 | 4041.73 | 3998.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 4019.00 | 4041.73 | 3998.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 3992.45 | 4031.88 | 3997.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 3992.45 | 4031.88 | 3997.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 4120.40 | 4049.58 | 4008.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 12:15:00 | 4146.30 | 4049.58 | 4008.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 4204.05 | 4103.17 | 4050.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 3896.20 | 4180.74 | 4216.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 3896.20 | 4180.74 | 4216.09 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 4137.05 | 4074.69 | 4069.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 4206.80 | 4117.73 | 4093.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 4202.00 | 4206.46 | 4179.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:45:00 | 4236.90 | 4216.88 | 4189.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 13:30:00 | 4239.40 | 4224.68 | 4197.98 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 4242.00 | 4222.99 | 4201.80 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 4290.10 | 4297.75 | 4276.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:45:00 | 4298.00 | 4297.75 | 4276.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 4277.00 | 4293.60 | 4276.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 4349.00 | 4293.60 | 4276.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 4300.30 | 4291.08 | 4276.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:15:00 | 4296.10 | 4291.09 | 4277.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:00:00 | 4294.80 | 4291.83 | 4279.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 4300.50 | 4293.56 | 4281.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 4287.10 | 4293.56 | 4281.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 4307.70 | 4309.05 | 4296.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 4309.60 | 4309.05 | 4296.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 14:15:00 | 4309.10 | 4308.50 | 4297.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 14:15:00 | 4296.60 | 4306.12 | 4297.73 | SL hit (close<ema400) qty=1.00 sl=4297.73 alert=retest1 |

### Cycle 121 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 4218.80 | 4286.86 | 4290.32 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 4413.20 | 4290.56 | 4279.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 4431.60 | 4318.76 | 4293.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 4496.70 | 4541.67 | 4463.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 09:30:00 | 4494.00 | 4541.67 | 4463.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 4481.00 | 4507.61 | 4478.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 4535.90 | 4507.61 | 4478.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:30:00 | 4497.40 | 4519.23 | 4499.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 4495.00 | 4542.61 | 4544.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 4495.00 | 4542.61 | 4544.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 4393.00 | 4457.53 | 4485.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 09:15:00 | 4524.70 | 4470.96 | 4489.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 4524.70 | 4470.96 | 4489.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 4524.70 | 4470.96 | 4489.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:45:00 | 4518.50 | 4470.96 | 4489.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 4506.00 | 4477.97 | 4490.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 13:00:00 | 4486.50 | 4485.12 | 4492.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 4465.60 | 4491.06 | 4493.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 09:15:00 | 4582.80 | 4481.11 | 4481.18 | SL hit (close>static) qty=1.00 sl=4535.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 10:15:00 | 4598.00 | 4504.48 | 4491.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 11:15:00 | 4633.20 | 4530.23 | 4504.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 5025.80 | 5047.99 | 4930.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:45:00 | 5011.50 | 5047.99 | 4930.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 4920.00 | 5011.52 | 4969.72 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 4879.80 | 4948.03 | 4949.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 4855.20 | 4929.47 | 4941.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 4940.00 | 4919.18 | 4933.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 4940.00 | 4919.18 | 4933.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 4940.00 | 4919.18 | 4933.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 4940.00 | 4919.18 | 4933.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 4956.00 | 4926.54 | 4935.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 4956.00 | 4926.54 | 4935.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 4931.40 | 4927.51 | 4935.37 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 4986.80 | 4947.77 | 4943.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 14:15:00 | 5000.00 | 4958.22 | 4948.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 4976.00 | 5011.91 | 4991.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 4976.00 | 5011.91 | 4991.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 4976.00 | 5011.91 | 4991.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 4978.50 | 5011.91 | 4991.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 4998.00 | 5009.13 | 4991.79 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 4956.70 | 4980.97 | 4984.15 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 5017.00 | 4989.53 | 4987.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 5087.90 | 5019.78 | 5004.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 5025.00 | 5030.50 | 5019.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 5025.00 | 5030.50 | 5019.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 5025.00 | 5030.50 | 5019.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 5011.40 | 5030.50 | 5019.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 5028.10 | 5030.02 | 5020.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:45:00 | 5018.50 | 5030.02 | 5020.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 5025.00 | 5032.00 | 5023.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:45:00 | 5028.00 | 5032.00 | 5023.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 5019.40 | 5029.48 | 5023.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 5019.00 | 5029.48 | 5023.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 5020.50 | 5027.69 | 5023.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 5017.90 | 5027.69 | 5023.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 4985.80 | 5015.25 | 5017.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 4976.60 | 5000.63 | 5010.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 5005.00 | 5000.92 | 5008.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 15:15:00 | 5005.00 | 5000.92 | 5008.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 5005.00 | 5000.92 | 5008.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 5044.00 | 5000.92 | 5008.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 5029.70 | 5006.68 | 5010.40 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 5040.80 | 5013.50 | 5013.16 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 4991.90 | 5010.83 | 5012.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 4977.90 | 5004.25 | 5009.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 11:15:00 | 5011.00 | 4995.45 | 5002.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 11:15:00 | 5011.00 | 4995.45 | 5002.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 5011.00 | 4995.45 | 5002.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:45:00 | 5009.00 | 4995.45 | 5002.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 5029.00 | 5002.16 | 5004.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:00:00 | 5029.00 | 5002.16 | 5004.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 5023.00 | 5006.33 | 5006.22 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 4977.00 | 5005.54 | 5007.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 4926.30 | 4982.13 | 4995.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 4957.00 | 4956.88 | 4977.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 13:30:00 | 4948.80 | 4956.88 | 4977.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 5027.90 | 4972.34 | 4979.19 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 5017.40 | 4987.87 | 4985.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 5032.70 | 4996.84 | 4989.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 5008.50 | 5025.39 | 5007.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 5008.50 | 5025.39 | 5007.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 5008.50 | 5025.39 | 5007.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 5010.00 | 5025.39 | 5007.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 4994.10 | 5019.14 | 5006.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 4999.50 | 5019.14 | 5006.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 4981.40 | 5011.59 | 5004.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 4981.40 | 5011.59 | 5004.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 4994.80 | 5002.69 | 5001.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 4994.80 | 5002.69 | 5001.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 4982.00 | 4998.55 | 4999.82 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 5018.50 | 5002.76 | 5001.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 5115.00 | 5028.39 | 5014.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 5030.00 | 5080.79 | 5064.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 5030.00 | 5080.79 | 5064.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 5030.00 | 5080.79 | 5064.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 5042.10 | 5080.79 | 5064.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 5052.20 | 5075.07 | 5063.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 5055.10 | 5067.86 | 5061.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:45:00 | 5054.70 | 5066.63 | 5061.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 5041.50 | 5056.72 | 5057.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 5041.50 | 5056.72 | 5057.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 4954.90 | 5033.04 | 5046.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 5059.00 | 5015.03 | 5032.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 5059.00 | 5015.03 | 5032.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5059.00 | 5015.03 | 5032.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 5059.00 | 5015.03 | 5032.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 5056.00 | 5023.23 | 5035.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 5056.00 | 5023.23 | 5035.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 4996.30 | 5019.98 | 5031.62 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 5059.30 | 5034.84 | 5033.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 5097.80 | 5057.16 | 5045.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 5063.70 | 5071.85 | 5056.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 5063.70 | 5071.85 | 5056.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 5063.70 | 5071.85 | 5056.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 5063.70 | 5071.85 | 5056.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 5060.00 | 5069.48 | 5056.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 5060.00 | 5069.48 | 5056.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 5052.60 | 5066.10 | 5056.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 5052.60 | 5066.10 | 5056.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 5043.30 | 5061.54 | 5055.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 5038.00 | 5061.54 | 5055.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 5030.60 | 5055.41 | 5053.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 5024.90 | 5055.41 | 5053.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 5028.00 | 5049.93 | 5051.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 4999.90 | 5039.92 | 5046.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 4921.90 | 4921.75 | 4961.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 4921.90 | 4921.75 | 4961.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 4975.00 | 4932.01 | 4955.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 4975.00 | 4932.01 | 4955.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 4960.00 | 4937.61 | 4956.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 5011.20 | 4937.61 | 4956.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 5004.90 | 4951.06 | 4960.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 5006.80 | 4951.06 | 4960.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 5042.00 | 4980.43 | 4973.10 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 10:15:00 | 4908.00 | 4975.85 | 4978.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 15:15:00 | 4883.00 | 4920.53 | 4947.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 10:15:00 | 4823.50 | 4822.54 | 4866.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:30:00 | 4821.00 | 4822.54 | 4866.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 4879.30 | 4829.00 | 4849.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 4875.90 | 4829.00 | 4849.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 4877.00 | 4838.60 | 4851.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:45:00 | 4864.00 | 4840.58 | 4851.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:00:00 | 4865.00 | 4845.47 | 4852.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:15:00 | 4866.00 | 4850.43 | 4854.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 14:15:00 | 4896.70 | 4859.69 | 4858.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 4896.70 | 4859.69 | 4858.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 4904.00 | 4881.20 | 4869.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 4875.10 | 4879.98 | 4870.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 4875.10 | 4879.98 | 4870.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 4852.40 | 4874.47 | 4868.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 4848.10 | 4874.47 | 4868.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 4866.40 | 4872.85 | 4868.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:15:00 | 4877.90 | 4872.85 | 4868.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:00:00 | 4885.30 | 4901.64 | 4895.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 4942.50 | 4994.91 | 4996.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 4942.50 | 4994.91 | 4996.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 4918.00 | 4971.26 | 4985.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 4892.00 | 4884.87 | 4915.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 4892.00 | 4884.87 | 4915.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 4902.10 | 4888.06 | 4909.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 4902.10 | 4888.06 | 4909.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 4923.30 | 4894.96 | 4905.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 4923.30 | 4894.96 | 4905.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 4908.00 | 4897.57 | 4906.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:15:00 | 4897.50 | 4897.57 | 4906.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 4900.00 | 4898.05 | 4905.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:30:00 | 4887.30 | 4898.98 | 4904.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 4881.50 | 4898.50 | 4903.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 4642.94 | 4695.89 | 4758.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 4637.43 | 4695.89 | 4758.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 10:15:00 | 4699.60 | 4696.63 | 4752.85 | SL hit (close>ema200) qty=0.50 sl=4696.63 alert=retest2 |

### Cycle 144 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 4761.90 | 4758.67 | 4758.52 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 4691.00 | 4745.00 | 4752.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 4673.50 | 4697.29 | 4716.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 4487.90 | 4481.26 | 4538.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:45:00 | 4478.50 | 4481.26 | 4538.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 4514.60 | 4497.53 | 4528.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 4503.80 | 4497.53 | 4528.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 4505.30 | 4526.15 | 4531.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 4569.10 | 4542.03 | 4538.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 4569.10 | 4542.03 | 4538.34 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 4481.50 | 4529.26 | 4533.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 4436.90 | 4486.59 | 4508.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 4523.10 | 4485.00 | 4503.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 4523.10 | 4485.00 | 4503.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 4523.10 | 4485.00 | 4503.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 4523.10 | 4485.00 | 4503.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 4499.40 | 4487.88 | 4503.41 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 4535.50 | 4513.81 | 4512.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 13:15:00 | 4563.00 | 4530.90 | 4522.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 4522.10 | 4543.26 | 4531.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 4522.10 | 4543.26 | 4531.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 4522.10 | 4543.26 | 4531.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 4522.10 | 4543.26 | 4531.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 4532.00 | 4541.01 | 4531.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 13:15:00 | 4561.00 | 4541.56 | 4533.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 12:15:00 | 4495.50 | 4526.03 | 4529.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 4495.50 | 4526.03 | 4529.50 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 4549.00 | 4531.15 | 4530.80 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 4490.90 | 4523.10 | 4527.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 10:15:00 | 4476.90 | 4513.86 | 4522.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 4479.30 | 4472.98 | 4494.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:15:00 | 4511.00 | 4472.98 | 4494.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 4526.60 | 4483.70 | 4497.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 4529.20 | 4483.70 | 4497.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 4483.40 | 4483.64 | 4496.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 4473.10 | 4483.64 | 4496.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 4433.00 | 4386.02 | 4427.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 4495.20 | 4449.00 | 4447.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 4495.20 | 4449.00 | 4447.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 4542.40 | 4475.97 | 4460.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 4492.00 | 4493.57 | 4473.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 09:45:00 | 4493.10 | 4493.57 | 4473.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 4494.40 | 4492.36 | 4476.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 4510.70 | 4504.87 | 4483.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 4454.00 | 4513.01 | 4510.35 | SL hit (close<static) qty=1.00 sl=4472.40 alert=retest2 |

### Cycle 153 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 4464.10 | 4503.23 | 4506.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 13:15:00 | 4436.10 | 4475.58 | 4491.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 4532.70 | 4480.84 | 4489.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 4532.70 | 4480.84 | 4489.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 4532.70 | 4480.84 | 4489.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 4511.00 | 4488.48 | 4492.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 4521.70 | 4490.78 | 4489.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 4521.70 | 4490.78 | 4489.70 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 4480.00 | 4489.78 | 4491.08 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 4501.40 | 4493.42 | 4492.59 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 4480.00 | 4491.35 | 4491.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 4457.10 | 4484.50 | 4488.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 4417.30 | 4407.14 | 4434.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:45:00 | 4422.90 | 4407.14 | 4434.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 4392.10 | 4379.88 | 4401.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 4380.60 | 4379.88 | 4401.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 4424.50 | 4395.44 | 4393.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 4424.50 | 4395.44 | 4393.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 4435.00 | 4403.35 | 4396.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 4441.60 | 4449.75 | 4434.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 4441.60 | 4449.75 | 4434.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 4443.00 | 4448.40 | 4435.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 4443.00 | 4448.40 | 4435.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 4449.00 | 4448.52 | 4436.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 4441.60 | 4448.52 | 4436.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 4451.40 | 4449.39 | 4440.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 4438.00 | 4449.39 | 4440.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 4447.00 | 4451.11 | 4443.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 4446.30 | 4451.11 | 4443.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 4431.90 | 4447.27 | 4442.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 4431.90 | 4447.27 | 4442.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 4431.80 | 4444.18 | 4441.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 4455.60 | 4444.18 | 4441.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 4433.90 | 4440.57 | 4440.11 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 4396.60 | 4431.78 | 4436.15 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 4450.80 | 4435.65 | 4434.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 4464.10 | 4446.07 | 4440.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 4848.50 | 4872.00 | 4831.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:00:00 | 4848.50 | 4872.00 | 4831.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 4867.00 | 4891.70 | 4873.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 4867.00 | 4891.70 | 4873.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 4855.00 | 4884.36 | 4871.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 4855.00 | 4884.36 | 4871.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 4864.80 | 4880.45 | 4870.78 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 4811.00 | 4857.08 | 4861.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 4795.00 | 4837.45 | 4851.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 4770.70 | 4754.21 | 4781.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 4770.70 | 4754.21 | 4781.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 4770.70 | 4754.21 | 4781.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 4766.70 | 4754.21 | 4781.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 4792.00 | 4761.77 | 4782.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 4792.00 | 4761.77 | 4782.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 4776.00 | 4764.62 | 4782.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 4767.70 | 4767.23 | 4780.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:00:00 | 4766.00 | 4766.98 | 4779.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 4753.70 | 4781.63 | 4782.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 4815.00 | 4771.81 | 4775.84 | SL hit (close>static) qty=1.00 sl=4805.10 alert=retest2 |

### Cycle 162 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 4789.00 | 4765.25 | 4763.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 4844.20 | 4799.65 | 4782.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 4842.00 | 4850.93 | 4825.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 4842.10 | 4850.93 | 4825.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 4826.90 | 4841.90 | 4833.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 4826.90 | 4841.90 | 4833.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 4815.70 | 4836.66 | 4831.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 4816.00 | 4836.66 | 4831.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 4835.70 | 4836.47 | 4831.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 4843.90 | 4836.47 | 4831.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:45:00 | 4841.10 | 4838.34 | 4833.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 4845.00 | 4838.34 | 4833.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 4810.00 | 4833.74 | 4832.46 | SL hit (close<static) qty=1.00 sl=4812.40 alert=retest2 |

### Cycle 163 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 4772.90 | 4821.57 | 4827.05 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 4852.80 | 4819.65 | 4816.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 4879.90 | 4831.70 | 4822.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 4831.00 | 4848.56 | 4836.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 4831.00 | 4848.56 | 4836.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 4831.00 | 4848.56 | 4836.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 4831.00 | 4848.56 | 4836.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 4832.00 | 4845.25 | 4836.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 4769.90 | 4845.25 | 4836.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 4759.30 | 4816.82 | 4824.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 4734.30 | 4800.31 | 4815.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 4766.20 | 4765.65 | 4789.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 4766.20 | 4765.65 | 4789.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 4766.20 | 4765.65 | 4789.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 4739.20 | 4762.40 | 4786.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 13:00:00 | 4739.90 | 4755.99 | 4778.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 4742.00 | 4753.50 | 4773.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 4794.00 | 4762.97 | 4773.01 | SL hit (close>static) qty=1.00 sl=4793.90 alert=retest2 |

### Cycle 166 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 4867.80 | 4792.54 | 4785.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 4972.00 | 4872.58 | 4839.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 4878.00 | 4898.34 | 4868.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 15:00:00 | 4878.00 | 4898.34 | 4868.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 4873.80 | 4893.43 | 4869.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 4892.10 | 4893.43 | 4869.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:30:00 | 4885.00 | 4889.98 | 4871.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 4855.60 | 4877.54 | 4870.08 | SL hit (close<static) qty=1.00 sl=4865.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 4852.00 | 4866.47 | 4866.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 12:15:00 | 4847.20 | 4861.29 | 4864.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 11:15:00 | 4839.20 | 4835.66 | 4847.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 11:15:00 | 4839.20 | 4835.66 | 4847.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 4839.20 | 4835.66 | 4847.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:45:00 | 4845.20 | 4835.66 | 4847.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 4682.00 | 4668.07 | 4694.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 4679.30 | 4668.07 | 4694.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 4680.00 | 4670.45 | 4693.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 4677.00 | 4670.45 | 4693.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 4689.40 | 4672.91 | 4690.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 4689.40 | 4672.91 | 4690.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 4697.20 | 4677.76 | 4690.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:30:00 | 4665.40 | 4675.81 | 4685.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:30:00 | 4667.50 | 4673.70 | 4682.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 4693.00 | 4687.69 | 4687.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 4693.00 | 4687.69 | 4687.27 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 4654.00 | 4682.73 | 4685.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 4629.30 | 4672.04 | 4680.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 4597.10 | 4595.01 | 4623.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:45:00 | 4593.00 | 4595.01 | 4623.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 4622.30 | 4600.47 | 4623.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:45:00 | 4623.30 | 4600.47 | 4623.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 4625.00 | 4605.37 | 4623.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 4689.00 | 4605.37 | 4623.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 4750.00 | 4634.30 | 4635.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 4750.00 | 4634.30 | 4635.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 4772.00 | 4661.84 | 4647.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 4780.00 | 4702.30 | 4669.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 14:15:00 | 4751.50 | 4854.40 | 4817.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 14:15:00 | 4751.50 | 4854.40 | 4817.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 4751.50 | 4854.40 | 4817.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 4734.00 | 4854.40 | 4817.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 4730.10 | 4829.54 | 4809.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 4732.30 | 4829.54 | 4809.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 4742.00 | 4795.50 | 4796.78 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 4814.00 | 4770.52 | 4764.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 09:15:00 | 4829.80 | 4782.37 | 4770.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 4775.60 | 4802.33 | 4790.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 4775.60 | 4802.33 | 4790.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 4775.60 | 4802.33 | 4790.61 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 4759.10 | 4781.05 | 4782.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 4740.40 | 4763.92 | 4773.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 4470.00 | 4455.68 | 4504.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:30:00 | 4472.00 | 4455.68 | 4504.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 4507.60 | 4468.20 | 4502.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 4507.60 | 4468.20 | 4502.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 4501.20 | 4474.80 | 4502.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:30:00 | 4514.80 | 4474.80 | 4502.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 4504.00 | 4480.64 | 4502.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:45:00 | 4505.60 | 4480.64 | 4502.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 4517.40 | 4487.99 | 4503.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 4517.40 | 4487.99 | 4503.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 4512.00 | 4492.79 | 4504.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 4508.00 | 4492.79 | 4504.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 4523.00 | 4492.42 | 4496.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 4523.00 | 4492.42 | 4496.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 4527.80 | 4499.50 | 4499.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 4540.10 | 4518.21 | 4509.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 4522.00 | 4522.46 | 4512.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:00:00 | 4522.00 | 4522.46 | 4512.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 4505.50 | 4519.07 | 4512.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 4505.50 | 4519.07 | 4512.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 4515.70 | 4518.39 | 4512.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 4525.20 | 4518.39 | 4512.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 11:15:00 | 4498.70 | 4516.93 | 4515.93 | SL hit (close<static) qty=1.00 sl=4504.60 alert=retest2 |

### Cycle 175 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 4493.00 | 4512.14 | 4513.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 4476.90 | 4502.63 | 4508.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 4462.60 | 4456.96 | 4477.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 4456.00 | 4456.96 | 4477.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 4501.80 | 4467.86 | 4478.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 4501.80 | 4467.86 | 4478.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 4490.30 | 4472.34 | 4479.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 4496.00 | 4472.34 | 4479.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 4491.00 | 4477.83 | 4480.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 4491.00 | 4477.83 | 4480.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 15:15:00 | 4522.00 | 4486.66 | 4484.72 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 4466.40 | 4482.61 | 4483.06 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 4489.10 | 4483.91 | 4483.61 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 4466.00 | 4480.33 | 4482.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 13:15:00 | 4464.00 | 4476.62 | 4480.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 4309.30 | 4297.72 | 4346.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 4309.30 | 4297.72 | 4346.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 4303.20 | 4294.33 | 4315.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 4296.00 | 4294.33 | 4315.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 4320.20 | 4300.04 | 4314.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 4320.20 | 4300.04 | 4314.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 4328.20 | 4305.67 | 4315.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:30:00 | 4331.50 | 4305.67 | 4315.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 4314.30 | 4311.24 | 4316.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 4347.90 | 4311.24 | 4316.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 4352.30 | 4319.45 | 4319.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 4346.00 | 4319.45 | 4319.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 4329.60 | 4321.48 | 4320.39 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 4282.60 | 4313.98 | 4317.18 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 4335.00 | 4319.12 | 4317.02 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 4257.40 | 4308.31 | 4312.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 4249.00 | 4285.75 | 4300.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 4252.40 | 4237.30 | 4255.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 4252.40 | 4237.30 | 4255.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 4252.40 | 4237.30 | 4255.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 4253.10 | 4237.30 | 4255.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 4256.90 | 4241.22 | 4255.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 4265.70 | 4241.22 | 4255.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 4235.70 | 4240.12 | 4253.53 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 4288.10 | 4263.69 | 4261.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 4296.30 | 4274.45 | 4266.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 4422.00 | 4428.01 | 4399.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:30:00 | 4417.50 | 4428.01 | 4399.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 4413.20 | 4432.96 | 4417.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 4422.60 | 4432.96 | 4417.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 4412.90 | 4428.95 | 4417.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 4439.10 | 4428.95 | 4417.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 4394.10 | 4423.97 | 4419.94 | SL hit (close<static) qty=1.00 sl=4401.10 alert=retest2 |

### Cycle 185 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 4376.50 | 4414.47 | 4415.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 4366.70 | 4398.44 | 4408.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 4373.80 | 4350.28 | 4372.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 4373.80 | 4350.28 | 4372.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 4373.80 | 4350.28 | 4372.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 4360.70 | 4350.28 | 4372.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 4391.30 | 4358.49 | 4374.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 4388.50 | 4358.49 | 4374.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 4391.70 | 4365.13 | 4375.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 4391.60 | 4365.13 | 4375.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 4380.00 | 4375.07 | 4377.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 4380.00 | 4375.07 | 4377.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 4384.60 | 4376.98 | 4378.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 4384.60 | 4376.98 | 4378.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 4392.60 | 4381.09 | 4380.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 4400.00 | 4384.87 | 4381.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 4388.90 | 4395.39 | 4389.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 13:15:00 | 4388.90 | 4395.39 | 4389.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 4388.90 | 4395.39 | 4389.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 4388.90 | 4395.39 | 4389.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 4418.30 | 4399.97 | 4392.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:30:00 | 4391.70 | 4399.97 | 4392.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 4515.00 | 4504.15 | 4479.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 4532.50 | 4513.29 | 4496.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 12:00:00 | 4527.00 | 4527.65 | 4509.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 4469.90 | 4503.95 | 4503.27 | SL hit (close<static) qty=1.00 sl=4478.60 alert=retest2 |

### Cycle 187 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 4494.00 | 4501.96 | 4502.43 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 4515.50 | 4504.67 | 4503.62 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 4489.90 | 4500.99 | 4502.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 4463.50 | 4493.49 | 4498.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 4484.90 | 4476.69 | 4485.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 4484.90 | 4476.69 | 4485.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 4484.90 | 4476.69 | 4485.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:45:00 | 4490.10 | 4476.69 | 4485.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 4503.60 | 4482.07 | 4487.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 4503.60 | 4482.07 | 4487.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 4515.20 | 4488.70 | 4489.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 4524.70 | 4488.70 | 4489.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 4525.20 | 4496.00 | 4493.10 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 4450.70 | 4484.19 | 4488.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 4434.60 | 4474.27 | 4483.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 4463.10 | 4459.85 | 4472.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 4463.10 | 4459.85 | 4472.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 4463.10 | 4459.85 | 4472.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 4453.80 | 4459.85 | 4472.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 4482.60 | 4464.40 | 4473.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 4484.80 | 4464.40 | 4473.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 4481.30 | 4467.78 | 4473.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 4494.00 | 4467.78 | 4473.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 4467.40 | 4471.96 | 4474.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 4488.50 | 4471.96 | 4474.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 4430.10 | 4462.15 | 4469.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 4408.20 | 4439.30 | 4456.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 12:15:00 | 4483.10 | 4454.95 | 4454.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 4483.10 | 4454.95 | 4454.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 14:15:00 | 4499.00 | 4468.81 | 4460.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 4448.70 | 4470.10 | 4463.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 4448.70 | 4470.10 | 4463.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 4448.70 | 4470.10 | 4463.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 4439.00 | 4470.10 | 4463.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 4445.70 | 4465.22 | 4461.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 4451.00 | 4465.22 | 4461.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 4431.40 | 4458.45 | 4458.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 4402.40 | 4447.24 | 4453.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 4343.00 | 4309.53 | 4355.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 4343.00 | 4309.53 | 4355.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 4351.00 | 4322.87 | 4344.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 4351.00 | 4322.87 | 4344.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 4366.80 | 4331.65 | 4346.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 4354.00 | 4331.65 | 4346.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 4342.20 | 4336.56 | 4346.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 4347.00 | 4336.56 | 4346.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 4325.30 | 4334.30 | 4344.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 4306.50 | 4331.90 | 4342.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 14:30:00 | 4307.10 | 4323.17 | 4336.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 15:00:00 | 4301.70 | 4323.17 | 4336.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:30:00 | 4303.90 | 4318.27 | 4330.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 4342.10 | 4317.72 | 4326.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 4342.10 | 4317.72 | 4326.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 4367.70 | 4327.72 | 4329.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 4367.70 | 4327.72 | 4329.93 | SL hit (close>static) qty=1.00 sl=4347.00 alert=retest2 |

### Cycle 194 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 4447.40 | 4351.66 | 4340.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 4490.30 | 4379.38 | 4354.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 4578.60 | 4591.84 | 4540.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 4578.60 | 4591.84 | 4540.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 4485.40 | 4600.90 | 4571.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 4485.40 | 4600.90 | 4571.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 4506.60 | 4582.04 | 4565.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 4421.00 | 4582.04 | 4565.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 4425.10 | 4550.65 | 4552.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 4375.00 | 4515.52 | 4536.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 4348.90 | 4334.30 | 4413.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 4348.90 | 4334.30 | 4413.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 4413.30 | 4352.31 | 4408.29 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 4469.70 | 4436.56 | 4434.50 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 4218.50 | 4397.82 | 4417.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 3965.70 | 4204.16 | 4297.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 4068.00 | 4053.15 | 4122.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 4068.00 | 4053.15 | 4122.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 4112.00 | 4072.99 | 4110.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 4112.00 | 4072.99 | 4110.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 4125.20 | 4083.43 | 4111.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 4125.20 | 4083.43 | 4111.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 4128.10 | 4092.37 | 4112.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:15:00 | 4133.10 | 4092.37 | 4112.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 4151.00 | 4108.51 | 4117.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 4175.00 | 4108.51 | 4117.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 4162.20 | 4129.63 | 4125.78 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 4114.20 | 4130.15 | 4131.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 4101.10 | 4123.08 | 4127.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 4171.60 | 4123.25 | 4125.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 13:15:00 | 4171.60 | 4123.25 | 4125.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 4171.60 | 4123.25 | 4125.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:00:00 | 4171.60 | 4123.25 | 4125.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 4148.10 | 4128.22 | 4127.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 09:15:00 | 4196.00 | 4149.87 | 4137.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 15:15:00 | 4204.50 | 4206.59 | 4178.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 09:15:00 | 4202.10 | 4206.59 | 4178.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 4190.60 | 4203.39 | 4179.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:45:00 | 4192.80 | 4203.39 | 4179.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 4186.90 | 4200.09 | 4179.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 4186.90 | 4200.09 | 4179.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 4189.90 | 4198.05 | 4180.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:45:00 | 4186.60 | 4198.05 | 4180.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 4197.20 | 4232.31 | 4218.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:00:00 | 4243.20 | 4230.35 | 4220.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 4177.00 | 4222.29 | 4222.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 4177.00 | 4222.29 | 4222.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 4172.00 | 4212.23 | 4218.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 4200.00 | 4180.89 | 4197.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 4200.00 | 4180.89 | 4197.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 4200.00 | 4180.89 | 4197.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 4200.00 | 4180.89 | 4197.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 4207.00 | 4186.11 | 4198.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 4207.00 | 4186.11 | 4198.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 4173.40 | 4183.57 | 4195.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 4121.00 | 4178.52 | 4190.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 14:15:00 | 3914.95 | 3956.86 | 3980.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 3949.00 | 3944.65 | 3967.88 | SL hit (close>ema200) qty=0.50 sl=3944.65 alert=retest2 |

### Cycle 202 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 3922.80 | 3917.74 | 3917.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 3951.10 | 3924.41 | 3920.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 13:15:00 | 3972.20 | 3975.66 | 3956.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 14:00:00 | 3972.20 | 3975.66 | 3956.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3943.80 | 3973.01 | 3960.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 3919.70 | 3973.01 | 3960.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 3965.80 | 3971.57 | 3960.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:30:00 | 3976.70 | 3971.25 | 3961.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 3974.50 | 3971.25 | 3961.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 09:45:00 | 3974.60 | 3996.86 | 3992.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 3974.80 | 3996.86 | 3992.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 3970.20 | 3991.53 | 3990.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 4005.40 | 3996.75 | 3992.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 3898.50 | 3985.66 | 3990.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 3898.50 | 3985.66 | 3990.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 3892.30 | 3923.49 | 3950.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 3910.00 | 3909.01 | 3933.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 3910.00 | 3909.01 | 3933.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 3910.00 | 3909.01 | 3933.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 3926.50 | 3909.01 | 3933.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3930.60 | 3913.01 | 3931.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3930.60 | 3913.01 | 3931.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3939.80 | 3918.37 | 3931.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 3951.10 | 3918.37 | 3931.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 3927.20 | 3920.13 | 3931.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 3914.10 | 3920.13 | 3931.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 3964.70 | 3932.27 | 3935.21 | SL hit (close>static) qty=1.00 sl=3941.50 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 3965.00 | 3938.82 | 3937.92 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 3885.00 | 3939.79 | 3941.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 3858.70 | 3907.29 | 3925.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 3868.00 | 3864.65 | 3895.88 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:45:00 | 3820.10 | 3852.31 | 3884.69 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 3629.09 | 3780.97 | 3836.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 3687.90 | 3662.84 | 3718.10 | SL hit (close>ema200) qty=0.50 sl=3662.84 alert=retest1 |

### Cycle 206 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 3660.00 | 3614.37 | 3609.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 3681.10 | 3627.72 | 3616.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3594.60 | 3621.09 | 3614.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3594.60 | 3621.09 | 3614.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3594.60 | 3621.09 | 3614.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 3594.60 | 3621.09 | 3614.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 3610.20 | 3618.92 | 3613.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:30:00 | 3594.00 | 3618.92 | 3613.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 3632.40 | 3621.61 | 3615.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 3642.00 | 3621.61 | 3615.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 3580.00 | 3613.29 | 3612.24 | SL hit (close<static) qty=1.00 sl=3605.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 4350.00 | 4361.87 | 4362.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 4319.40 | 4353.37 | 4358.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 4305.00 | 4289.59 | 4315.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 4305.00 | 4289.59 | 4315.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 4305.00 | 4289.59 | 4315.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 4339.00 | 4289.59 | 4315.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 4306.20 | 4292.91 | 4314.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 4317.30 | 4292.91 | 4314.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 4313.50 | 4298.84 | 4313.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 4307.90 | 4298.84 | 4313.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 4308.70 | 4300.81 | 4313.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 4305.00 | 4302.67 | 4312.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 4322.90 | 4307.89 | 4313.46 | SL hit (close>static) qty=1.00 sl=4318.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 4339.00 | 4316.37 | 4315.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 4533.30 | 4375.02 | 4350.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 13:15:00 | 4620.40 | 4623.14 | 4570.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 14:00:00 | 4620.40 | 4623.14 | 4570.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-26 10:45:00 | 1509.00 | 2023-05-30 09:15:00 | 1530.97 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2023-05-26 11:15:00 | 1510.00 | 2023-05-30 09:15:00 | 1530.97 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-05-26 11:45:00 | 1509.88 | 2023-05-30 09:15:00 | 1530.97 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-05-26 13:45:00 | 1508.35 | 2023-05-30 09:15:00 | 1530.97 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2023-05-29 13:30:00 | 1503.50 | 2023-05-30 09:15:00 | 1530.97 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2023-05-31 15:00:00 | 1560.35 | 2023-06-06 15:15:00 | 1716.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-06-27 11:15:00 | 1854.95 | 2023-06-28 15:15:00 | 1864.97 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2023-06-28 09:45:00 | 1848.60 | 2023-06-28 15:15:00 | 1864.97 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-07-06 11:30:00 | 1862.97 | 2023-07-07 09:15:00 | 1881.30 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-08-08 11:15:00 | 1878.00 | 2023-08-09 10:15:00 | 1901.95 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2023-08-08 12:00:00 | 1879.00 | 2023-08-09 10:15:00 | 1901.95 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-08-11 09:15:00 | 1900.43 | 2023-08-11 14:15:00 | 1885.55 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-08-16 15:00:00 | 1942.97 | 2023-08-18 09:15:00 | 1918.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-08-17 14:30:00 | 1939.50 | 2023-08-18 09:15:00 | 1918.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-08-29 09:30:00 | 1949.78 | 2023-08-30 12:15:00 | 1965.47 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2023-08-29 14:45:00 | 1948.65 | 2023-08-30 12:15:00 | 1965.47 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-09-06 09:15:00 | 1986.72 | 2023-09-12 10:15:00 | 1995.00 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2023-09-14 11:15:00 | 1976.60 | 2023-09-18 09:15:00 | 2005.53 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2023-09-14 13:45:00 | 1977.10 | 2023-09-18 09:15:00 | 2005.53 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-09-15 09:30:00 | 1977.00 | 2023-09-18 09:15:00 | 2005.53 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-09-15 15:00:00 | 1973.03 | 2023-09-18 09:15:00 | 2005.53 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-09-21 09:15:00 | 4015.00 | 2023-09-22 09:15:00 | 1949.62 | STOP_HIT | 1.00 | -51.44% |
| BUY | retest2 | 2023-10-13 11:15:00 | 1978.00 | 2023-10-16 14:15:00 | 1964.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-10-13 12:15:00 | 1978.30 | 2023-10-16 14:15:00 | 1964.10 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-10-13 13:00:00 | 1977.55 | 2023-10-16 14:15:00 | 1964.10 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2023-11-10 09:15:00 | 2073.90 | 2023-11-10 13:15:00 | 2032.45 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-11-12 18:30:00 | 2068.95 | 2023-11-22 11:15:00 | 2103.00 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2023-11-13 11:15:00 | 2065.80 | 2023-11-22 11:15:00 | 2103.00 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest2 | 2023-11-13 13:30:00 | 2071.00 | 2023-11-22 11:15:00 | 2103.00 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2023-11-15 10:15:00 | 2065.25 | 2023-11-22 11:15:00 | 2103.00 | STOP_HIT | 1.00 | 1.83% |
| BUY | retest1 | 2023-12-08 09:15:00 | 2746.50 | 2023-12-12 13:15:00 | 2735.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-12-08 10:00:00 | 2742.75 | 2023-12-12 13:15:00 | 2735.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-12-11 09:30:00 | 2773.70 | 2023-12-12 13:15:00 | 2735.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2023-12-13 13:45:00 | 2771.15 | 2023-12-15 11:15:00 | 2745.35 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-12-13 15:15:00 | 2777.00 | 2023-12-15 11:15:00 | 2745.35 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-12-14 15:15:00 | 2771.00 | 2023-12-15 11:15:00 | 2745.35 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-12-19 11:45:00 | 2801.25 | 2023-12-20 13:15:00 | 2702.45 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2023-12-19 13:30:00 | 2802.80 | 2023-12-20 13:15:00 | 2702.45 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2023-12-20 09:30:00 | 2817.20 | 2023-12-20 13:15:00 | 2702.45 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2023-12-20 11:45:00 | 2798.00 | 2023-12-20 13:15:00 | 2702.45 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2024-01-01 09:15:00 | 2844.50 | 2024-01-02 10:15:00 | 2773.95 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-01-02 10:00:00 | 2814.85 | 2024-01-02 10:15:00 | 2773.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-01-11 09:15:00 | 3032.00 | 2024-01-12 12:15:00 | 2999.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-01-11 14:15:00 | 3023.25 | 2024-01-12 12:15:00 | 2999.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-01-11 15:15:00 | 3022.00 | 2024-01-12 12:15:00 | 2999.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-01-25 10:15:00 | 2909.60 | 2024-01-29 12:15:00 | 2973.65 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-01-31 12:30:00 | 3001.15 | 2024-02-02 13:15:00 | 2988.65 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-01-31 13:45:00 | 3005.40 | 2024-02-02 14:15:00 | 2971.45 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-01-31 15:15:00 | 3009.00 | 2024-02-02 14:15:00 | 2971.45 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-02-01 14:45:00 | 3003.70 | 2024-02-02 14:15:00 | 2971.45 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-02-02 09:15:00 | 3017.20 | 2024-02-02 14:15:00 | 2971.45 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-02-07 10:30:00 | 2932.65 | 2024-02-08 09:15:00 | 3025.10 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2024-02-07 11:15:00 | 2930.60 | 2024-02-08 09:15:00 | 3025.10 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2024-02-14 09:15:00 | 2885.90 | 2024-02-14 14:15:00 | 2951.65 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-02-28 09:15:00 | 3155.00 | 2024-02-29 10:15:00 | 3037.05 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2024-02-28 11:00:00 | 3127.80 | 2024-02-29 10:15:00 | 3037.05 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-02-28 12:00:00 | 3123.60 | 2024-02-29 10:15:00 | 3037.05 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-03-06 14:15:00 | 3216.65 | 2024-03-12 15:15:00 | 3274.90 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2024-03-15 10:15:00 | 3056.30 | 2024-03-20 10:15:00 | 2919.35 | PARTIAL | 0.50 | 4.48% |
| SELL | retest2 | 2024-03-15 10:15:00 | 3056.30 | 2024-03-21 09:15:00 | 3069.15 | STOP_HIT | 0.50 | -0.42% |
| SELL | retest2 | 2024-03-15 13:45:00 | 3073.00 | 2024-03-21 13:15:00 | 3103.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-03-19 10:45:00 | 3060.50 | 2024-03-21 13:15:00 | 3103.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-04-09 09:45:00 | 3579.85 | 2024-04-09 13:15:00 | 3534.65 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-04-09 12:45:00 | 3573.85 | 2024-04-09 13:15:00 | 3534.65 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-04-10 09:30:00 | 3612.35 | 2024-04-24 10:15:00 | 3940.20 | TARGET_HIT | 1.00 | 9.08% |
| BUY | retest2 | 2024-04-12 09:15:00 | 3582.00 | 2024-04-24 11:15:00 | 3973.59 | TARGET_HIT | 1.00 | 10.93% |
| BUY | retest2 | 2024-04-12 12:15:00 | 3641.80 | 2024-04-25 09:15:00 | 4005.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-13 12:15:00 | 4904.00 | 2024-06-18 09:15:00 | 5394.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-28 10:15:00 | 5273.75 | 2024-07-01 14:15:00 | 5399.20 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-06-28 12:15:00 | 5261.00 | 2024-07-01 14:15:00 | 5399.20 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-06-28 14:30:00 | 5271.00 | 2024-07-01 14:15:00 | 5399.20 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-07-01 10:00:00 | 5271.80 | 2024-07-01 14:15:00 | 5399.20 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest1 | 2024-07-08 09:15:00 | 5618.45 | 2024-07-09 10:15:00 | 5539.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-07-09 12:45:00 | 5554.70 | 2024-07-10 09:15:00 | 5470.95 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-07-09 13:30:00 | 5559.90 | 2024-07-10 09:15:00 | 5470.95 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-07-09 14:30:00 | 5548.00 | 2024-07-10 09:15:00 | 5470.95 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-07-10 09:15:00 | 5559.95 | 2024-07-10 09:15:00 | 5470.95 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-07-11 10:15:00 | 5477.50 | 2024-07-11 13:15:00 | 5554.05 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-07-23 12:15:00 | 4712.10 | 2024-07-26 15:15:00 | 4908.00 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2024-08-06 13:30:00 | 4585.20 | 2024-08-07 14:15:00 | 4746.40 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2024-08-07 10:00:00 | 4590.00 | 2024-08-07 14:15:00 | 4746.40 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2024-08-30 12:15:00 | 4662.55 | 2024-08-30 15:15:00 | 4688.45 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-09-11 09:15:00 | 4668.25 | 2024-09-17 11:15:00 | 4434.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 10:00:00 | 4666.30 | 2024-09-17 11:15:00 | 4435.07 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2024-09-13 09:45:00 | 4668.50 | 2024-09-17 12:15:00 | 4432.98 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2024-09-11 09:15:00 | 4668.25 | 2024-09-19 11:15:00 | 4201.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-11 10:00:00 | 4666.30 | 2024-09-19 11:15:00 | 4199.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-13 09:45:00 | 4668.50 | 2024-09-19 11:15:00 | 4201.65 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-09-25 09:15:00 | 4428.00 | 2024-09-26 09:15:00 | 4371.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-10-01 09:15:00 | 4429.75 | 2024-10-03 09:15:00 | 4399.40 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-10-01 13:00:00 | 4427.50 | 2024-10-03 09:15:00 | 4399.40 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-10-07 10:15:00 | 4194.90 | 2024-10-08 10:15:00 | 4326.00 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-10-07 11:30:00 | 4194.40 | 2024-10-08 10:15:00 | 4326.00 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2024-10-17 12:30:00 | 4556.10 | 2024-10-17 14:15:00 | 4519.40 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-10-21 15:15:00 | 4496.95 | 2024-10-23 09:15:00 | 4272.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:15:00 | 4496.95 | 2024-10-25 15:15:00 | 4182.00 | STOP_HIT | 0.50 | 7.00% |
| BUY | retest2 | 2024-10-31 11:45:00 | 4265.80 | 2024-11-04 13:15:00 | 4212.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-10-31 12:15:00 | 4261.85 | 2024-11-04 13:15:00 | 4212.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-11-01 18:00:00 | 4276.00 | 2024-11-04 13:15:00 | 4212.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-11-04 09:30:00 | 4268.90 | 2024-11-04 13:15:00 | 4212.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-11-11 10:15:00 | 4441.30 | 2024-11-12 11:15:00 | 4326.85 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-11-14 14:15:00 | 4087.05 | 2024-11-18 09:15:00 | 4192.90 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-11-18 14:15:00 | 4079.25 | 2024-11-22 14:15:00 | 4113.60 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-11-19 14:15:00 | 4092.25 | 2024-11-22 14:15:00 | 4113.60 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-11-22 13:15:00 | 4088.05 | 2024-11-22 14:15:00 | 4113.60 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-12-03 09:15:00 | 4532.95 | 2024-12-17 13:15:00 | 4631.30 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2024-12-04 09:15:00 | 4580.50 | 2024-12-17 13:15:00 | 4631.30 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2024-12-05 09:15:00 | 4559.70 | 2024-12-17 13:15:00 | 4631.30 | STOP_HIT | 1.00 | 1.57% |
| SELL | retest2 | 2025-01-01 09:15:00 | 4167.10 | 2025-01-02 09:15:00 | 4218.70 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-01-01 14:45:00 | 4170.80 | 2025-01-02 09:15:00 | 4218.70 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-01-09 09:15:00 | 4069.90 | 2025-01-13 11:15:00 | 3891.20 | PARTIAL | 0.50 | 4.39% |
| SELL | retest2 | 2025-01-09 10:15:00 | 4096.00 | 2025-01-13 11:15:00 | 3892.48 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-01-09 14:30:00 | 4097.35 | 2025-01-13 12:15:00 | 3866.40 | PARTIAL | 0.50 | 5.64% |
| SELL | retest2 | 2025-01-09 09:15:00 | 4069.90 | 2025-01-14 10:15:00 | 3864.85 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2025-01-09 10:15:00 | 4096.00 | 2025-01-14 10:15:00 | 3864.85 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest2 | 2025-01-09 14:30:00 | 4097.35 | 2025-01-14 10:15:00 | 3864.85 | STOP_HIT | 0.50 | 5.67% |
| SELL | retest2 | 2025-01-24 14:30:00 | 3844.65 | 2025-01-28 09:15:00 | 3652.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:30:00 | 3844.65 | 2025-01-28 12:15:00 | 3650.55 | STOP_HIT | 0.50 | 5.05% |
| BUY | retest2 | 2025-02-07 09:15:00 | 3813.80 | 2025-02-11 09:15:00 | 3687.70 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2025-02-07 10:00:00 | 3786.15 | 2025-02-11 09:15:00 | 3687.70 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-02-07 10:30:00 | 3793.95 | 2025-02-11 09:15:00 | 3687.70 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-02-10 09:45:00 | 3796.85 | 2025-02-11 09:15:00 | 3687.70 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-02-10 11:30:00 | 3810.25 | 2025-02-11 09:15:00 | 3687.70 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-02-14 10:15:00 | 3574.55 | 2025-02-17 09:15:00 | 3395.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:15:00 | 3574.55 | 2025-02-19 09:15:00 | 3392.55 | STOP_HIT | 0.50 | 5.09% |
| BUY | retest2 | 2025-03-07 10:30:00 | 3481.00 | 2025-03-12 11:15:00 | 3406.25 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-03-07 11:00:00 | 3480.50 | 2025-03-12 11:15:00 | 3406.25 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-03-10 09:15:00 | 3521.55 | 2025-03-12 11:15:00 | 3406.25 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2025-03-26 12:15:00 | 4146.30 | 2025-04-07 09:15:00 | 3896.20 | STOP_HIT | 1.00 | -6.03% |
| BUY | retest2 | 2025-03-27 09:15:00 | 4204.05 | 2025-04-07 09:15:00 | 3896.20 | STOP_HIT | 1.00 | -7.32% |
| BUY | retest1 | 2025-04-17 11:45:00 | 4236.90 | 2025-04-24 14:15:00 | 4296.60 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest1 | 2025-04-17 13:30:00 | 4239.40 | 2025-04-24 14:15:00 | 4296.60 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest1 | 2025-04-21 09:15:00 | 4242.00 | 2025-04-24 14:15:00 | 4296.60 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-04-23 09:15:00 | 4349.00 | 2025-04-25 09:15:00 | 4218.80 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-04-23 10:15:00 | 4300.30 | 2025-04-25 09:15:00 | 4218.80 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-04-23 11:15:00 | 4296.10 | 2025-04-25 09:15:00 | 4218.80 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-04-23 12:00:00 | 4294.80 | 2025-04-25 09:15:00 | 4218.80 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-04-24 13:15:00 | 4309.60 | 2025-04-25 09:15:00 | 4218.80 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-04-24 14:15:00 | 4309.10 | 2025-04-25 09:15:00 | 4218.80 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-05-02 09:15:00 | 4535.90 | 2025-05-07 09:15:00 | 4495.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-05-02 14:30:00 | 4497.40 | 2025-05-07 09:15:00 | 4495.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-05-09 13:00:00 | 4486.50 | 2025-05-13 09:15:00 | 4582.80 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-05-12 09:15:00 | 4465.60 | 2025-05-13 09:15:00 | 4582.80 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-06-12 09:15:00 | 5055.10 | 2025-06-12 11:15:00 | 5041.50 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-06-12 09:45:00 | 5054.70 | 2025-06-12 11:15:00 | 5041.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-27 11:45:00 | 4864.00 | 2025-06-27 14:15:00 | 4896.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-06-27 13:00:00 | 4865.00 | 2025-06-27 14:15:00 | 4896.70 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-06-27 14:15:00 | 4866.00 | 2025-06-27 14:15:00 | 4896.70 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-06-30 15:15:00 | 4877.90 | 2025-07-10 10:15:00 | 4942.50 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-07-02 14:00:00 | 4885.30 | 2025-07-10 10:15:00 | 4942.50 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-07-15 13:30:00 | 4887.30 | 2025-07-21 09:15:00 | 4642.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 09:15:00 | 4881.50 | 2025-07-21 09:15:00 | 4637.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 13:30:00 | 4887.30 | 2025-07-21 10:15:00 | 4699.60 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-07-16 09:15:00 | 4881.50 | 2025-07-21 10:15:00 | 4699.60 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-07-30 10:15:00 | 4503.80 | 2025-07-31 12:15:00 | 4569.10 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-31 10:00:00 | 4505.30 | 2025-07-31 12:15:00 | 4569.10 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-08-06 13:15:00 | 4561.00 | 2025-08-07 12:15:00 | 4495.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-11 12:15:00 | 4473.10 | 2025-08-13 11:15:00 | 4495.20 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-08-12 14:30:00 | 4433.00 | 2025-08-13 11:15:00 | 4495.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-08-14 12:45:00 | 4510.70 | 2025-08-19 09:15:00 | 4454.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-08-20 10:30:00 | 4511.00 | 2025-08-22 09:15:00 | 4521.70 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-08-29 12:15:00 | 4380.60 | 2025-09-01 14:15:00 | 4424.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-25 14:00:00 | 4767.70 | 2025-09-29 09:15:00 | 4815.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-25 15:00:00 | 4766.00 | 2025-09-29 09:15:00 | 4815.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-26 13:15:00 | 4753.70 | 2025-09-29 09:15:00 | 4815.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-29 11:30:00 | 4759.60 | 2025-10-01 09:15:00 | 4797.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-30 14:15:00 | 4728.20 | 2025-10-01 10:15:00 | 4789.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-10-07 13:15:00 | 4843.90 | 2025-10-08 09:15:00 | 4810.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-07 14:45:00 | 4841.10 | 2025-10-08 09:15:00 | 4810.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-07 15:15:00 | 4845.00 | 2025-10-08 09:15:00 | 4810.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-14 11:15:00 | 4739.20 | 2025-10-15 10:15:00 | 4794.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-10-14 13:00:00 | 4739.90 | 2025-10-15 10:15:00 | 4794.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-14 15:15:00 | 4742.00 | 2025-10-15 10:15:00 | 4794.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-20 09:15:00 | 4892.10 | 2025-10-20 13:15:00 | 4855.60 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-20 10:30:00 | 4885.00 | 2025-10-20 13:15:00 | 4855.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-03 11:30:00 | 4665.40 | 2025-11-04 10:15:00 | 4693.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-11-03 13:30:00 | 4667.50 | 2025-11-04 10:15:00 | 4693.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-12-01 12:15:00 | 4525.20 | 2025-12-02 11:15:00 | 4498.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-29 09:15:00 | 4439.10 | 2025-12-29 13:15:00 | 4394.10 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-01-07 14:45:00 | 4532.50 | 2026-01-08 15:15:00 | 4469.90 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-01-08 12:00:00 | 4527.00 | 2026-01-08 15:15:00 | 4469.90 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-01-16 13:00:00 | 4408.20 | 2026-01-19 12:15:00 | 4483.10 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-23 13:15:00 | 4306.50 | 2026-01-27 15:15:00 | 4367.70 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-01-23 14:30:00 | 4307.10 | 2026-01-27 15:15:00 | 4367.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-23 15:00:00 | 4301.70 | 2026-01-27 15:15:00 | 4367.70 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-27 10:30:00 | 4303.90 | 2026-01-27 15:15:00 | 4367.70 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-02-18 13:00:00 | 4243.20 | 2026-02-19 11:15:00 | 4177.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-02-23 09:15:00 | 4121.00 | 2026-02-27 14:15:00 | 3914.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:15:00 | 4121.00 | 2026-03-02 10:15:00 | 3949.00 | STOP_HIT | 0.50 | 4.17% |
| BUY | retest2 | 2026-03-10 11:30:00 | 3976.70 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2026-03-10 12:15:00 | 3974.50 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-03-12 09:45:00 | 3974.60 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-03-12 10:15:00 | 3974.80 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-03-12 11:30:00 | 4005.40 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-03-17 12:15:00 | 3914.10 | 2026-03-17 13:15:00 | 3964.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest1 | 2026-03-20 11:45:00 | 3820.10 | 2026-03-23 09:15:00 | 3629.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-20 11:45:00 | 3820.10 | 2026-03-24 12:15:00 | 3687.90 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2026-03-25 11:15:00 | 3679.50 | 2026-03-30 14:15:00 | 3495.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:15:00 | 3684.00 | 2026-03-30 14:15:00 | 3499.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:45:00 | 3682.70 | 2026-03-30 14:15:00 | 3498.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:15:00 | 3679.50 | 2026-04-01 09:15:00 | 3633.90 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2026-03-25 14:15:00 | 3684.00 | 2026-04-01 09:15:00 | 3633.90 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2026-03-25 14:45:00 | 3682.70 | 2026-04-01 09:15:00 | 3633.90 | STOP_HIT | 0.50 | 1.33% |
| BUY | retest2 | 2026-04-02 12:15:00 | 3642.00 | 2026-04-02 12:15:00 | 3580.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-04-02 14:00:00 | 3643.60 | 2026-04-09 09:15:00 | 4007.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:45:00 | 3658.20 | 2026-04-09 09:15:00 | 4024.02 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 14:30:00 | 4305.00 | 2026-04-28 09:15:00 | 4322.90 | STOP_HIT | 1.00 | -0.42% |
