# Bharti Airtel Ltd. (BHARTIARTL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1834.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 48 |
| ALERT2 | 49 |
| ALERT2_SKIP | 17 |
| ALERT3 | 126 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 54 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 60 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 50
- **Target hits / Stop hits / Partials:** 0 / 58 / 2
- **Avg / median % per leg:** -0.27% / -0.67%
- **Sum % (uncompounded):** -16.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 5 | 19.2% | 0 | 26 | 0 | -0.58% | -15.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.69% | -2.1% |
| BUY @ 3rd Alert (retest2) | 23 | 5 | 21.7% | 0 | 23 | 0 | -0.57% | -13.0% |
| SELL (all) | 34 | 5 | 14.7% | 0 | 32 | 2 | -0.03% | -1.1% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.34% | 1.3% |
| SELL @ 3rd Alert (retest2) | 33 | 4 | 12.1% | 0 | 31 | 2 | -0.07% | -2.5% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.18% | -0.7% |
| retest2 (combined) | 56 | 9 | 16.1% | 0 | 54 | 2 | -0.28% | -15.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 1868.10 | 1845.41 | 1844.04 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 09:15:00 | 1816.70 | 1842.50 | 1843.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1802.70 | 1812.72 | 1819.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1821.50 | 1813.63 | 1818.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1821.50 | 1813.63 | 1818.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1821.50 | 1813.63 | 1818.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1821.50 | 1813.63 | 1818.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1817.90 | 1814.48 | 1818.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 1819.50 | 1814.48 | 1818.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1811.20 | 1813.83 | 1817.50 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 1832.20 | 1821.03 | 1819.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 1834.40 | 1823.71 | 1821.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 1832.50 | 1834.49 | 1830.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 1832.50 | 1834.49 | 1830.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1831.00 | 1833.79 | 1830.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1839.90 | 1833.79 | 1830.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 1844.70 | 1854.48 | 1855.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 1844.70 | 1854.48 | 1855.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 1838.40 | 1849.08 | 1852.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 1846.80 | 1845.88 | 1849.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:00:00 | 1846.80 | 1845.88 | 1849.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1841.30 | 1844.97 | 1848.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:30:00 | 1846.00 | 1844.97 | 1848.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1872.80 | 1849.99 | 1850.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 1872.80 | 1849.99 | 1850.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1870.60 | 1854.12 | 1852.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 11:15:00 | 1873.70 | 1858.03 | 1854.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 12:15:00 | 1876.10 | 1876.51 | 1871.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 12:30:00 | 1877.30 | 1876.51 | 1871.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1870.00 | 1875.13 | 1871.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1870.00 | 1875.13 | 1871.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1870.10 | 1874.12 | 1871.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 1854.00 | 1874.12 | 1871.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1855.00 | 1870.30 | 1870.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:45:00 | 1851.40 | 1870.30 | 1870.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 10:15:00 | 1866.10 | 1869.46 | 1869.82 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 1872.80 | 1870.13 | 1870.09 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 1862.60 | 1868.62 | 1869.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 1854.50 | 1863.70 | 1866.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 11:15:00 | 1859.30 | 1858.76 | 1862.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 12:00:00 | 1859.30 | 1858.76 | 1862.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1861.20 | 1859.02 | 1861.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1857.50 | 1859.02 | 1861.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 1863.20 | 1859.61 | 1861.31 | SL hit (close>static) qty=1.00 sl=1862.40 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1864.60 | 1852.29 | 1851.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 09:15:00 | 1876.10 | 1865.90 | 1861.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 1942.70 | 1944.26 | 1927.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 12:30:00 | 1937.80 | 1944.26 | 1927.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 2017.20 | 2027.73 | 2024.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 2017.20 | 2027.73 | 2024.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 2019.00 | 2025.98 | 2023.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 2022.60 | 2025.98 | 2023.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 2009.00 | 2020.83 | 2021.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 2009.00 | 2020.83 | 2021.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 2007.40 | 2018.14 | 2020.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 2016.30 | 2015.28 | 2018.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 2016.30 | 2015.28 | 2018.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2016.30 | 2015.28 | 2018.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 2016.30 | 2015.28 | 2018.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 2018.30 | 2015.88 | 2018.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 2023.90 | 2015.88 | 2018.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2023.00 | 2017.30 | 2018.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 2018.10 | 2017.30 | 2018.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 2020.10 | 2017.86 | 2018.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 2026.00 | 2017.86 | 2018.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 2014.20 | 2017.55 | 2018.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 13:45:00 | 2010.70 | 2017.70 | 2018.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 2032.10 | 2020.58 | 2019.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 2032.10 | 2020.58 | 2019.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 2036.20 | 2027.62 | 2024.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 12:15:00 | 2026.50 | 2028.38 | 2025.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:00:00 | 2026.50 | 2028.38 | 2025.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2029.00 | 2028.51 | 2025.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 2029.00 | 2028.51 | 2025.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2021.10 | 2027.02 | 2025.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:45:00 | 2014.70 | 2027.02 | 2025.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 2019.20 | 2025.46 | 2024.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 2006.30 | 2025.46 | 2024.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 1987.40 | 2017.85 | 2021.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 1983.40 | 2005.85 | 2015.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 1922.20 | 1919.96 | 1941.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 1922.20 | 1919.96 | 1941.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1930.30 | 1922.53 | 1938.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 1935.30 | 1922.53 | 1938.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1936.90 | 1925.40 | 1938.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1936.90 | 1925.40 | 1938.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1931.80 | 1926.68 | 1938.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 1941.50 | 1926.68 | 1938.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1934.80 | 1930.50 | 1936.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 1930.60 | 1930.50 | 1936.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1932.60 | 1930.92 | 1936.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 1905.00 | 1933.44 | 1935.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 1924.50 | 1909.97 | 1909.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 1924.50 | 1909.97 | 1909.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 12:15:00 | 1929.70 | 1916.35 | 1912.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 12:15:00 | 1937.90 | 1939.96 | 1929.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:00:00 | 1937.90 | 1939.96 | 1929.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1942.20 | 1939.72 | 1932.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:00:00 | 1948.00 | 1941.37 | 1933.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 1916.50 | 1933.12 | 1933.01 | SL hit (close<static) qty=1.00 sl=1925.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1905.10 | 1927.51 | 1930.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 1894.00 | 1916.42 | 1924.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 1912.30 | 1905.26 | 1914.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 1912.30 | 1905.26 | 1914.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1921.80 | 1908.57 | 1915.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 1921.80 | 1908.57 | 1915.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1916.80 | 1910.22 | 1915.48 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1935.00 | 1918.14 | 1917.86 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1905.30 | 1919.65 | 1920.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 1892.30 | 1908.81 | 1913.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1901.40 | 1898.65 | 1905.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 1901.40 | 1898.65 | 1905.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1910.00 | 1900.92 | 1906.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1910.00 | 1900.92 | 1906.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1919.50 | 1904.64 | 1907.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1919.50 | 1904.64 | 1907.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1914.60 | 1906.63 | 1907.99 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 1926.90 | 1912.04 | 1910.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 1933.70 | 1916.37 | 1912.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 13:15:00 | 1929.10 | 1931.54 | 1925.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 14:00:00 | 1929.10 | 1931.54 | 1925.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1932.00 | 1931.63 | 1925.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 1925.50 | 1931.63 | 1925.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1924.00 | 1930.11 | 1925.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 1928.30 | 1930.11 | 1925.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1915.60 | 1927.20 | 1924.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 1915.60 | 1927.20 | 1924.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1917.00 | 1925.16 | 1923.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 1917.00 | 1925.16 | 1923.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1910.40 | 1922.21 | 1922.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 1867.50 | 1908.93 | 1915.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 1860.90 | 1855.14 | 1864.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:45:00 | 1861.40 | 1855.14 | 1864.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 1862.90 | 1856.78 | 1863.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 1862.90 | 1856.78 | 1863.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 1861.20 | 1857.66 | 1863.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 1861.70 | 1857.66 | 1863.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1860.70 | 1858.27 | 1862.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1860.70 | 1858.27 | 1862.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1867.90 | 1860.20 | 1863.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 1864.90 | 1860.20 | 1863.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1870.90 | 1862.34 | 1864.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1872.60 | 1862.34 | 1864.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 1875.00 | 1866.51 | 1865.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1888.90 | 1874.40 | 1870.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 1905.90 | 1907.37 | 1895.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:00:00 | 1905.90 | 1907.37 | 1895.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1931.20 | 1930.40 | 1923.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1927.40 | 1930.40 | 1923.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1919.90 | 1928.30 | 1923.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1919.90 | 1928.30 | 1923.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1926.30 | 1927.90 | 1923.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 1941.70 | 1929.86 | 1925.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 1915.70 | 1928.15 | 1925.62 | SL hit (close<static) qty=1.00 sl=1918.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1914.20 | 1925.66 | 1925.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1897.90 | 1910.14 | 1916.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1896.90 | 1890.03 | 1895.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1896.90 | 1890.03 | 1895.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1896.90 | 1890.03 | 1895.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1896.90 | 1890.03 | 1895.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1898.00 | 1891.62 | 1896.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 1901.20 | 1891.62 | 1896.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1894.40 | 1892.18 | 1896.00 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1907.80 | 1898.90 | 1898.11 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 1894.80 | 1897.56 | 1897.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 1880.20 | 1894.08 | 1896.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 13:15:00 | 1886.70 | 1884.49 | 1889.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:00:00 | 1886.70 | 1884.49 | 1889.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1884.20 | 1884.43 | 1888.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 1877.80 | 1883.08 | 1886.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 1879.20 | 1883.28 | 1886.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:45:00 | 1880.40 | 1883.60 | 1886.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 1881.40 | 1881.69 | 1884.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1884.80 | 1882.17 | 1884.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1884.80 | 1882.17 | 1884.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1891.70 | 1884.08 | 1885.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 1891.70 | 1884.08 | 1885.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 1906.50 | 1888.56 | 1887.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 1906.50 | 1888.56 | 1887.06 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 1886.80 | 1888.81 | 1888.96 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 1897.70 | 1890.59 | 1889.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 12:15:00 | 1901.60 | 1892.79 | 1890.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 14:15:00 | 1894.20 | 1894.26 | 1891.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 14:15:00 | 1894.20 | 1894.26 | 1891.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1894.20 | 1894.26 | 1891.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:30:00 | 1894.80 | 1894.26 | 1891.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1887.90 | 1893.99 | 1892.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 1887.90 | 1893.99 | 1892.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1886.40 | 1892.47 | 1892.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1886.40 | 1892.47 | 1892.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 1886.80 | 1891.33 | 1891.60 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 1893.90 | 1891.92 | 1891.82 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 1890.30 | 1891.60 | 1891.68 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 1895.00 | 1892.28 | 1891.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 1902.80 | 1894.71 | 1893.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 13:15:00 | 1908.50 | 1909.37 | 1903.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:00:00 | 1908.50 | 1909.37 | 1903.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1902.40 | 1907.98 | 1903.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 1902.40 | 1907.98 | 1903.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 1903.00 | 1906.98 | 1903.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 1906.40 | 1906.98 | 1903.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1906.10 | 1906.81 | 1903.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 1910.30 | 1906.81 | 1903.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 1911.60 | 1904.80 | 1903.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 13:15:00 | 1943.80 | 1945.38 | 1945.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 1943.80 | 1945.38 | 1945.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1939.20 | 1944.14 | 1944.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 1935.60 | 1933.30 | 1938.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 1935.60 | 1933.30 | 1938.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1936.00 | 1933.41 | 1937.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 1939.20 | 1933.41 | 1937.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1939.30 | 1934.59 | 1937.45 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 13:15:00 | 1945.60 | 1939.45 | 1939.04 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1935.00 | 1938.31 | 1938.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1930.80 | 1936.81 | 1937.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 1918.00 | 1917.50 | 1924.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 1918.00 | 1917.50 | 1924.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1900.70 | 1913.96 | 1921.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 1898.50 | 1909.28 | 1918.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 1899.60 | 1888.92 | 1888.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 1899.60 | 1888.92 | 1888.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 1904.70 | 1893.64 | 1890.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 15:15:00 | 1939.00 | 1940.50 | 1928.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 09:15:00 | 1944.80 | 1940.50 | 1928.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1940.30 | 1940.75 | 1937.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 1938.90 | 1940.75 | 1937.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1958.40 | 1944.16 | 1939.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 1962.60 | 1952.20 | 1948.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 12:30:00 | 1962.60 | 1955.99 | 1951.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 2062.80 | 2075.41 | 2076.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 2062.80 | 2075.41 | 2076.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 2053.10 | 2064.74 | 2069.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 2075.20 | 2062.16 | 2065.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 2075.20 | 2062.16 | 2065.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 2075.20 | 2062.16 | 2065.13 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 2120.00 | 2076.11 | 2071.10 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 2011.50 | 2081.94 | 2086.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 10:15:00 | 2002.70 | 2066.09 | 2078.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 2022.00 | 2019.41 | 2037.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:00:00 | 2022.00 | 2019.41 | 2037.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2049.00 | 2025.55 | 2035.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 2049.00 | 2025.55 | 2035.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2041.70 | 2028.78 | 2036.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 2049.30 | 2028.78 | 2036.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2065.80 | 2044.70 | 2041.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 12:15:00 | 2071.70 | 2056.29 | 2048.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 2085.00 | 2086.80 | 2074.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:00:00 | 2085.00 | 2086.80 | 2074.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2158.70 | 2158.55 | 2149.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:45:00 | 2171.10 | 2160.31 | 2152.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 14:15:00 | 2170.80 | 2161.29 | 2153.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 15:15:00 | 2145.00 | 2155.36 | 2154.99 | SL hit (close<static) qty=1.00 sl=2148.60 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 2147.20 | 2153.73 | 2154.28 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 2165.50 | 2156.19 | 2155.25 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 2116.40 | 2150.30 | 2153.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 2092.20 | 2104.13 | 2113.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 2102.60 | 2097.87 | 2106.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 2102.60 | 2097.87 | 2106.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 2102.60 | 2097.87 | 2106.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 2107.10 | 2097.87 | 2106.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 2106.00 | 2099.50 | 2106.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 2108.70 | 2099.50 | 2106.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 2098.40 | 2099.28 | 2105.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 2093.80 | 2102.52 | 2105.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 2095.70 | 2090.74 | 2092.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 2099.90 | 2094.41 | 2094.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 2099.90 | 2094.41 | 2094.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 2108.00 | 2099.34 | 2096.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 2097.10 | 2100.56 | 2097.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 2097.10 | 2100.56 | 2097.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 2097.10 | 2100.56 | 2097.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 2097.10 | 2100.56 | 2097.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 2097.20 | 2099.89 | 2097.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 2094.70 | 2099.89 | 2097.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 2089.20 | 2097.75 | 2097.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 2089.20 | 2097.75 | 2097.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 2089.00 | 2096.00 | 2096.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 2081.60 | 2093.12 | 2094.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2089.50 | 2088.72 | 2091.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 2089.50 | 2088.72 | 2091.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2087.50 | 2088.48 | 2091.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 2083.50 | 2088.04 | 2090.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 2083.20 | 2087.92 | 2090.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 2093.70 | 2089.07 | 2090.66 | SL hit (close>static) qty=1.00 sl=2091.40 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 2083.10 | 2070.44 | 2070.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 2092.50 | 2076.56 | 2073.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 2093.70 | 2104.41 | 2099.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 2093.70 | 2104.41 | 2099.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2093.70 | 2104.41 | 2099.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 2093.70 | 2104.41 | 2099.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 2092.20 | 2101.97 | 2098.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 2088.70 | 2101.97 | 2098.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2094.00 | 2098.89 | 2097.79 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 11:15:00 | 2088.70 | 2096.85 | 2096.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 12:15:00 | 2084.10 | 2094.30 | 2095.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 2098.00 | 2094.69 | 2095.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 14:15:00 | 2098.00 | 2094.69 | 2095.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 2098.00 | 2094.69 | 2095.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 2098.00 | 2094.69 | 2095.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 2090.90 | 2093.93 | 2095.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 2115.10 | 2093.93 | 2095.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 2131.30 | 2101.40 | 2098.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 2140.00 | 2109.12 | 2102.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 2130.70 | 2133.00 | 2119.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 2130.10 | 2133.00 | 2119.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2125.80 | 2129.72 | 2121.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 2125.70 | 2129.72 | 2121.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 2123.40 | 2128.04 | 2122.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 2123.40 | 2128.04 | 2122.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 2123.00 | 2127.03 | 2122.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 2126.70 | 2127.03 | 2122.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 2129.60 | 2127.54 | 2122.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 2132.30 | 2127.54 | 2122.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 2131.70 | 2129.60 | 2124.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 2107.90 | 2121.60 | 2122.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 2107.90 | 2121.60 | 2122.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 2100.30 | 2109.38 | 2115.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 2094.70 | 2092.69 | 2102.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:45:00 | 2090.50 | 2092.69 | 2102.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 2099.40 | 2094.03 | 2102.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 2099.40 | 2094.03 | 2102.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 2099.90 | 2095.86 | 2100.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 2099.90 | 2095.86 | 2100.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2100.90 | 2096.87 | 2100.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 2098.80 | 2096.87 | 2100.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 2099.40 | 2097.37 | 2100.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 2095.00 | 2097.37 | 2100.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2096.40 | 2097.18 | 2100.40 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 2105.00 | 2102.38 | 2102.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 2121.70 | 2114.37 | 2110.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 2098.10 | 2111.12 | 2109.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 2098.10 | 2111.12 | 2109.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 2098.10 | 2111.12 | 2109.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 2098.10 | 2111.12 | 2109.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 2102.90 | 2109.48 | 2108.51 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 13:15:00 | 2105.10 | 2107.39 | 2107.65 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 10:15:00 | 2114.10 | 2108.06 | 2107.51 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 2094.40 | 2105.33 | 2106.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 2083.30 | 2099.53 | 2102.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 2089.20 | 2088.46 | 2094.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 2089.20 | 2088.46 | 2094.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 2068.70 | 2071.64 | 2081.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 2062.10 | 2069.17 | 2079.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:15:00 | 1958.99 | 1971.70 | 1981.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 1966.50 | 1958.18 | 1968.09 | SL hit (close>ema200) qty=0.50 sl=1958.18 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1973.00 | 1967.54 | 1967.08 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 1939.00 | 1961.58 | 1964.44 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1988.00 | 1963.87 | 1961.26 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 2012.10 | 2024.86 | 2024.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 2002.10 | 2018.17 | 2021.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 2017.80 | 2015.88 | 2019.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 2017.80 | 2015.88 | 2019.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 2017.80 | 2015.88 | 2019.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 2017.80 | 2015.88 | 2019.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 2014.90 | 2015.68 | 2019.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:30:00 | 2009.60 | 2017.26 | 2018.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 2009.30 | 2017.26 | 2018.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 2007.60 | 2007.84 | 2012.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 13:15:00 | 2026.50 | 2013.56 | 2013.67 | SL hit (close>static) qty=1.00 sl=2020.40 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 2029.60 | 2016.77 | 2015.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 2030.00 | 2019.41 | 2016.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 2021.10 | 2024.42 | 2020.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 13:15:00 | 2021.10 | 2024.42 | 2020.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 2021.10 | 2024.42 | 2020.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 2021.10 | 2024.42 | 2020.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 2020.70 | 2023.67 | 2020.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 2019.90 | 2023.67 | 2020.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 2022.00 | 2023.34 | 2020.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 2027.50 | 2022.87 | 2020.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 2027.00 | 2022.26 | 2021.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 2015.30 | 2020.87 | 2020.81 | SL hit (close<static) qty=1.00 sl=2017.50 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 2015.30 | 2019.76 | 2020.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 2005.70 | 2016.94 | 2018.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1992.40 | 1985.46 | 1995.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 10:15:00 | 1991.90 | 1985.46 | 1995.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1991.10 | 1986.59 | 1994.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1987.10 | 1986.59 | 1994.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1997.00 | 1991.32 | 1994.57 | SL hit (close>static) qty=1.00 sl=1996.30 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 1908.00 | 1897.73 | 1896.87 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 10:15:00 | 1887.40 | 1898.23 | 1899.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 12:15:00 | 1880.50 | 1892.82 | 1896.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 15:15:00 | 1855.70 | 1855.68 | 1865.14 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1839.50 | 1855.68 | 1865.14 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1814.90 | 1795.08 | 1801.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 1814.90 | 1795.08 | 1801.74 | SL hit (close>ema400) qty=1.00 sl=1801.74 alert=retest1 |

### Cycle 59 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 1821.70 | 1805.93 | 1805.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 1833.00 | 1811.34 | 1808.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1843.00 | 1848.13 | 1834.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1843.00 | 1848.13 | 1834.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1843.00 | 1848.13 | 1834.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:45:00 | 1853.40 | 1847.90 | 1836.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:15:00 | 1853.80 | 1848.36 | 1837.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 13:30:00 | 1853.60 | 1847.53 | 1838.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:00:00 | 1854.10 | 1844.57 | 1839.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1843.90 | 1848.26 | 1843.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1843.90 | 1848.26 | 1843.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1844.80 | 1847.57 | 1843.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 1812.00 | 1847.57 | 1843.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 1800.10 | 1838.07 | 1839.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1800.10 | 1838.07 | 1839.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 14:15:00 | 1792.60 | 1812.32 | 1824.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1806.00 | 1802.98 | 1814.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 1803.80 | 1802.98 | 1814.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1812.50 | 1804.54 | 1812.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1812.70 | 1804.54 | 1812.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1819.40 | 1807.51 | 1813.05 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1840.10 | 1816.65 | 1816.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1841.70 | 1824.79 | 1820.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 1831.00 | 1831.42 | 1825.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 11:30:00 | 1832.10 | 1831.42 | 1825.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1825.20 | 1830.18 | 1825.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 1824.20 | 1830.18 | 1825.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 1824.70 | 1829.08 | 1825.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 1824.70 | 1829.08 | 1825.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1846.10 | 1832.48 | 1827.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:15:00 | 1850.00 | 1832.48 | 1827.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1803.00 | 1829.39 | 1827.15 | SL hit (close<static) qty=1.00 sl=1818.40 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1796.40 | 1822.79 | 1824.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 1790.90 | 1810.28 | 1817.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1806.70 | 1802.20 | 1811.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1806.70 | 1802.20 | 1811.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1806.70 | 1802.20 | 1811.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 1800.10 | 1801.78 | 1810.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 1799.90 | 1801.12 | 1809.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1807.50 | 1789.79 | 1788.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 1807.50 | 1789.79 | 1788.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 1810.20 | 1793.87 | 1790.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1864.00 | 1865.19 | 1855.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1864.00 | 1865.19 | 1855.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1864.00 | 1865.19 | 1855.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 1868.70 | 1865.89 | 1856.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1879.00 | 1871.50 | 1863.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 15:15:00 | 1857.50 | 1863.28 | 1863.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 15:15:00 | 1857.50 | 1863.28 | 1863.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 09:15:00 | 1840.50 | 1858.73 | 1861.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 11:15:00 | 1840.40 | 1839.48 | 1846.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 12:00:00 | 1840.40 | 1839.48 | 1846.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1846.40 | 1841.95 | 1846.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 15:00:00 | 1846.40 | 1841.95 | 1846.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 1847.30 | 1843.02 | 1846.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 1848.80 | 1843.02 | 1846.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1848.50 | 1844.11 | 1846.56 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 1852.90 | 1848.70 | 1848.23 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1842.00 | 1846.86 | 1847.45 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1858.80 | 1849.25 | 1848.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1864.70 | 1852.34 | 1849.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 1854.90 | 1856.53 | 1853.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 14:15:00 | 1854.90 | 1856.53 | 1853.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1854.90 | 1856.53 | 1853.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:45:00 | 1854.30 | 1856.53 | 1853.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1858.70 | 1856.97 | 1853.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1850.50 | 1856.97 | 1853.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1837.60 | 1853.09 | 1852.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 1837.60 | 1853.09 | 1852.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 1841.30 | 1850.74 | 1851.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 1833.60 | 1844.17 | 1847.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 13:15:00 | 1837.40 | 1836.92 | 1841.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 14:15:00 | 1837.90 | 1836.92 | 1841.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1840.50 | 1837.64 | 1841.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 1840.50 | 1837.64 | 1841.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 1841.50 | 1838.41 | 1841.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1824.70 | 1838.41 | 1841.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1817.00 | 1834.13 | 1839.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 1808.70 | 1822.71 | 1831.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 1834.40 | 1828.49 | 1827.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 1834.40 | 1828.49 | 1827.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 1851.00 | 1835.98 | 1831.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1869.80 | 1874.91 | 1859.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 11:15:00 | 1877.90 | 1874.25 | 1860.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 11:45:00 | 1879.60 | 1875.22 | 1862.38 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 13:00:00 | 1879.30 | 1876.03 | 1863.92 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1866.00 | 1880.65 | 1870.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1866.00 | 1880.65 | 1870.79 | SL hit (close<ema400) qty=1.00 sl=1870.79 alert=retest1 |

### Cycle 70 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 1840.20 | 1863.43 | 1864.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 13:15:00 | 1830.40 | 1856.83 | 1861.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1831.60 | 1823.21 | 1835.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 10:00:00 | 1831.60 | 1823.21 | 1835.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1834.10 | 1825.38 | 1835.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:15:00 | 1835.90 | 1825.38 | 1835.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1834.50 | 1827.21 | 1835.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:30:00 | 1834.90 | 1827.21 | 1835.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1833.50 | 1829.68 | 1835.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:15:00 | 1837.90 | 1829.68 | 1835.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1832.10 | 1830.16 | 1834.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1830.90 | 1830.89 | 1834.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:45:00 | 1824.30 | 1829.71 | 1833.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 12:45:00 | 1829.50 | 1830.24 | 1833.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:15:00 | 1829.30 | 1830.24 | 1833.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1820.00 | 1827.42 | 1830.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-08 14:15:00 | 1834.40 | 1832.10 | 1832.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 14:15:00 | 1834.40 | 1832.10 | 1832.09 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 09:15:00 | 1854.00 | 2025-05-15 14:15:00 | 1868.10 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-05-14 09:45:00 | 1853.50 | 2025-05-15 14:15:00 | 1868.10 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1839.90 | 2025-06-02 10:15:00 | 1844.70 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1857.50 | 2025-06-12 09:15:00 | 1863.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-06-12 10:30:00 | 1857.50 | 2025-06-16 10:15:00 | 1868.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-06-12 11:30:00 | 1855.50 | 2025-06-16 10:15:00 | 1868.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-06-12 12:00:00 | 1855.00 | 2025-06-16 10:15:00 | 1868.60 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-04 09:15:00 | 2022.60 | 2025-07-04 10:15:00 | 2009.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-07 13:45:00 | 2010.70 | 2025-07-07 14:15:00 | 2032.10 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-18 09:15:00 | 1905.00 | 2025-07-23 10:15:00 | 1924.50 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-25 11:00:00 | 1948.00 | 2025-07-28 09:15:00 | 1916.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-08-22 13:30:00 | 1941.70 | 2025-08-25 09:15:00 | 1915.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-08-25 12:45:00 | 1929.80 | 2025-08-26 09:15:00 | 1914.20 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-08-25 14:30:00 | 1930.10 | 2025-08-26 09:15:00 | 1914.20 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-04 10:30:00 | 1877.80 | 2025-09-05 13:15:00 | 1906.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-09-04 12:15:00 | 1879.20 | 2025-09-05 13:15:00 | 1906.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-04 14:45:00 | 1880.40 | 2025-09-05 13:15:00 | 1906.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-05 09:45:00 | 1881.40 | 2025-09-05 13:15:00 | 1906.50 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-15 10:15:00 | 1910.30 | 2025-09-23 13:15:00 | 1943.80 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1911.60 | 2025-09-23 13:15:00 | 1943.80 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-09-30 09:30:00 | 1898.50 | 2025-10-06 10:15:00 | 1899.60 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-10-15 11:15:00 | 1962.60 | 2025-10-30 13:15:00 | 2062.80 | STOP_HIT | 1.00 | 5.11% |
| BUY | retest2 | 2025-10-15 12:30:00 | 1962.60 | 2025-10-30 13:15:00 | 2062.80 | STOP_HIT | 1.00 | 5.11% |
| BUY | retest2 | 2025-11-21 12:45:00 | 2171.10 | 2025-11-24 15:15:00 | 2145.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-21 14:15:00 | 2170.80 | 2025-11-24 15:15:00 | 2145.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-03 09:15:00 | 2093.80 | 2025-12-05 10:15:00 | 2099.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-12-05 09:15:00 | 2095.70 | 2025-12-05 10:15:00 | 2099.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-12-09 14:15:00 | 2083.50 | 2025-12-10 09:15:00 | 2093.70 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-12-10 09:15:00 | 2083.20 | 2025-12-10 09:15:00 | 2093.70 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-12-10 11:30:00 | 2082.50 | 2025-12-12 14:15:00 | 2083.10 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-12-24 10:15:00 | 2132.30 | 2025-12-26 10:15:00 | 2107.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-12-24 12:45:00 | 2131.70 | 2025-12-26 10:15:00 | 2107.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-09 10:45:00 | 2062.10 | 2026-01-28 11:15:00 | 1958.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 10:45:00 | 2062.10 | 2026-01-29 11:15:00 | 1966.50 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2026-02-13 10:30:00 | 2009.60 | 2026-02-16 13:15:00 | 2026.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-13 11:15:00 | 2009.30 | 2026-02-16 13:15:00 | 2026.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-02-16 10:15:00 | 2007.60 | 2026-02-16 13:15:00 | 2026.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-02-18 09:30:00 | 2027.50 | 2026-02-19 09:15:00 | 2015.30 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-02-19 09:15:00 | 2027.00 | 2026-02-19 09:15:00 | 2015.30 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1987.10 | 2026-02-23 14:15:00 | 1997.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1946.60 | 2026-03-02 09:15:00 | 1849.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1946.60 | 2026-03-04 10:15:00 | 1891.10 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest1 | 2026-03-11 09:15:00 | 1839.50 | 2026-03-17 09:15:00 | 1814.90 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2026-03-19 10:45:00 | 1853.40 | 2026-03-23 09:15:00 | 1800.10 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2026-03-19 12:15:00 | 1853.80 | 2026-03-23 09:15:00 | 1800.10 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2026-03-19 13:30:00 | 1853.60 | 2026-03-23 09:15:00 | 1800.10 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-03-20 11:00:00 | 1854.10 | 2026-03-23 09:15:00 | 1800.10 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2026-03-27 15:15:00 | 1850.00 | 2026-03-30 09:15:00 | 1803.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-04-01 11:00:00 | 1800.10 | 2026-04-07 09:15:00 | 1807.50 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-04-01 11:30:00 | 1799.90 | 2026-04-07 09:15:00 | 1807.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-04-13 11:00:00 | 1868.70 | 2026-04-15 15:15:00 | 1857.50 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1879.00 | 2026-04-15 15:15:00 | 1857.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-04-24 13:30:00 | 1808.70 | 2026-04-28 11:15:00 | 1834.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest1 | 2026-04-30 11:15:00 | 1877.90 | 2026-05-04 09:15:00 | 1866.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2026-04-30 11:45:00 | 1879.60 | 2026-05-04 09:15:00 | 1866.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2026-04-30 13:00:00 | 1879.30 | 2026-05-04 09:15:00 | 1866.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-05-07 09:15:00 | 1830.90 | 2026-05-08 14:15:00 | 1834.40 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-05-07 09:45:00 | 1824.30 | 2026-05-08 14:15:00 | 1834.40 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-05-07 12:45:00 | 1829.50 | 2026-05-08 14:15:00 | 1834.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-05-07 13:15:00 | 1829.30 | 2026-05-08 14:15:00 | 1834.40 | STOP_HIT | 1.00 | -0.28% |
