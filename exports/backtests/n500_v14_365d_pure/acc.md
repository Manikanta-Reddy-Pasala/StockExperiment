# ACC Ltd. (ACC)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1393.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 46 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 41
- **Target hits / Stop hits / Partials:** 1 / 45 / 5
- **Avg / median % per leg:** -1.24% / -1.64%
- **Sum % (uncompounded):** -63.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.28% | -1.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.28% | -1.3% |
| SELL (all) | 50 | 10 | 20.0% | 1 | 44 | 5 | -1.24% | -62.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 50 | 10 | 20.0% | 1 | 44 | 5 | -1.24% | -62.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 51 | 10 | 19.6% | 1 | 45 | 5 | -1.24% | -63.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 1999.70 | 1917.35 | 1916.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 2002.70 | 1918.19 | 1917.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1922.00 | 1943.98 | 1932.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1922.00 | 1943.98 | 1932.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1922.00 | 1943.98 | 1932.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1922.00 | 1943.98 | 1932.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1923.40 | 1943.78 | 1932.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:00:00 | 1934.70 | 1943.69 | 1932.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 1910.00 | 1943.62 | 1932.94 | SL hit (close<static) qty=1.00 sl=1920.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 1829.00 | 1923.25 | 1923.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 1814.20 | 1922.16 | 1922.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1854.90 | 1851.47 | 1879.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 10:15:00 | 1859.30 | 1851.47 | 1879.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1851.00 | 1837.94 | 1862.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 1848.10 | 1838.10 | 1862.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 1848.50 | 1838.25 | 1862.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 1850.20 | 1838.37 | 1862.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 1850.00 | 1838.21 | 1860.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1851.00 | 1839.93 | 1859.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 1853.60 | 1839.93 | 1859.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1860.00 | 1841.02 | 1858.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 1860.00 | 1841.02 | 1858.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1859.10 | 1841.20 | 1858.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 1859.10 | 1841.20 | 1858.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1860.00 | 1841.39 | 1858.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:45:00 | 1863.20 | 1841.39 | 1858.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1859.10 | 1841.56 | 1858.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 1861.00 | 1841.56 | 1858.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1858.90 | 1841.74 | 1858.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 1858.90 | 1841.74 | 1858.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1860.10 | 1841.92 | 1858.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 1860.10 | 1841.92 | 1858.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1860.00 | 1842.10 | 1858.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1875.40 | 1842.10 | 1858.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1860.80 | 1844.53 | 1859.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:45:00 | 1861.80 | 1844.53 | 1859.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1858.90 | 1844.67 | 1859.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 1859.60 | 1844.67 | 1859.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1860.50 | 1845.09 | 1859.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 1864.40 | 1845.09 | 1859.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1864.00 | 1845.28 | 1859.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 1864.00 | 1845.28 | 1859.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1850.00 | 1845.33 | 1859.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1903.90 | 1848.41 | 1859.98 | SL hit (close>static) qty=1.00 sl=1881.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1903.90 | 1848.41 | 1859.98 | SL hit (close>static) qty=1.00 sl=1881.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1903.90 | 1848.41 | 1859.98 | SL hit (close>static) qty=1.00 sl=1881.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1903.90 | 1848.41 | 1859.98 | SL hit (close>static) qty=1.00 sl=1881.10 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:30:00 | 1836.90 | 1853.72 | 1861.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:00:00 | 1833.90 | 1852.56 | 1860.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 1838.00 | 1849.16 | 1858.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:15:00 | 1826.00 | 1848.86 | 1857.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1847.10 | 1847.99 | 1857.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:30:00 | 1842.70 | 1847.99 | 1857.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:45:00 | 1841.20 | 1847.96 | 1856.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 12:45:00 | 1841.40 | 1847.86 | 1856.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 1841.80 | 1847.49 | 1856.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1856.10 | 1847.46 | 1856.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:00:00 | 1856.10 | 1847.46 | 1856.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1857.40 | 1847.55 | 1856.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:15:00 | 1857.50 | 1847.55 | 1856.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1857.40 | 1847.65 | 1856.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1857.40 | 1847.65 | 1856.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1857.50 | 1847.75 | 1856.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1863.20 | 1847.75 | 1856.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1857.90 | 1847.85 | 1856.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 1859.30 | 1847.85 | 1856.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1844.30 | 1847.82 | 1856.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1863.90 | 1848.05 | 1856.26 | SL hit (close>static) qty=1.00 sl=1863.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1863.90 | 1848.05 | 1856.26 | SL hit (close>static) qty=1.00 sl=1863.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1863.90 | 1848.05 | 1856.26 | SL hit (close>static) qty=1.00 sl=1863.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1863.90 | 1848.05 | 1856.26 | SL hit (close>static) qty=1.00 sl=1863.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1867.80 | 1848.38 | 1856.34 | SL hit (close>static) qty=1.00 sl=1865.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1867.80 | 1848.38 | 1856.34 | SL hit (close>static) qty=1.00 sl=1865.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1867.80 | 1848.38 | 1856.34 | SL hit (close>static) qty=1.00 sl=1865.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1867.80 | 1848.38 | 1856.34 | SL hit (close>static) qty=1.00 sl=1865.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:15:00 | 1836.50 | 1854.67 | 1858.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 1838.40 | 1852.26 | 1856.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 1859.00 | 1851.65 | 1856.40 | SL hit (close>static) qty=1.00 sl=1858.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 1859.00 | 1851.65 | 1856.40 | SL hit (close>static) qty=1.00 sl=1858.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 1840.70 | 1851.31 | 1856.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 1840.60 | 1851.22 | 1855.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1854.00 | 1851.20 | 1855.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:30:00 | 1855.00 | 1851.20 | 1855.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1863.10 | 1851.32 | 1855.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 1863.10 | 1851.32 | 1855.92 | SL hit (close>static) qty=1.00 sl=1858.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 1863.10 | 1851.32 | 1855.92 | SL hit (close>static) qty=1.00 sl=1858.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 1863.10 | 1851.32 | 1855.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1855.00 | 1851.35 | 1855.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:15:00 | 1852.00 | 1851.39 | 1855.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:30:00 | 1854.70 | 1851.59 | 1855.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 1865.50 | 1851.73 | 1855.95 | SL hit (close>static) qty=1.00 sl=1864.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 1865.50 | 1851.73 | 1855.95 | SL hit (close>static) qty=1.00 sl=1864.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 1854.10 | 1857.17 | 1858.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:00:00 | 1853.60 | 1852.08 | 1855.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1857.50 | 1852.14 | 1855.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 1857.50 | 1852.14 | 1855.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 1852.50 | 1852.14 | 1855.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 1851.50 | 1852.14 | 1855.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1850.90 | 1852.17 | 1855.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 1850.80 | 1852.20 | 1855.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 1851.70 | 1852.28 | 1855.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1844.10 | 1851.44 | 1854.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:45:00 | 1840.60 | 1850.99 | 1854.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 1841.10 | 1850.81 | 1854.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 1840.40 | 1850.20 | 1853.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:00:00 | 1840.00 | 1849.82 | 1853.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1858.00 | 1849.60 | 1853.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 1851.30 | 1849.62 | 1853.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:45:00 | 1851.40 | 1849.69 | 1853.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:45:00 | 1851.40 | 1849.67 | 1853.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1847.80 | 1849.70 | 1853.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1840.80 | 1849.61 | 1853.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:30:00 | 1838.90 | 1849.50 | 1853.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:45:00 | 1838.90 | 1849.39 | 1853.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 12:45:00 | 1837.80 | 1849.27 | 1853.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:30:00 | 1838.90 | 1849.17 | 1852.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1864.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1864.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1858.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1858.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1858.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1858.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1858.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1858.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1858.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1858.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1865.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1865.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1865.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1865.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1855.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1855.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1855.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1916.20 | 1848.26 | 1852.33 | SL hit (close>static) qty=1.00 sl=1855.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1916.20 | 1848.26 | 1852.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1930.00 | 1849.07 | 1852.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1878.70 | 1849.07 | 1852.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1784.76 | 1843.44 | 1849.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 1690.83 | 1766.42 | 1794.56 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-20 14:15:00 | 1924.10 | 2025-05-23 09:15:00 | 1964.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-05-21 11:45:00 | 1925.10 | 2025-05-23 09:15:00 | 1964.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-05-28 14:15:00 | 1922.10 | 2025-06-19 12:15:00 | 1828.18 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-05-29 13:15:00 | 1924.40 | 2025-06-19 13:15:00 | 1825.99 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-06-10 15:00:00 | 1913.30 | 2025-06-20 09:15:00 | 1817.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:30:00 | 1911.50 | 2025-06-20 09:15:00 | 1815.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-28 14:15:00 | 1922.10 | 2025-06-26 14:15:00 | 1882.10 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2025-05-29 13:15:00 | 1924.40 | 2025-06-26 14:15:00 | 1882.10 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2025-06-10 15:00:00 | 1913.30 | 2025-06-26 14:15:00 | 1882.10 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2025-06-11 12:30:00 | 1911.50 | 2025-06-26 14:15:00 | 1882.10 | STOP_HIT | 0.50 | 1.54% |
| SELL | retest2 | 2025-06-30 10:30:00 | 1911.60 | 2025-07-02 10:15:00 | 1931.70 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-06-30 11:45:00 | 1913.60 | 2025-07-02 10:15:00 | 1931.70 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-07-23 12:00:00 | 1934.70 | 2025-07-24 13:15:00 | 1910.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-09-04 11:15:00 | 1848.10 | 2025-09-22 10:15:00 | 1903.90 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-09-04 12:15:00 | 1848.50 | 2025-09-22 10:15:00 | 1903.90 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-09-04 13:00:00 | 1850.20 | 2025-09-22 10:15:00 | 1903.90 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-09-09 10:00:00 | 1850.00 | 2025-09-22 10:15:00 | 1903.90 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-09-26 11:30:00 | 1836.90 | 2025-10-08 14:15:00 | 1863.90 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-29 10:00:00 | 1833.90 | 2025-10-08 14:15:00 | 1863.90 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-01 10:15:00 | 1838.00 | 2025-10-08 14:15:00 | 1863.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-10-01 14:15:00 | 1826.00 | 2025-10-08 14:15:00 | 1863.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-10-06 10:30:00 | 1842.70 | 2025-10-09 09:15:00 | 1867.80 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-06 11:45:00 | 1841.20 | 2025-10-09 09:15:00 | 1867.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-10-06 12:45:00 | 1841.40 | 2025-10-09 09:15:00 | 1867.80 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-10-07 09:30:00 | 1841.80 | 2025-10-09 09:15:00 | 1867.80 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-10-17 13:15:00 | 1836.50 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-23 09:15:00 | 1838.40 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-27 10:15:00 | 1840.70 | 2025-10-27 14:15:00 | 1863.10 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-27 10:45:00 | 1840.60 | 2025-10-27 14:15:00 | 1863.10 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-28 10:15:00 | 1852.00 | 2025-10-28 15:15:00 | 1865.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-10-28 14:30:00 | 1854.70 | 2025-10-28 15:15:00 | 1865.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-04 09:30:00 | 1854.10 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-11-12 11:00:00 | 1853.60 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-11-12 14:15:00 | 1851.50 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-11-13 09:15:00 | 1850.90 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2025-11-13 10:45:00 | 1850.80 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2025-11-13 12:30:00 | 1851.70 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-11-18 09:45:00 | 1840.60 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-11-18 11:30:00 | 1841.10 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2025-11-19 09:30:00 | 1840.40 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -4.12% |
| SELL | retest2 | 2025-11-19 13:00:00 | 1840.00 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2025-11-20 12:00:00 | 1851.30 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-11-20 13:45:00 | 1851.40 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-11-20 14:45:00 | 1851.40 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1847.80 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-11-21 10:30:00 | 1838.90 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-11-21 11:45:00 | 1838.90 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-11-21 12:45:00 | 1837.80 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-11-21 13:30:00 | 1838.90 | 2025-11-24 14:15:00 | 1916.20 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1878.70 | 2025-12-09 09:15:00 | 1784.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1878.70 | 2026-01-12 09:15:00 | 1690.83 | TARGET_HIT | 0.50 | 10.00% |
