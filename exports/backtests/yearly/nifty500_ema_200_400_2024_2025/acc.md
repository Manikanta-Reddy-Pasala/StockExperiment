# ACC Ltd. (ACC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1393.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 72 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 61 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 53
- **Target hits / Stop hits / Partials:** 2 / 59 / 7
- **Avg / median % per leg:** -1.06% / -1.47%
- **Sum % (uncompounded):** -71.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 1 | 8 | 0 | -2.36% | -21.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 1 | 8 | 0 | -2.36% | -21.2% |
| SELL (all) | 59 | 14 | 23.7% | 1 | 51 | 7 | -0.86% | -50.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 59 | 14 | 23.7% | 1 | 51 | 7 | -0.86% | -50.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 15 | 22.1% | 2 | 59 | 7 | -1.06% | -71.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 2386.00 | 2579.18 | 2579.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 12:15:00 | 2378.60 | 2575.28 | 2577.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 2408.25 | 2395.75 | 2456.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 10:00:00 | 2408.25 | 2395.75 | 2456.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 2449.65 | 2400.64 | 2454.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:15:00 | 2456.00 | 2400.64 | 2454.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 2458.40 | 2401.21 | 2454.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:30:00 | 2448.40 | 2402.27 | 2454.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:15:00 | 2447.75 | 2402.76 | 2454.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:30:00 | 2448.05 | 2404.05 | 2454.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 09:15:00 | 2466.40 | 2405.47 | 2454.00 | SL hit (close>static) qty=1.00 sl=2465.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 14:15:00 | 2081.10 | 1972.43 | 1972.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 10:15:00 | 2091.70 | 1981.22 | 1976.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 1959.90 | 1985.95 | 1979.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 1959.90 | 1985.95 | 1979.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1959.90 | 1985.95 | 1979.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1959.90 | 1985.95 | 1979.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1943.50 | 1985.52 | 1979.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1943.50 | 1985.52 | 1979.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 1877.20 | 1972.95 | 1973.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 1872.50 | 1966.14 | 1969.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 11:15:00 | 1927.40 | 1911.94 | 1936.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:45:00 | 1932.10 | 1911.94 | 1936.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1945.80 | 1912.75 | 1936.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 1945.80 | 1912.75 | 1936.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 1937.70 | 1913.00 | 1936.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 14:15:00 | 1924.10 | 1915.41 | 1936.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 1925.10 | 1915.92 | 1936.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1964.00 | 1918.68 | 1936.60 | SL hit (close>static) qty=1.00 sl=1947.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-07-10 13:15:00)

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

### Cycle 5 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 1829.00 | 1923.25 | 1923.40 | EMA200 below EMA400 |
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


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 14:15:00 | 2517.00 | 2024-06-04 10:15:00 | 2414.55 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2024-05-18 09:15:00 | 2534.40 | 2024-06-04 10:15:00 | 2414.55 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2024-05-21 10:45:00 | 2520.90 | 2024-06-04 10:15:00 | 2414.55 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2024-05-22 09:45:00 | 2520.05 | 2024-06-04 10:15:00 | 2414.55 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-05-31 09:15:00 | 2534.85 | 2024-06-04 10:15:00 | 2414.55 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2024-05-31 11:00:00 | 2516.00 | 2024-06-04 10:15:00 | 2414.55 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2024-06-04 09:30:00 | 2514.85 | 2024-06-04 10:15:00 | 2414.55 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-06-10 09:15:00 | 2533.00 | 2024-07-02 09:15:00 | 2786.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-10 09:30:00 | 2448.40 | 2024-09-11 09:15:00 | 2466.40 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-09-10 11:15:00 | 2447.75 | 2024-09-11 09:15:00 | 2466.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-09-10 13:30:00 | 2448.05 | 2024-09-11 09:15:00 | 2466.40 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-09-11 14:00:00 | 2442.85 | 2024-09-12 14:15:00 | 2467.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-09-26 10:15:00 | 2433.40 | 2024-09-26 14:15:00 | 2473.05 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-10-04 13:30:00 | 2430.55 | 2024-10-10 11:15:00 | 2309.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 09:15:00 | 2428.65 | 2024-10-10 11:15:00 | 2307.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 13:30:00 | 2430.55 | 2024-10-30 09:15:00 | 2353.20 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2024-10-07 09:15:00 | 2428.65 | 2024-10-30 09:15:00 | 2353.20 | STOP_HIT | 0.50 | 3.11% |
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
