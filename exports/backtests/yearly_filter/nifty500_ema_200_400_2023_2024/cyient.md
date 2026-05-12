# Cyient Ltd. (CYIENT)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 902.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 35 |
| PARTIAL | 12 |
| TARGET_HIT | 6 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 28
- **Target hits / Stop hits / Partials:** 6 / 33 / 12
- **Avg / median % per leg:** 0.79% / -1.19%
- **Sum % (uncompounded):** 40.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.44% | -13.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.44% | -13.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 23 | 48.9% | 6 | 29 | 12 | 1.15% | 54.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 47 | 23 | 48.9% | 6 | 29 | 12 | 1.15% | 54.1% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.44% | -13.8% |
| retest2 (combined) | 47 | 23 | 48.9% | 6 | 29 | 12 | 1.15% | 54.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 10:15:00 | 1957.60 | 2060.92 | 2061.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 11:15:00 | 1933.20 | 2053.08 | 2057.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 13:15:00 | 2028.85 | 2025.27 | 2041.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-15 14:00:00 | 2028.85 | 2025.27 | 2041.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 1980.05 | 2017.30 | 2036.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 09:15:00 | 1945.05 | 2013.86 | 2033.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 12:00:00 | 1971.80 | 2012.69 | 2032.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 13:00:00 | 1968.05 | 2012.24 | 2032.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 11:00:00 | 1956.00 | 2009.41 | 2030.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 1998.45 | 2001.67 | 2024.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:30:00 | 2019.80 | 2001.67 | 2024.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 2012.40 | 2001.45 | 2023.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 14:45:00 | 2005.85 | 2001.45 | 2023.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 2035.35 | 2002.31 | 2022.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:00:00 | 2035.35 | 2002.31 | 2022.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 2063.70 | 2002.92 | 2023.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-03 10:15:00 | 2063.70 | 2002.92 | 2023.06 | SL hit (close>static) qty=1.00 sl=2053.65 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 10:15:00 | 2086.85 | 2039.90 | 2039.77 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-04-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 12:15:00 | 1991.15 | 2039.52 | 2039.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 13:15:00 | 1984.60 | 2038.98 | 2039.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 15:15:00 | 1796.00 | 1795.81 | 1863.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:15:00 | 1799.00 | 1795.81 | 1863.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1884.70 | 1797.29 | 1861.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:00:00 | 1884.70 | 1797.29 | 1861.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 1862.85 | 1797.95 | 1861.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 11:15:00 | 1848.55 | 1848.70 | 1870.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 14:30:00 | 1850.90 | 1848.19 | 1869.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 15:00:00 | 1841.75 | 1848.19 | 1869.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:30:00 | 1851.00 | 1848.47 | 1868.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 1871.30 | 1848.70 | 1868.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-28 11:15:00 | 1887.70 | 1849.08 | 1869.07 | SL hit (close>static) qty=1.00 sl=1885.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 2000.90 | 1830.30 | 1830.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 12:15:00 | 2007.75 | 1833.78 | 1831.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 15:15:00 | 1987.00 | 1996.73 | 1939.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:45:00 | 2000.15 | 1996.75 | 1940.24 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 10:15:00 | 2001.05 | 1996.75 | 1940.24 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 10:45:00 | 2003.05 | 1996.86 | 1940.57 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 11:45:00 | 2001.75 | 1996.86 | 1940.85 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1932.55 | 1995.75 | 1943.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 1932.55 | 1995.75 | 1943.56 | SL hit (close<ema400) qty=1.00 sl=1943.56 alert=retest1 |

### Cycle 5 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 1833.00 | 1913.71 | 1913.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 1824.70 | 1910.37 | 1912.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1871.70 | 1853.60 | 1877.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 1871.70 | 1853.60 | 1877.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1871.70 | 1853.60 | 1877.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 1871.70 | 1853.60 | 1877.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 1863.90 | 1854.03 | 1877.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 10:15:00 | 1861.65 | 1854.71 | 1877.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 1883.85 | 1855.00 | 1877.07 | SL hit (close>static) qty=1.00 sl=1879.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 2025.00 | 1875.24 | 1875.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 2048.65 | 1882.30 | 1878.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 1946.40 | 1960.35 | 1925.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:00:00 | 1946.40 | 1960.35 | 1925.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1922.70 | 1959.97 | 1925.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1922.70 | 1959.97 | 1925.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1923.85 | 1959.61 | 1925.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 1917.30 | 1959.61 | 1925.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1918.45 | 1959.20 | 1925.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 1915.05 | 1959.20 | 1925.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1907.50 | 1958.69 | 1924.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 1907.50 | 1958.69 | 1924.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1916.80 | 1955.09 | 1924.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:00:00 | 1916.80 | 1955.09 | 1924.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 1910.15 | 1954.65 | 1924.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:00:00 | 1910.15 | 1954.65 | 1924.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 1907.45 | 1954.18 | 1924.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:15:00 | 1908.10 | 1954.18 | 1924.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 1883.20 | 1951.28 | 1923.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:00:00 | 1883.20 | 1951.28 | 1923.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1926.70 | 1948.80 | 1923.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 1926.70 | 1948.80 | 1923.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1920.80 | 1948.52 | 1923.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:00:00 | 1920.80 | 1948.52 | 1923.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 1930.40 | 1948.34 | 1923.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:30:00 | 1917.40 | 1948.34 | 1923.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 1921.45 | 1947.84 | 1923.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 1882.25 | 1947.84 | 1923.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1864.80 | 1947.01 | 1922.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 1859.40 | 1947.01 | 1922.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1849.45 | 1946.04 | 1922.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 1849.45 | 1946.04 | 1922.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1753.15 | 1902.87 | 1903.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1743.95 | 1901.29 | 1902.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 11:15:00 | 1342.75 | 1331.79 | 1457.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 12:00:00 | 1342.75 | 1331.79 | 1457.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1288.00 | 1218.10 | 1287.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:00:00 | 1288.00 | 1218.10 | 1287.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 1283.70 | 1218.75 | 1287.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 1274.00 | 1309.00 | 1310.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 14:15:00 | 1297.60 | 1308.21 | 1309.95 | SL hit (close>static) qty=1.00 sl=1291.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-03-22 09:15:00 | 1945.05 | 2024-04-03 10:15:00 | 2063.70 | STOP_HIT | 1.00 | -6.10% |
| SELL | retest2 | 2024-03-22 12:00:00 | 1971.80 | 2024-04-03 10:15:00 | 2063.70 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2024-03-22 13:00:00 | 1968.05 | 2024-04-03 10:15:00 | 2063.70 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2024-03-26 11:00:00 | 1956.00 | 2024-04-03 10:15:00 | 2063.70 | STOP_HIT | 1.00 | -5.51% |
| SELL | retest2 | 2024-06-26 11:15:00 | 1848.55 | 2024-06-28 11:15:00 | 1887.70 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-06-27 14:30:00 | 1850.90 | 2024-06-28 11:15:00 | 1887.70 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-06-27 15:00:00 | 1841.75 | 2024-06-28 11:15:00 | 1887.70 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-06-28 09:30:00 | 1851.00 | 2024-06-28 11:15:00 | 1887.70 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-06-28 11:30:00 | 1852.60 | 2024-06-28 12:15:00 | 1892.70 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-06-28 14:15:00 | 1858.25 | 2024-07-11 11:15:00 | 1768.90 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2024-07-01 10:15:00 | 1849.85 | 2024-07-11 12:15:00 | 1765.34 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2024-06-28 14:15:00 | 1858.25 | 2024-07-12 10:15:00 | 1842.10 | STOP_HIT | 0.50 | 0.87% |
| SELL | retest2 | 2024-07-01 10:15:00 | 1849.85 | 2024-07-12 10:15:00 | 1842.10 | STOP_HIT | 0.50 | 0.42% |
| SELL | retest2 | 2024-07-02 12:15:00 | 1862.00 | 2024-07-15 09:15:00 | 1889.85 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-07-02 15:15:00 | 1842.90 | 2024-07-15 09:15:00 | 1889.85 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-07-12 12:30:00 | 1824.00 | 2024-07-15 09:15:00 | 1889.85 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2024-07-12 13:45:00 | 1841.95 | 2024-07-15 09:15:00 | 1889.85 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-07-12 15:00:00 | 1832.45 | 2024-07-15 09:15:00 | 1889.85 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-07-16 09:15:00 | 1842.25 | 2024-07-26 09:15:00 | 1750.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 12:45:00 | 1843.35 | 2024-07-26 09:15:00 | 1751.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 10:45:00 | 1843.00 | 2024-07-26 09:15:00 | 1750.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 14:15:00 | 1844.05 | 2024-07-26 09:15:00 | 1751.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-26 09:15:00 | 1747.85 | 2024-08-05 10:15:00 | 1660.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 09:15:00 | 1842.25 | 2024-08-05 11:15:00 | 1658.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-18 12:45:00 | 1843.35 | 2024-08-05 11:15:00 | 1659.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-19 10:45:00 | 1843.00 | 2024-08-05 11:15:00 | 1658.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-19 14:15:00 | 1844.05 | 2024-08-05 11:15:00 | 1659.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-26 09:15:00 | 1747.85 | 2024-08-16 11:15:00 | 1778.95 | STOP_HIT | 0.50 | -1.78% |
| BUY | retest1 | 2024-09-25 09:45:00 | 2000.15 | 2024-09-27 09:15:00 | 1932.55 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest1 | 2024-09-25 10:15:00 | 2001.05 | 2024-09-27 09:15:00 | 1932.55 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest1 | 2024-09-25 10:45:00 | 2003.05 | 2024-09-27 09:15:00 | 1932.55 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest1 | 2024-09-25 11:45:00 | 2001.75 | 2024-09-27 09:15:00 | 1932.55 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2024-11-06 10:15:00 | 1861.65 | 2024-11-06 10:15:00 | 1883.85 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-11-11 13:30:00 | 1860.50 | 2024-11-18 09:15:00 | 1767.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 09:15:00 | 1859.75 | 2024-11-18 09:15:00 | 1766.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:15:00 | 1858.70 | 2024-11-18 09:15:00 | 1765.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:30:00 | 1860.50 | 2024-11-25 09:15:00 | 1848.35 | STOP_HIT | 0.50 | 0.65% |
| SELL | retest2 | 2024-11-12 09:15:00 | 1859.75 | 2024-11-25 09:15:00 | 1848.35 | STOP_HIT | 0.50 | 0.61% |
| SELL | retest2 | 2024-11-12 10:15:00 | 1858.70 | 2024-11-25 09:15:00 | 1848.35 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2024-11-26 11:45:00 | 1878.40 | 2024-12-04 09:15:00 | 1946.50 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2024-11-28 09:15:00 | 1870.00 | 2024-12-04 09:15:00 | 1946.50 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2024-11-28 10:15:00 | 1878.15 | 2024-12-04 09:15:00 | 1946.50 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2024-12-02 12:30:00 | 1875.80 | 2024-12-04 09:15:00 | 1946.50 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-06-26 11:15:00 | 1274.00 | 2025-06-26 14:15:00 | 1297.60 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-07-10 12:00:00 | 1279.80 | 2025-07-10 14:15:00 | 1292.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-11 11:00:00 | 1276.70 | 2025-07-14 13:15:00 | 1292.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-07-11 11:45:00 | 1280.20 | 2025-07-14 13:15:00 | 1292.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-17 09:15:00 | 1299.80 | 2025-07-24 14:15:00 | 1234.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 12:00:00 | 1298.60 | 2025-07-24 15:15:00 | 1233.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 09:15:00 | 1299.80 | 2025-08-07 11:15:00 | 1169.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-17 12:00:00 | 1298.60 | 2025-08-07 11:15:00 | 1168.74 | TARGET_HIT | 0.50 | 10.00% |
