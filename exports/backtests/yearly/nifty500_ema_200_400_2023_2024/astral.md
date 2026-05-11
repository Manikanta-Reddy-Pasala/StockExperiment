# Astral Ltd. (ASTRAL)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 1567.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 12 |
| ALERT2 | 13 |
| ALERT2_SKIP | 3 |
| ALERT3 | 120 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 79 |
| PARTIAL | 7 |
| TARGET_HIT | 12 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 67
- **Target hits / Stop hits / Partials:** 12 / 67 / 7
- **Avg / median % per leg:** 0.35% / -1.29%
- **Sum % (uncompounded):** 30.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 5 | 11.6% | 5 | 38 | 0 | -0.63% | -27.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 43 | 5 | 11.6% | 5 | 38 | 0 | -0.63% | -27.2% |
| SELL (all) | 43 | 14 | 32.6% | 7 | 29 | 7 | 1.34% | 57.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 14 | 32.6% | 7 | 29 | 7 | 1.34% | 57.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 86 | 19 | 22.1% | 12 | 67 | 7 | 0.35% | 30.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 11:15:00 | 1832.10 | 1900.91 | 1901.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 1807.85 | 1899.98 | 1900.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 14:15:00 | 1872.85 | 1871.04 | 1883.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-07 15:00:00 | 1872.85 | 1871.04 | 1883.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 1876.45 | 1871.12 | 1883.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 11:45:00 | 1866.05 | 1874.92 | 1883.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 13:45:00 | 1870.20 | 1874.85 | 1883.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 14:15:00 | 1870.65 | 1874.85 | 1883.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 15:15:00 | 1868.00 | 1874.84 | 1883.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 1894.40 | 1874.97 | 1883.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 1894.40 | 1874.97 | 1883.58 | SL hit (close>static) qty=1.00 sl=1886.35 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 11:15:00 | 1961.80 | 1890.33 | 1890.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 11:15:00 | 1975.70 | 1902.84 | 1896.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 13:15:00 | 1923.05 | 1927.07 | 1911.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 14:00:00 | 1923.05 | 1927.07 | 1911.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 1918.70 | 1927.06 | 1911.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:30:00 | 1920.40 | 1927.06 | 1911.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 1918.60 | 1928.39 | 1913.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:45:00 | 1916.00 | 1928.39 | 1913.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 10:15:00 | 1901.20 | 1928.12 | 1913.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 11:00:00 | 1901.20 | 1928.12 | 1913.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 11:15:00 | 1901.05 | 1927.85 | 1913.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 11:30:00 | 1900.00 | 1927.85 | 1913.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 1920.80 | 1927.61 | 1913.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 14:30:00 | 1921.95 | 1927.61 | 1913.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 1917.55 | 1927.43 | 1913.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:45:00 | 1913.35 | 1927.43 | 1913.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 1925.70 | 1927.42 | 1913.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:30:00 | 1917.05 | 1927.42 | 1913.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 1905.15 | 1931.68 | 1917.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 15:00:00 | 1905.15 | 1931.68 | 1917.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 1911.00 | 1931.47 | 1917.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 1926.00 | 1931.47 | 1917.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 1884.80 | 1931.25 | 1917.67 | SL hit (close<static) qty=1.00 sl=1903.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 11:15:00 | 1848.60 | 1909.09 | 1909.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 13:15:00 | 1843.00 | 1907.82 | 1908.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 11:15:00 | 1859.05 | 1845.45 | 1869.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-29 11:45:00 | 1860.60 | 1845.45 | 1869.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 1868.00 | 1846.00 | 1869.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 15:00:00 | 1868.00 | 1846.00 | 1869.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 15:15:00 | 1872.35 | 1846.26 | 1869.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:15:00 | 1868.00 | 1846.26 | 1869.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 1884.30 | 1846.64 | 1869.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 09:30:00 | 1853.55 | 1848.50 | 1869.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 11:15:00 | 1891.95 | 1848.10 | 1868.30 | SL hit (close>static) qty=1.00 sl=1889.95 alert=retest2 |

### Cycle 4 — BUY (started 2024-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 11:15:00 | 1977.85 | 1880.49 | 1880.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 09:15:00 | 1996.00 | 1902.62 | 1892.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 2005.55 | 2008.28 | 1960.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-13 11:45:00 | 2011.60 | 2008.28 | 1960.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 1977.05 | 2008.21 | 1963.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:30:00 | 1960.25 | 2008.21 | 1963.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 1967.05 | 2007.42 | 1965.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 14:30:00 | 1956.25 | 2007.42 | 1965.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 15:15:00 | 1972.00 | 2007.07 | 1965.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 09:15:00 | 1936.60 | 2007.07 | 1965.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 1917.90 | 2006.18 | 1965.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 1917.90 | 2006.18 | 1965.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1887.00 | 2005.00 | 1964.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:00:00 | 1887.00 | 2005.00 | 1964.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 1917.80 | 2000.90 | 1963.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 1917.80 | 2000.90 | 1963.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 1915.00 | 2000.04 | 1963.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:00:00 | 1915.00 | 2000.04 | 1963.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 1981.25 | 1986.24 | 1960.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 11:45:00 | 1986.10 | 1986.19 | 1960.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 13:15:00 | 1986.60 | 1986.11 | 1960.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 09:15:00 | 1983.15 | 2002.81 | 1979.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 12:45:00 | 1984.55 | 2002.20 | 1980.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 1995.75 | 2002.14 | 1980.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 09:30:00 | 1999.20 | 2001.79 | 1980.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 11:00:00 | 1999.45 | 2001.77 | 1980.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 15:15:00 | 1999.00 | 2001.46 | 1980.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 09:15:00 | 1952.05 | 2000.95 | 1980.68 | SL hit (close<static) qty=1.00 sl=1980.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 2038.00 | 2196.86 | 2197.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1928.95 | 2194.19 | 2196.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 1971.00 | 1958.78 | 2022.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 10:00:00 | 1971.00 | 1958.78 | 2022.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 2017.00 | 1963.43 | 2020.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:45:00 | 2005.70 | 1963.83 | 2020.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 2005.20 | 1966.60 | 2020.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:15:00 | 2009.15 | 1968.07 | 2020.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:45:00 | 2008.50 | 1968.47 | 2020.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 2016.60 | 1970.07 | 2020.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 2018.00 | 1970.07 | 2020.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 2008.00 | 1970.88 | 2019.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 13:30:00 | 2005.80 | 1971.54 | 2019.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:45:00 | 2005.95 | 1971.88 | 2019.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 1989.50 | 1972.27 | 2019.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 1908.69 | 1972.04 | 2015.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 1908.07 | 1972.04 | 2015.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1905.41 | 1970.40 | 2013.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1904.94 | 1970.40 | 2013.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1905.51 | 1970.40 | 2013.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1905.65 | 1970.40 | 2013.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1890.02 | 1970.40 | 2013.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-22 14:15:00 | 1805.13 | 1917.71 | 1966.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 1511.80 | 1377.05 | 1376.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 1528.50 | 1382.68 | 1379.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 1486.30 | 1493.86 | 1459.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 1486.30 | 1493.86 | 1459.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1469.40 | 1490.45 | 1466.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:45:00 | 1463.80 | 1490.45 | 1466.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1478.30 | 1493.85 | 1473.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1478.30 | 1493.85 | 1473.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1473.50 | 1493.65 | 1473.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1463.10 | 1493.65 | 1473.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1459.70 | 1493.31 | 1473.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1459.70 | 1493.31 | 1473.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1455.00 | 1492.93 | 1472.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 1456.60 | 1492.93 | 1472.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1467.70 | 1491.26 | 1472.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 1469.70 | 1491.26 | 1472.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1461.50 | 1490.96 | 1472.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 1461.50 | 1490.96 | 1472.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1471.70 | 1490.14 | 1472.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 1474.10 | 1490.14 | 1472.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1471.50 | 1489.95 | 1472.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1472.10 | 1489.95 | 1472.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1459.70 | 1489.65 | 1472.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 1459.70 | 1489.65 | 1472.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1456.00 | 1489.32 | 1472.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 1458.00 | 1489.32 | 1472.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1419.60 | 1459.50 | 1459.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1403.90 | 1456.51 | 1458.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1438.60 | 1402.69 | 1426.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 1438.60 | 1402.69 | 1426.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1438.60 | 1402.69 | 1426.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1438.60 | 1402.69 | 1426.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1435.30 | 1403.01 | 1426.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:30:00 | 1436.10 | 1403.01 | 1426.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1427.20 | 1403.47 | 1426.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 1426.20 | 1403.47 | 1426.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1420.90 | 1403.65 | 1426.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1417.70 | 1403.81 | 1426.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 12:15:00 | 1432.10 | 1404.79 | 1426.61 | SL hit (close>static) qty=1.00 sl=1431.80 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 1451.50 | 1431.35 | 1431.33 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1404.00 | 1431.45 | 1431.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1400.50 | 1431.15 | 1431.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 13:15:00 | 1417.50 | 1411.74 | 1420.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 13:45:00 | 1413.70 | 1411.74 | 1420.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1417.80 | 1411.80 | 1420.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 1420.00 | 1411.80 | 1420.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1421.30 | 1411.89 | 1420.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1415.90 | 1411.89 | 1420.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1411.00 | 1411.88 | 1420.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:45:00 | 1405.80 | 1411.81 | 1420.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 1404.80 | 1411.74 | 1420.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:00:00 | 1402.90 | 1411.60 | 1420.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 1398.00 | 1410.85 | 1419.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 1422.40 | 1410.91 | 1419.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:45:00 | 1422.30 | 1410.91 | 1419.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 1431.00 | 1411.11 | 1419.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 1431.00 | 1411.11 | 1419.24 | SL hit (close>static) qty=1.00 sl=1427.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 1453.00 | 1424.47 | 1424.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 1457.30 | 1425.97 | 1425.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 1468.50 | 1482.73 | 1458.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:00:00 | 1468.50 | 1482.73 | 1458.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1465.20 | 1482.56 | 1458.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 1460.90 | 1482.56 | 1458.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1466.40 | 1482.40 | 1458.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 1457.30 | 1482.40 | 1458.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1443.00 | 1481.40 | 1458.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1443.00 | 1481.40 | 1458.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1450.00 | 1481.08 | 1458.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1438.70 | 1481.08 | 1458.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1453.80 | 1476.74 | 1457.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:30:00 | 1458.10 | 1476.59 | 1457.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 1450.00 | 1475.59 | 1457.89 | SL hit (close<static) qty=1.00 sl=1450.30 alert=retest2 |

### Cycle 11 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 1405.70 | 1449.37 | 1449.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 1395.90 | 1436.24 | 1442.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 13:15:00 | 1426.70 | 1420.72 | 1432.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 13:45:00 | 1426.40 | 1420.72 | 1432.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1435.30 | 1420.86 | 1432.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1435.30 | 1420.86 | 1432.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1435.00 | 1421.00 | 1432.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1444.40 | 1421.00 | 1432.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 1440.30 | 1439.64 | 1440.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 1437.00 | 1439.64 | 1440.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1441.00 | 1439.65 | 1440.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:30:00 | 1434.40 | 1439.54 | 1440.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 12:45:00 | 1436.30 | 1439.36 | 1440.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 14:15:00 | 1432.40 | 1439.34 | 1440.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 1428.40 | 1439.31 | 1440.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1430.40 | 1439.23 | 1440.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 1434.00 | 1439.23 | 1440.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 1433.30 | 1439.17 | 1440.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:30:00 | 1440.70 | 1439.17 | 1440.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 1431.50 | 1439.09 | 1440.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 1428.10 | 1438.98 | 1440.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 1447.30 | 1439.03 | 1440.39 | SL hit (close>static) qty=1.00 sl=1442.70 alert=retest2 |

### Cycle 12 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 1472.60 | 1441.94 | 1441.82 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 1416.50 | 1441.52 | 1441.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 1404.10 | 1441.15 | 1441.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 1428.80 | 1426.65 | 1433.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 1428.80 | 1426.65 | 1433.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1428.80 | 1426.65 | 1433.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 1428.80 | 1426.65 | 1433.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1445.50 | 1426.84 | 1433.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1445.50 | 1426.84 | 1433.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1445.00 | 1427.02 | 1433.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:15:00 | 1443.40 | 1427.02 | 1433.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1465.80 | 1434.36 | 1436.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:30:00 | 1462.40 | 1434.62 | 1436.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:30:00 | 1459.20 | 1435.08 | 1437.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1486.70 | 1436.32 | 1437.63 | SL hit (close>static) qty=1.00 sl=1468.90 alert=retest2 |

### Cycle 14 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 1506.40 | 1439.50 | 1439.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 1510.80 | 1453.65 | 1446.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 15:15:00 | 1610.00 | 1610.66 | 1556.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:15:00 | 1615.50 | 1610.66 | 1556.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1563.50 | 1618.74 | 1569.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 1563.50 | 1618.74 | 1569.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 1554.40 | 1618.10 | 1569.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 1554.40 | 1618.10 | 1569.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 1556.70 | 1615.77 | 1569.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 1556.70 | 1615.77 | 1569.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 1559.90 | 1615.22 | 1569.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 1585.70 | 1615.22 | 1569.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:00:00 | 1568.30 | 1614.33 | 1569.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 1468.20 | 1613.52 | 1576.05 | SL hit (close<static) qty=1.00 sl=1551.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-07-10 13:30:00 | 1790.20 | 2023-07-31 09:15:00 | 1969.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-11 09:15:00 | 1800.50 | 2023-07-31 09:15:00 | 1969.22 | TARGET_HIT | 1.00 | 9.37% |
| BUY | retest2 | 2023-07-12 09:45:00 | 1790.20 | 2023-07-31 09:15:00 | 1966.69 | TARGET_HIT | 1.00 | 9.86% |
| BUY | retest2 | 2023-07-12 10:30:00 | 1787.90 | 2023-07-31 10:15:00 | 1980.55 | TARGET_HIT | 1.00 | 10.78% |
| BUY | retest2 | 2023-09-04 09:15:00 | 1927.65 | 2023-09-04 11:15:00 | 1895.65 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-09-05 10:15:00 | 1917.10 | 2023-09-05 12:15:00 | 1899.10 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-09-05 15:00:00 | 1916.35 | 2023-09-06 10:15:00 | 1897.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-09-11 09:30:00 | 1915.90 | 2023-09-12 09:15:00 | 1883.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2023-09-27 09:30:00 | 1932.50 | 2023-09-28 14:15:00 | 1890.05 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2023-09-27 10:30:00 | 1930.75 | 2023-09-28 14:15:00 | 1890.05 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2023-10-11 09:15:00 | 1937.35 | 2023-10-19 09:15:00 | 1851.00 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2023-10-11 11:15:00 | 1933.30 | 2023-10-19 09:15:00 | 1851.00 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2023-11-13 11:45:00 | 1866.05 | 2023-11-15 09:15:00 | 1894.40 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2023-11-13 13:45:00 | 1870.20 | 2023-11-15 09:15:00 | 1894.40 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-11-13 14:15:00 | 1870.65 | 2023-11-15 09:15:00 | 1894.40 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2023-11-13 15:15:00 | 1868.00 | 2023-11-15 09:15:00 | 1894.40 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-11-15 11:30:00 | 1888.00 | 2023-11-17 09:15:00 | 1910.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2023-11-16 15:00:00 | 1889.25 | 2023-11-17 09:15:00 | 1910.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-12-20 09:15:00 | 1926.00 | 2023-12-20 13:15:00 | 1884.80 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2023-12-22 12:00:00 | 1914.25 | 2023-12-28 10:15:00 | 1902.50 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-12-22 12:30:00 | 1916.45 | 2023-12-28 10:15:00 | 1902.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-12-27 14:00:00 | 1913.30 | 2023-12-28 10:15:00 | 1902.50 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-01-31 09:30:00 | 1853.55 | 2024-02-01 11:15:00 | 1891.95 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-02-05 14:15:00 | 1861.25 | 2024-02-06 10:15:00 | 1893.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-02-09 09:45:00 | 1861.95 | 2024-02-12 10:15:00 | 1895.70 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-03-26 11:45:00 | 1986.10 | 2024-04-19 09:15:00 | 1952.05 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-03-26 13:15:00 | 1986.60 | 2024-04-19 09:15:00 | 1952.05 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-04-16 09:15:00 | 1983.15 | 2024-04-19 09:15:00 | 1952.05 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-04-16 12:45:00 | 1984.55 | 2024-04-19 11:15:00 | 1949.15 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-04-18 09:30:00 | 1999.20 | 2024-04-19 11:15:00 | 1949.15 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-04-18 11:00:00 | 1999.45 | 2024-04-19 11:15:00 | 1949.15 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-04-18 15:15:00 | 1999.00 | 2024-04-19 11:15:00 | 1949.15 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-04-25 09:15:00 | 2008.45 | 2024-05-13 14:15:00 | 2209.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 14:15:00 | 2227.30 | 2024-07-29 11:15:00 | 2204.50 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-29 09:15:00 | 2233.00 | 2024-07-29 11:15:00 | 2204.50 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-07-29 09:45:00 | 2233.50 | 2024-07-29 11:15:00 | 2204.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-07-29 10:30:00 | 2233.20 | 2024-07-29 11:15:00 | 2204.50 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-09-25 10:45:00 | 2005.70 | 2024-10-03 13:15:00 | 1908.69 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2024-09-26 09:15:00 | 2005.20 | 2024-10-03 13:15:00 | 1908.07 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2024-09-26 12:15:00 | 2009.15 | 2024-10-04 09:15:00 | 1905.41 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2024-09-26 12:45:00 | 2008.50 | 2024-10-04 09:15:00 | 1904.94 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2024-09-27 13:30:00 | 2005.80 | 2024-10-04 09:15:00 | 1905.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 14:45:00 | 2005.95 | 2024-10-04 09:15:00 | 1905.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1989.50 | 2024-10-04 09:15:00 | 1890.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 10:45:00 | 2005.70 | 2024-10-22 14:15:00 | 1805.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-26 09:15:00 | 2005.20 | 2024-10-22 14:15:00 | 1804.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-26 12:15:00 | 2009.15 | 2024-10-22 14:15:00 | 1808.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-26 12:45:00 | 2008.50 | 2024-10-22 14:15:00 | 1807.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-27 13:30:00 | 2005.80 | 2024-10-22 14:15:00 | 1805.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-27 14:45:00 | 2005.95 | 2024-10-22 14:15:00 | 1805.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1989.50 | 2024-10-22 14:15:00 | 1790.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-22 09:15:00 | 1417.70 | 2025-08-22 12:15:00 | 1432.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-08-25 09:15:00 | 1417.30 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1416.40 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-02 11:00:00 | 1418.60 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-04 09:15:00 | 1413.80 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-09-04 10:00:00 | 1411.00 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-10-08 10:45:00 | 1405.80 | 2025-10-10 12:15:00 | 1431.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-10-08 12:00:00 | 1404.80 | 2025-10-10 12:15:00 | 1431.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-10-08 14:00:00 | 1402.90 | 2025-10-10 12:15:00 | 1431.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-10-09 15:15:00 | 1398.00 | 2025-10-10 12:15:00 | 1431.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-10-14 09:30:00 | 1419.10 | 2025-10-15 09:15:00 | 1436.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-10-14 15:00:00 | 1417.80 | 2025-10-15 09:15:00 | 1436.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-27 13:30:00 | 1418.10 | 2025-10-27 14:15:00 | 1435.70 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-20 11:30:00 | 1458.10 | 2025-11-21 10:15:00 | 1450.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-11-24 15:00:00 | 1474.30 | 2025-11-28 09:15:00 | 1446.30 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-11-25 11:00:00 | 1460.90 | 2025-11-28 09:15:00 | 1446.30 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-05 13:30:00 | 1458.40 | 2025-12-08 09:15:00 | 1433.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-12 10:30:00 | 1434.40 | 2026-01-14 09:15:00 | 1447.30 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-12 12:45:00 | 1436.30 | 2026-01-14 10:15:00 | 1467.30 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-01-12 14:15:00 | 1432.40 | 2026-01-14 10:15:00 | 1467.30 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-01-13 09:30:00 | 1428.40 | 2026-01-14 10:15:00 | 1467.30 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-01-13 13:45:00 | 1428.10 | 2026-01-14 10:15:00 | 1467.30 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-02-01 14:30:00 | 1462.40 | 2026-02-03 09:15:00 | 1486.70 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-02 09:30:00 | 1459.20 | 2026-02-03 09:15:00 | 1486.70 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-03-24 09:15:00 | 1585.70 | 2026-04-02 09:15:00 | 1468.20 | STOP_HIT | 1.00 | -7.41% |
| BUY | retest2 | 2026-03-24 11:00:00 | 1568.30 | 2026-04-02 09:15:00 | 1468.20 | STOP_HIT | 1.00 | -6.38% |
| BUY | retest2 | 2026-04-02 15:15:00 | 1570.00 | 2026-04-06 09:15:00 | 1496.20 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1587.00 | 2026-04-08 11:15:00 | 1546.90 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1599.00 | 2026-04-16 11:15:00 | 1556.80 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-04-16 14:45:00 | 1579.50 | 2026-04-24 12:15:00 | 1554.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-04-23 11:45:00 | 1566.90 | 2026-04-24 12:15:00 | 1554.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-04-27 09:15:00 | 1569.40 | 2026-04-27 12:15:00 | 1561.80 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2026-05-07 11:45:00 | 1592.00 | 2026-05-07 14:15:00 | 1569.70 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-05-08 10:30:00 | 1589.40 | 2026-05-08 14:15:00 | 1569.10 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-05-08 11:30:00 | 1588.70 | 2026-05-08 14:15:00 | 1569.10 | STOP_HIT | 1.00 | -1.23% |
