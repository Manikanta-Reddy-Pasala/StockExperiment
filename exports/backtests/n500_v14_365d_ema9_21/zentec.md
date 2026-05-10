# Zen Technologies Ltd. (ZENTEC)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1626.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 76 |
| ALERT1 | 51 |
| ALERT2 | 51 |
| ALERT2_SKIP | 27 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 68 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 55
- **Target hits / Stop hits / Partials:** 6 / 64 / 6
- **Avg / median % per leg:** 0.00% / -1.19%
- **Sum % (uncompounded):** 0.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 7 | 23.3% | 4 | 26 | 0 | 0.02% | 0.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 7 | 23.3% | 4 | 26 | 0 | 0.02% | 0.5% |
| SELL (all) | 46 | 14 | 30.4% | 2 | 38 | 6 | -0.01% | -0.5% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 1 | 1 | 0 | 3.85% | 7.7% |
| SELL @ 3rd Alert (retest2) | 44 | 13 | 29.5% | 1 | 37 | 6 | -0.19% | -8.2% |
| retest1 (combined) | 2 | 1 | 50.0% | 1 | 1 | 0 | 3.85% | 7.7% |
| retest2 (combined) | 74 | 20 | 27.0% | 5 | 63 | 6 | -0.10% | -7.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 2084.80 | 2160.13 | 2166.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 09:15:00 | 2032.30 | 2095.19 | 2127.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 2038.00 | 2035.95 | 2074.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:45:00 | 2045.50 | 2035.95 | 2074.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1967.00 | 2015.47 | 2044.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:00:00 | 1962.30 | 1996.60 | 2030.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:30:00 | 1961.30 | 1988.40 | 2023.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1945.10 | 1975.24 | 2011.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 1956.00 | 1968.43 | 2001.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1989.60 | 1941.07 | 1967.22 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1864.18 | 1941.07 | 1967.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1989.60 | 1941.07 | 1967.22 | SL hit (close>static) qty=0.50 sl=1941.07 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1863.23 | 1941.07 | 1967.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1989.60 | 1941.07 | 1967.22 | SL hit (close>static) qty=0.50 sl=1941.07 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1847.84 | 1941.07 | 1967.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1989.60 | 1941.07 | 1967.22 | SL hit (close>static) qty=0.50 sl=1941.07 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1858.20 | 1941.07 | 1967.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1989.60 | 1941.07 | 1967.22 | SL hit (close>static) qty=0.50 sl=1941.07 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-06-13 09:45:00 | 1980.80 | 1941.07 | 1967.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 1991.80 | 1951.22 | 1969.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 1991.40 | 1951.22 | 1969.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1968.90 | 1960.38 | 1969.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 1968.90 | 1960.38 | 1969.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1968.30 | 1961.96 | 1969.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:30:00 | 1965.00 | 1961.96 | 1969.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1987.00 | 1966.97 | 1971.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 1979.00 | 1966.97 | 1971.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1951.10 | 1963.80 | 1969.38 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1975.00 | 1970.72 | 1970.69 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 10:15:00 | 1955.80 | 1968.24 | 1969.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 1938.00 | 1962.19 | 1966.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 1966.40 | 1954.85 | 1960.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1966.40 | 1954.85 | 1960.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1966.40 | 1954.85 | 1960.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 1971.80 | 1954.85 | 1960.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1949.00 | 1953.68 | 1959.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 1934.10 | 1952.34 | 1957.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 1935.00 | 1948.88 | 1955.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:00:00 | 1932.00 | 1943.60 | 1951.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1979.10 | 1917.97 | 1921.11 | SL hit (close>static) qty=1.00 sl=1966.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1979.10 | 1917.97 | 1921.11 | SL hit (close>static) qty=1.00 sl=1966.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1979.10 | 1917.97 | 1921.11 | SL hit (close>static) qty=1.00 sl=1966.40 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1994.60 | 1933.30 | 1927.79 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 1922.40 | 1942.90 | 1942.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 09:15:00 | 1883.70 | 1926.54 | 1934.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 10:15:00 | 1909.70 | 1903.17 | 1915.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 10:15:00 | 1909.70 | 1903.17 | 1915.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 1909.70 | 1903.17 | 1915.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 1909.70 | 1903.17 | 1915.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1911.00 | 1904.73 | 1914.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 1911.00 | 1904.73 | 1914.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 1900.20 | 1903.83 | 1913.42 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 15:15:00 | 1929.80 | 1914.43 | 1913.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 1968.50 | 1925.24 | 1918.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 11:15:00 | 1973.90 | 1974.38 | 1955.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 12:00:00 | 1973.90 | 1974.38 | 1955.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1953.50 | 1970.19 | 1960.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1953.50 | 1970.19 | 1960.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1954.80 | 1967.11 | 1959.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 1954.80 | 1967.11 | 1959.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 1936.60 | 1952.91 | 1954.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 1923.60 | 1938.82 | 1946.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 1961.90 | 1939.76 | 1944.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 1961.90 | 1939.76 | 1944.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1961.90 | 1939.76 | 1944.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 1972.00 | 1939.76 | 1944.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1953.20 | 1942.45 | 1945.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 1942.00 | 1943.74 | 1945.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 12:15:00 | 1941.10 | 1943.74 | 1945.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 1942.20 | 1930.08 | 1933.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 15:15:00 | 1953.00 | 1934.23 | 1933.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 15:15:00 | 1953.00 | 1934.23 | 1933.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 15:15:00 | 1953.00 | 1934.23 | 1933.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 1953.00 | 1934.23 | 1933.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 1982.10 | 1943.80 | 1937.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 1952.90 | 1954.31 | 1945.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 15:00:00 | 1952.90 | 1954.31 | 1945.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1912.00 | 1946.72 | 1944.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1912.00 | 1946.72 | 1944.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 1911.00 | 1939.58 | 1941.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 1907.00 | 1933.06 | 1937.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1848.60 | 1832.32 | 1854.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1848.60 | 1832.32 | 1854.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1848.60 | 1832.32 | 1854.13 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 14:15:00 | 1853.40 | 1842.47 | 1842.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 1872.00 | 1851.10 | 1846.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 1850.10 | 1850.90 | 1846.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 10:15:00 | 1850.10 | 1850.90 | 1846.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1850.10 | 1850.90 | 1846.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 1851.70 | 1850.90 | 1846.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1850.30 | 1850.78 | 1846.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 1845.10 | 1850.78 | 1846.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1847.30 | 1850.09 | 1846.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 1844.00 | 1850.09 | 1846.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1854.50 | 1850.97 | 1847.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 1856.70 | 1850.97 | 1847.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 1859.50 | 1858.87 | 1852.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:00:00 | 1856.60 | 1858.42 | 1853.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 1900.80 | 1864.19 | 1857.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1876.00 | 1882.25 | 1874.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:00:00 | 1880.80 | 1881.96 | 1875.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 14:00:00 | 1883.10 | 1881.89 | 1876.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1870.00 | 1876.94 | 1877.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1870.00 | 1876.94 | 1877.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1870.00 | 1876.94 | 1877.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1870.00 | 1876.94 | 1877.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1870.00 | 1876.94 | 1877.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1870.00 | 1876.94 | 1877.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 1870.00 | 1876.94 | 1877.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 1861.00 | 1873.75 | 1875.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 1518.00 | 1495.74 | 1543.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:45:00 | 1528.70 | 1495.74 | 1543.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1511.00 | 1522.18 | 1537.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:45:00 | 1483.50 | 1496.69 | 1512.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 1409.33 | 1441.51 | 1465.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 1459.00 | 1435.13 | 1450.12 | SL hit (close>ema200) qty=0.50 sl=1435.13 alert=retest2 |

### Cycle 12 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1441.10 | 1415.76 | 1415.08 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 1419.40 | 1425.89 | 1426.26 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 1437.30 | 1428.05 | 1427.09 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 1419.80 | 1425.30 | 1426.00 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 1461.70 | 1431.54 | 1428.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 14:15:00 | 1492.60 | 1452.82 | 1440.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1490.20 | 1501.26 | 1478.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1490.20 | 1501.26 | 1478.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1490.20 | 1501.26 | 1478.93 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 1461.80 | 1481.55 | 1481.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 15:15:00 | 1450.00 | 1475.24 | 1479.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 14:15:00 | 1461.20 | 1458.94 | 1468.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 15:00:00 | 1461.20 | 1458.94 | 1468.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1477.50 | 1463.11 | 1468.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1486.10 | 1463.11 | 1468.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1478.60 | 1466.21 | 1469.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1478.60 | 1466.21 | 1469.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 1501.00 | 1475.63 | 1473.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 1523.90 | 1499.82 | 1487.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1497.30 | 1499.32 | 1488.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 1497.30 | 1499.32 | 1488.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1492.00 | 1498.69 | 1490.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:15:00 | 1488.90 | 1498.69 | 1490.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1481.20 | 1495.19 | 1489.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:45:00 | 1483.10 | 1495.19 | 1489.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1484.50 | 1493.05 | 1489.33 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 1481.10 | 1486.48 | 1487.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 09:15:00 | 1478.00 | 1484.78 | 1486.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 12:15:00 | 1482.90 | 1481.60 | 1484.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 1482.90 | 1481.60 | 1484.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1482.90 | 1481.60 | 1484.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:45:00 | 1486.40 | 1481.60 | 1484.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1470.00 | 1477.15 | 1481.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 1458.00 | 1472.46 | 1476.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 1464.20 | 1455.39 | 1459.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 1479.40 | 1462.95 | 1462.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 1479.40 | 1462.95 | 1462.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 1479.40 | 1462.95 | 1462.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 1499.30 | 1470.22 | 1465.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1477.90 | 1478.07 | 1470.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 1477.90 | 1478.07 | 1470.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1472.40 | 1477.31 | 1471.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:00:00 | 1472.40 | 1477.31 | 1471.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1463.50 | 1474.55 | 1470.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 1463.50 | 1474.55 | 1470.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1463.70 | 1472.38 | 1469.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:45:00 | 1463.90 | 1472.38 | 1469.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 1463.80 | 1467.88 | 1468.13 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 1470.10 | 1468.49 | 1468.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 1491.50 | 1480.49 | 1475.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 1488.50 | 1489.41 | 1481.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:00:00 | 1488.50 | 1489.41 | 1481.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1483.90 | 1490.14 | 1484.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1475.20 | 1490.14 | 1484.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1480.70 | 1488.25 | 1483.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1480.70 | 1488.25 | 1483.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1479.60 | 1486.52 | 1483.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:45:00 | 1476.00 | 1486.52 | 1483.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 1470.00 | 1480.34 | 1481.22 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1493.80 | 1482.10 | 1481.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1533.40 | 1493.96 | 1487.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1591.00 | 1597.87 | 1563.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 15:15:00 | 1602.00 | 1605.48 | 1599.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1602.00 | 1605.48 | 1599.57 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1568.40 | 1592.11 | 1594.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 1540.70 | 1575.55 | 1585.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 15:15:00 | 1505.30 | 1502.47 | 1525.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:15:00 | 1503.40 | 1502.47 | 1525.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1463.70 | 1443.43 | 1461.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 1466.20 | 1443.43 | 1461.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1482.30 | 1451.21 | 1463.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 1482.30 | 1451.21 | 1463.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1482.60 | 1457.48 | 1465.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:45:00 | 1487.20 | 1457.48 | 1465.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1469.70 | 1461.27 | 1465.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 1469.70 | 1461.27 | 1465.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1466.60 | 1462.34 | 1465.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:15:00 | 1475.00 | 1462.34 | 1465.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1475.00 | 1464.87 | 1466.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1491.20 | 1464.87 | 1466.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1485.00 | 1468.90 | 1468.24 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1447.20 | 1468.66 | 1470.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 1441.00 | 1454.55 | 1461.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 1446.70 | 1444.93 | 1452.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 1446.70 | 1444.93 | 1452.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1410.40 | 1413.68 | 1424.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 1428.50 | 1413.68 | 1424.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1427.00 | 1410.97 | 1418.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 1427.00 | 1410.97 | 1418.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1419.90 | 1412.76 | 1418.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1420.00 | 1412.76 | 1418.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1401.90 | 1410.59 | 1417.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:30:00 | 1397.40 | 1407.07 | 1414.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 1378.00 | 1372.19 | 1371.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1378.00 | 1372.19 | 1371.72 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 1363.50 | 1370.11 | 1371.00 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 1375.00 | 1371.59 | 1371.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1400.60 | 1377.39 | 1374.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1382.40 | 1387.67 | 1382.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 13:15:00 | 1382.40 | 1387.67 | 1382.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1382.40 | 1387.67 | 1382.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 1382.40 | 1387.67 | 1382.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1387.30 | 1387.60 | 1382.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1390.90 | 1387.08 | 1382.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 1329.90 | 1382.21 | 1384.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 1329.90 | 1382.21 | 1384.93 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 1359.00 | 1352.62 | 1352.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 1377.60 | 1357.62 | 1354.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1364.90 | 1369.30 | 1363.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 1364.90 | 1369.30 | 1363.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1364.90 | 1369.30 | 1363.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1364.90 | 1369.30 | 1363.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1358.20 | 1367.08 | 1362.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 1359.50 | 1367.08 | 1362.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 1363.90 | 1366.44 | 1363.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1401.80 | 1362.43 | 1361.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 10:30:00 | 1370.50 | 1395.19 | 1393.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1369.90 | 1390.13 | 1391.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1369.90 | 1390.13 | 1391.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 1369.90 | 1390.13 | 1391.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1349.40 | 1375.44 | 1383.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1362.00 | 1361.38 | 1371.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:15:00 | 1359.60 | 1361.38 | 1371.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1363.70 | 1361.84 | 1370.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 1368.50 | 1361.84 | 1370.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1371.80 | 1363.83 | 1370.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1365.50 | 1363.83 | 1370.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1366.40 | 1364.35 | 1370.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:30:00 | 1361.70 | 1363.04 | 1369.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 1363.90 | 1364.00 | 1368.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 1357.00 | 1364.00 | 1368.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1382.80 | 1366.64 | 1368.80 | SL hit (close>static) qty=1.00 sl=1371.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1382.80 | 1366.64 | 1368.80 | SL hit (close>static) qty=1.00 sl=1371.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1382.80 | 1366.64 | 1368.80 | SL hit (close>static) qty=1.00 sl=1371.90 alert=retest2 |

### Cycle 34 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 1389.50 | 1371.21 | 1370.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 1411.00 | 1395.25 | 1387.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1392.30 | 1396.41 | 1390.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:45:00 | 1394.80 | 1396.41 | 1390.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1385.70 | 1394.27 | 1389.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 1385.70 | 1394.27 | 1389.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1371.10 | 1389.64 | 1388.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1371.10 | 1389.64 | 1388.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 1368.00 | 1385.31 | 1386.43 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 1433.40 | 1394.93 | 1390.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 1452.90 | 1406.52 | 1396.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 13:15:00 | 1450.10 | 1452.37 | 1433.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 14:00:00 | 1450.10 | 1452.37 | 1433.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1407.00 | 1442.65 | 1433.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1407.00 | 1442.65 | 1433.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1407.90 | 1435.70 | 1431.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1410.00 | 1435.70 | 1431.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1406.30 | 1425.71 | 1427.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1398.80 | 1417.59 | 1423.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1452.80 | 1411.05 | 1412.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1452.80 | 1411.05 | 1412.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1452.80 | 1411.05 | 1412.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1452.80 | 1411.05 | 1412.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1434.00 | 1415.64 | 1414.64 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 1404.00 | 1418.80 | 1420.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 1377.70 | 1406.95 | 1414.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1390.30 | 1387.48 | 1398.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:30:00 | 1395.30 | 1387.48 | 1398.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1391.40 | 1388.26 | 1397.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 1391.40 | 1388.26 | 1397.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 1395.20 | 1389.65 | 1397.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 1395.20 | 1389.65 | 1397.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 1394.40 | 1390.60 | 1397.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:30:00 | 1397.00 | 1390.60 | 1397.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1401.80 | 1392.84 | 1397.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 1401.80 | 1392.84 | 1397.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1397.70 | 1393.81 | 1397.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1397.70 | 1393.81 | 1397.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1396.50 | 1394.35 | 1397.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 1433.40 | 1394.35 | 1397.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1415.00 | 1398.48 | 1399.10 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 1414.70 | 1401.72 | 1400.52 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1401.90 | 1405.97 | 1406.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 13:15:00 | 1398.90 | 1403.82 | 1405.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 1405.50 | 1404.16 | 1405.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 1405.50 | 1404.16 | 1405.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1405.50 | 1404.16 | 1405.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 1405.50 | 1404.16 | 1405.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1403.60 | 1404.05 | 1405.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1410.50 | 1404.05 | 1405.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1400.60 | 1403.36 | 1404.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 13:15:00 | 1396.90 | 1401.13 | 1403.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 1397.00 | 1400.30 | 1402.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:45:00 | 1396.40 | 1400.24 | 1402.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 1392.20 | 1400.24 | 1402.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1380.00 | 1394.91 | 1399.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:45:00 | 1375.00 | 1386.24 | 1391.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:15:00 | 1373.90 | 1384.19 | 1390.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 1374.20 | 1380.79 | 1387.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 10:15:00 | 1375.50 | 1379.03 | 1384.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1380.80 | 1379.39 | 1384.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1381.40 | 1379.39 | 1384.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1401.00 | 1381.28 | 1382.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 1386.80 | 1381.28 | 1382.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1368.50 | 1384.74 | 1385.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 1365.20 | 1380.83 | 1383.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 15:15:00 | 1367.60 | 1367.16 | 1374.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:15:00 | 1336.20 | 1367.16 | 1374.43 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1367.00 | 1351.91 | 1359.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1367.00 | 1351.91 | 1359.41 | SL hit (close>ema400) qty=1.00 sl=1359.41 alert=retest1 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 1342.50 | 1351.53 | 1358.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 1347.00 | 1351.49 | 1354.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 1374.00 | 1357.80 | 1356.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 1374.00 | 1357.80 | 1356.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1374.00 | 1357.80 | 1356.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 1392.00 | 1368.45 | 1362.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 1372.10 | 1372.32 | 1366.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:45:00 | 1373.90 | 1372.32 | 1366.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1368.80 | 1371.99 | 1368.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 1368.80 | 1371.99 | 1368.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1367.50 | 1371.09 | 1368.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 1359.10 | 1371.09 | 1368.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1359.10 | 1368.69 | 1367.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1362.40 | 1368.69 | 1367.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1356.90 | 1366.33 | 1366.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1356.90 | 1366.33 | 1366.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1356.10 | 1364.29 | 1365.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 1351.20 | 1358.99 | 1361.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 1363.00 | 1355.64 | 1358.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 1363.00 | 1355.64 | 1358.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1363.00 | 1355.64 | 1358.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 1363.50 | 1355.64 | 1358.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1360.30 | 1356.57 | 1359.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1352.30 | 1356.57 | 1359.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1365.90 | 1354.24 | 1353.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1365.90 | 1354.24 | 1353.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1373.10 | 1360.05 | 1356.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 1389.30 | 1394.35 | 1384.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 1389.30 | 1394.35 | 1384.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1390.00 | 1394.79 | 1390.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1401.00 | 1394.79 | 1390.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1395.00 | 1394.83 | 1390.71 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 1378.90 | 1388.65 | 1389.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 1363.80 | 1379.30 | 1384.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 1388.90 | 1379.04 | 1383.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 1388.90 | 1379.04 | 1383.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1388.90 | 1379.04 | 1383.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1388.90 | 1379.04 | 1383.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1394.40 | 1382.11 | 1384.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 1374.90 | 1382.11 | 1384.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1360.20 | 1360.34 | 1369.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 1349.20 | 1359.07 | 1367.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 1373.00 | 1360.50 | 1364.79 | SL hit (close>static) qty=1.00 sl=1370.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1384.20 | 1365.16 | 1362.96 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1351.70 | 1363.49 | 1364.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1346.20 | 1358.26 | 1361.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 1331.00 | 1330.38 | 1341.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 1320.00 | 1327.36 | 1334.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1320.00 | 1327.36 | 1334.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:00:00 | 1306.90 | 1320.91 | 1329.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1241.56 | 1301.90 | 1317.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 1247.70 | 1246.59 | 1265.84 | SL hit (close>ema200) qty=0.50 sl=1246.59 alert=retest2 |

### Cycle 50 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 1322.10 | 1269.21 | 1265.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 15:15:00 | 1336.60 | 1311.06 | 1297.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1302.00 | 1309.24 | 1297.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 1302.00 | 1309.24 | 1297.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1302.00 | 1309.24 | 1297.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 1302.00 | 1309.24 | 1297.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1295.20 | 1306.44 | 1297.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 1295.20 | 1306.44 | 1297.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1288.10 | 1302.77 | 1296.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 1288.10 | 1302.77 | 1296.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 1284.60 | 1299.13 | 1295.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 1284.60 | 1299.13 | 1295.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1291.70 | 1295.27 | 1294.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1279.20 | 1295.27 | 1294.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1294.60 | 1295.14 | 1294.27 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1279.00 | 1291.91 | 1292.88 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 1318.50 | 1296.10 | 1293.91 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 1281.30 | 1296.35 | 1297.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 1272.90 | 1291.66 | 1295.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1292.00 | 1286.39 | 1290.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 1292.00 | 1286.39 | 1290.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1292.00 | 1286.39 | 1290.44 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 1309.00 | 1290.02 | 1288.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 1339.80 | 1303.81 | 1295.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1380.20 | 1401.50 | 1378.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1380.20 | 1401.50 | 1378.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1380.20 | 1401.50 | 1378.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1380.20 | 1401.50 | 1378.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1346.00 | 1390.40 | 1375.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1356.80 | 1390.40 | 1375.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1320.80 | 1376.48 | 1370.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 1320.80 | 1376.48 | 1370.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1317.00 | 1357.23 | 1362.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 1312.90 | 1323.26 | 1332.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 1333.60 | 1325.33 | 1332.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 1333.60 | 1325.33 | 1332.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1333.60 | 1325.33 | 1332.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 1341.70 | 1325.33 | 1332.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 1338.00 | 1327.86 | 1332.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:30:00 | 1319.30 | 1326.33 | 1331.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:30:00 | 1319.90 | 1325.98 | 1331.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 1341.20 | 1331.20 | 1330.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 1341.20 | 1331.20 | 1330.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 1341.20 | 1331.20 | 1330.38 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 10:15:00 | 1322.80 | 1328.84 | 1329.52 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 1345.10 | 1327.60 | 1327.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 13:15:00 | 1352.70 | 1339.61 | 1333.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1311.10 | 1335.92 | 1333.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1311.10 | 1335.92 | 1333.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1311.10 | 1335.92 | 1333.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1313.60 | 1335.92 | 1333.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1315.10 | 1331.76 | 1332.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 1305.00 | 1320.12 | 1325.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1332.10 | 1318.98 | 1322.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 1332.10 | 1318.98 | 1322.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1332.10 | 1318.98 | 1322.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 1334.90 | 1318.98 | 1322.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1326.00 | 1320.39 | 1322.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1333.20 | 1320.39 | 1322.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 1354.40 | 1327.19 | 1325.29 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1325.50 | 1331.52 | 1331.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1319.00 | 1329.02 | 1330.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 1341.00 | 1328.69 | 1329.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 1341.00 | 1328.69 | 1329.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1341.00 | 1328.69 | 1329.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 1366.90 | 1328.69 | 1329.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 1345.50 | 1332.05 | 1330.75 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 1322.00 | 1329.75 | 1330.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 1313.10 | 1326.42 | 1328.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 1322.00 | 1318.08 | 1322.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 1322.00 | 1318.08 | 1322.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1322.00 | 1318.08 | 1322.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 1323.90 | 1318.08 | 1322.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1320.00 | 1318.46 | 1322.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1309.40 | 1318.46 | 1322.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1319.80 | 1319.61 | 1320.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1331.20 | 1321.93 | 1321.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1331.20 | 1321.93 | 1321.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1331.20 | 1321.93 | 1321.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1339.30 | 1326.25 | 1323.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 1346.00 | 1348.70 | 1340.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:00:00 | 1346.00 | 1348.70 | 1340.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1366.30 | 1355.66 | 1347.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 09:30:00 | 1414.80 | 1365.32 | 1355.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 10:00:00 | 1426.00 | 1365.32 | 1355.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 15:15:00 | 1414.90 | 1387.02 | 1371.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:45:00 | 1411.60 | 1400.51 | 1380.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1408.70 | 1420.36 | 1410.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1448.00 | 1420.88 | 1413.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:45:00 | 1432.20 | 1425.10 | 1417.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 1429.80 | 1430.98 | 1424.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 12:00:00 | 1424.00 | 1428.64 | 1424.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1419.60 | 1426.83 | 1424.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1419.60 | 1426.83 | 1424.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1420.50 | 1425.57 | 1423.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1418.90 | 1425.57 | 1423.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1391.00 | 1410.05 | 1414.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1407.20 | 1364.75 | 1378.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1407.20 | 1364.75 | 1378.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1407.20 | 1364.75 | 1378.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 1407.20 | 1364.75 | 1378.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1419.00 | 1375.60 | 1382.39 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1412.80 | 1388.87 | 1387.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 1426.00 | 1400.38 | 1393.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 1432.70 | 1440.20 | 1428.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 1432.70 | 1440.20 | 1428.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1416.80 | 1435.52 | 1427.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1416.80 | 1435.52 | 1427.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 1424.40 | 1433.29 | 1427.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1439.60 | 1433.29 | 1427.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 1404.40 | 1423.62 | 1424.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 1404.40 | 1423.62 | 1424.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 12:15:00 | 1402.20 | 1419.34 | 1422.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1375.00 | 1367.02 | 1381.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1375.00 | 1367.02 | 1381.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1362.00 | 1367.64 | 1379.06 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 1387.00 | 1380.56 | 1380.26 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1351.40 | 1374.73 | 1377.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1344.60 | 1368.70 | 1374.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1351.30 | 1316.07 | 1334.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1351.30 | 1316.07 | 1334.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1351.30 | 1316.07 | 1334.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1381.90 | 1316.07 | 1334.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1329.50 | 1318.75 | 1333.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1300.90 | 1335.68 | 1337.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1362.00 | 1335.17 | 1335.22 | SL hit (close>static) qty=1.00 sl=1352.10 alert=retest2 |

### Cycle 70 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 1368.30 | 1341.80 | 1338.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 1372.10 | 1357.52 | 1347.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 1539.90 | 1542.05 | 1509.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 15:00:00 | 1539.90 | 1542.05 | 1509.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1499.30 | 1533.17 | 1510.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 1524.00 | 1529.07 | 1512.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 1521.00 | 1524.91 | 1513.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 1523.90 | 1525.43 | 1514.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1545.70 | 1524.34 | 1515.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 1519.60 | 1530.72 | 1523.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:00:00 | 1519.60 | 1530.72 | 1523.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 1520.30 | 1528.64 | 1523.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 1529.80 | 1528.64 | 1523.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 1505.80 | 1524.07 | 1521.76 | SL hit (close<static) qty=1.00 sl=1513.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 12:15:00 | 1509.20 | 1518.52 | 1519.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 12:15:00 | 1509.20 | 1518.52 | 1519.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 12:15:00 | 1509.20 | 1518.52 | 1519.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 12:15:00 | 1509.20 | 1518.52 | 1519.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 1509.20 | 1518.52 | 1519.59 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1534.30 | 1522.01 | 1520.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 1623.40 | 1552.33 | 1536.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 1739.00 | 1751.66 | 1718.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 1719.10 | 1745.15 | 1718.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1719.10 | 1745.15 | 1718.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 1719.10 | 1745.15 | 1718.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 1729.00 | 1741.92 | 1719.12 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1672.60 | 1708.05 | 1711.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1654.40 | 1697.32 | 1706.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 1697.00 | 1686.66 | 1696.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 10:15:00 | 1697.00 | 1686.66 | 1696.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1697.00 | 1686.66 | 1696.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1697.00 | 1686.66 | 1696.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1709.90 | 1691.31 | 1697.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:45:00 | 1709.00 | 1691.31 | 1697.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1718.50 | 1696.75 | 1699.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 1718.50 | 1696.75 | 1699.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1727.00 | 1702.80 | 1701.78 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 1683.90 | 1700.00 | 1702.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 11:15:00 | 1680.70 | 1690.74 | 1696.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 15:15:00 | 1690.00 | 1688.48 | 1693.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1659.00 | 1688.48 | 1693.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1647.90 | 1680.36 | 1689.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 1640.70 | 1680.36 | 1689.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1491.40 | 1670.22 | 1678.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-04 09:15:00 | 1493.10 | 1636.13 | 1662.17 | Target hit (10%) qty=1.00 alert=retest1 |
| Target hit | 2026-05-04 09:15:00 | 1476.63 | 1636.13 | 1662.17 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1613.70 | 1566.18 | 1563.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1613.70 | 1566.18 | 1563.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 1665.50 | 1616.99 | 1594.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 1626.00 | 1635.56 | 1614.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:45:00 | 1623.80 | 1635.56 | 1614.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-27 09:45:00 | 1944.00 | 2025-05-29 09:15:00 | 2107.60 | TARGET_HIT | 1.00 | 8.42% |
| BUY | retest2 | 2025-05-27 10:45:00 | 1916.00 | 2025-05-29 09:15:00 | 2107.71 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-05-27 11:30:00 | 1916.10 | 2025-05-29 09:15:00 | 2107.60 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2025-05-27 13:00:00 | 1916.00 | 2025-05-30 09:15:00 | 2138.40 | TARGET_HIT | 1.00 | 11.61% |
| BUY | retest2 | 2025-06-03 09:15:00 | 2160.00 | 2025-06-06 10:15:00 | 2084.80 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-06-03 10:30:00 | 2158.00 | 2025-06-06 10:15:00 | 2084.80 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-06-04 09:15:00 | 2213.10 | 2025-06-06 10:15:00 | 2084.80 | STOP_HIT | 1.00 | -5.80% |
| SELL | retest2 | 2025-06-11 12:00:00 | 1962.30 | 2025-06-13 09:15:00 | 1864.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:00:00 | 1962.30 | 2025-06-13 09:15:00 | 1989.60 | STOP_HIT | 0.50 | -1.39% |
| SELL | retest2 | 2025-06-11 12:30:00 | 1961.30 | 2025-06-13 09:15:00 | 1863.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:30:00 | 1961.30 | 2025-06-13 09:15:00 | 1989.60 | STOP_HIT | 0.50 | -1.44% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1945.10 | 2025-06-13 09:15:00 | 1847.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1945.10 | 2025-06-13 09:15:00 | 1989.60 | STOP_HIT | 0.50 | -2.29% |
| SELL | retest2 | 2025-06-12 10:15:00 | 1956.00 | 2025-06-13 09:15:00 | 1858.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 10:15:00 | 1956.00 | 2025-06-13 09:15:00 | 1989.60 | STOP_HIT | 0.50 | -1.72% |
| SELL | retest2 | 2025-06-18 14:15:00 | 1934.10 | 2025-06-23 09:15:00 | 1979.10 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-06-18 15:00:00 | 1935.00 | 2025-06-23 09:15:00 | 1979.10 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-06-19 10:00:00 | 1932.00 | 2025-06-23 09:15:00 | 1979.10 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-07-04 11:30:00 | 1942.00 | 2025-07-08 15:15:00 | 1953.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-04 12:15:00 | 1941.10 | 2025-07-08 15:15:00 | 1953.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-08 09:45:00 | 1942.20 | 2025-07-08 15:15:00 | 1953.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-18 14:15:00 | 1856.70 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-07-21 10:15:00 | 1859.50 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-07-21 11:00:00 | 1856.60 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-07-21 14:45:00 | 1900.80 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-23 12:00:00 | 1880.80 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-23 14:00:00 | 1883.10 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-08-05 11:45:00 | 1483.50 | 2025-08-07 09:15:00 | 1409.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 11:45:00 | 1483.50 | 2025-08-07 15:15:00 | 1459.00 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-09-08 15:00:00 | 1458.00 | 2025-09-10 13:15:00 | 1479.40 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-10 11:30:00 | 1464.20 | 2025-09-10 13:15:00 | 1479.40 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-13 10:30:00 | 1397.40 | 2025-10-17 13:15:00 | 1378.00 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1390.90 | 2025-10-27 09:15:00 | 1329.90 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-11-03 09:15:00 | 1401.80 | 2025-11-06 11:15:00 | 1369.90 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-11-06 10:30:00 | 1370.50 | 2025-11-06 11:15:00 | 1369.90 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-11-10 12:30:00 | 1361.70 | 2025-11-11 09:15:00 | 1382.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-10 14:45:00 | 1363.90 | 2025-11-11 09:15:00 | 1382.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-11-10 15:15:00 | 1357.00 | 2025-11-11 09:15:00 | 1382.80 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-12-01 13:15:00 | 1396.90 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-12-01 14:00:00 | 1397.00 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-12-01 14:45:00 | 1396.40 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-12-01 15:15:00 | 1392.20 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-12-03 09:45:00 | 1375.00 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-03 11:15:00 | 1373.90 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-12-03 12:45:00 | 1374.20 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-12-04 10:15:00 | 1375.50 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-12-05 10:15:00 | 1386.80 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-12-09 09:15:00 | 1336.20 | 2025-12-10 09:15:00 | 1367.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-10 11:15:00 | 1342.50 | 2025-12-11 14:15:00 | 1374.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-12-11 09:30:00 | 1347.00 | 2025-12-11 14:15:00 | 1374.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1352.30 | 2025-12-19 14:15:00 | 1365.90 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1349.20 | 2025-12-31 15:15:00 | 1373.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-01-09 14:00:00 | 1306.90 | 2026-01-12 09:15:00 | 1241.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 14:00:00 | 1306.90 | 2026-01-14 09:15:00 | 1247.70 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2026-02-06 09:30:00 | 1319.30 | 2026-02-09 14:15:00 | 1341.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-06 10:30:00 | 1319.90 | 2026-02-09 14:15:00 | 1341.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1309.40 | 2026-02-25 10:15:00 | 1331.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-25 09:30:00 | 1319.80 | 2026-02-25 10:15:00 | 1331.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-03-04 09:30:00 | 1414.80 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-03-04 10:00:00 | 1426.00 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-03-04 15:15:00 | 1414.90 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-03-05 09:45:00 | 1411.60 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-03-09 14:30:00 | 1448.00 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-10 10:45:00 | 1432.20 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-03-11 10:15:00 | 1429.80 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-03-11 12:00:00 | 1424.00 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1439.60 | 2026-03-20 11:15:00 | 1404.40 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1300.90 | 2026-04-02 13:15:00 | 1362.00 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest2 | 2026-04-13 11:45:00 | 1524.00 | 2026-04-16 09:15:00 | 1505.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-04-13 13:30:00 | 1521.00 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-04-13 14:45:00 | 1523.90 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1545.70 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-04-16 09:15:00 | 1529.80 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2026-04-30 09:15:00 | 1659.00 | 2026-05-04 09:15:00 | 1493.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 10:15:00 | 1640.70 | 2026-05-04 09:15:00 | 1476.63 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 09:15:00 | 1491.40 | 2026-05-07 09:15:00 | 1613.70 | STOP_HIT | 1.00 | -8.20% |
