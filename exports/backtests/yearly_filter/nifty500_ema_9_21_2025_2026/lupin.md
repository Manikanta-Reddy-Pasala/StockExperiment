# Lupin Ltd. (LUPIN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2373.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 84 |
| ALERT1 | 51 |
| ALERT2 | 51 |
| ALERT2_SKIP | 26 |
| ALERT3 | 138 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 48
- **Target hits / Stop hits / Partials:** 0 / 56 / 0
- **Avg / median % per leg:** -0.87% / -0.91%
- **Sum % (uncompounded):** -48.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 5 | 15.2% | 0 | 33 | 0 | -0.66% | -21.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 5 | 15.2% | 0 | 33 | 0 | -0.66% | -21.7% |
| SELL (all) | 23 | 3 | 13.0% | 0 | 23 | 0 | -1.18% | -27.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 3 | 13.0% | 0 | 23 | 0 | -1.18% | -27.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 56 | 8 | 14.3% | 0 | 56 | 0 | -0.87% | -49.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 2088.70 | 2042.69 | 2037.99 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 13:15:00 | 2051.00 | 2064.53 | 2064.55 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 2069.40 | 2065.51 | 2064.99 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 10:15:00 | 2055.10 | 2063.42 | 2064.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 11:15:00 | 2050.20 | 2060.78 | 2062.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2013.40 | 1999.00 | 2020.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:30:00 | 2006.00 | 1999.00 | 2020.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1995.60 | 1978.57 | 1983.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 1992.40 | 1978.57 | 1983.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1994.60 | 1981.78 | 1984.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:45:00 | 2000.20 | 1981.78 | 1984.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1972.00 | 1980.26 | 1983.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 1982.50 | 1980.26 | 1983.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1985.00 | 1981.21 | 1983.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 1985.00 | 1981.21 | 1983.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1987.90 | 1982.54 | 1983.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 11:30:00 | 1973.90 | 1981.04 | 1982.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:30:00 | 1973.50 | 1977.04 | 1980.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 1976.10 | 1956.18 | 1953.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 1976.10 | 1956.18 | 1953.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 1995.00 | 1968.05 | 1959.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 1994.00 | 1995.00 | 1984.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 1994.00 | 1995.00 | 1984.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 1986.20 | 1991.60 | 1985.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 1986.20 | 1991.60 | 1985.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 1995.10 | 1992.30 | 1986.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:45:00 | 1987.80 | 1992.30 | 1986.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2004.60 | 1995.22 | 1988.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 1993.80 | 1995.22 | 1988.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 2016.00 | 2028.68 | 2018.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 2016.00 | 2028.68 | 2018.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 2022.10 | 2027.37 | 2018.57 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 2000.40 | 2015.06 | 2015.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1995.50 | 2011.14 | 2013.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 2011.90 | 2003.91 | 2007.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 2011.90 | 2003.91 | 2007.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2011.90 | 2003.91 | 2007.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 2011.90 | 2003.91 | 2007.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 2008.20 | 2004.77 | 2007.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 2013.00 | 2004.77 | 2007.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 2009.60 | 2005.74 | 2008.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 2008.90 | 2005.74 | 2008.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 2009.50 | 2006.49 | 2008.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 2009.50 | 2006.49 | 2008.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 2013.00 | 2007.79 | 2008.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1994.30 | 2007.79 | 2008.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1974.70 | 2001.17 | 2005.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 1966.00 | 1993.54 | 2001.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 11:15:00 | 1957.40 | 1942.67 | 1942.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1957.40 | 1942.67 | 1942.22 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 1929.60 | 1940.57 | 1941.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 11:15:00 | 1924.00 | 1934.81 | 1938.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 14:15:00 | 1934.90 | 1933.58 | 1936.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:15:00 | 1933.60 | 1933.58 | 1936.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1933.60 | 1933.59 | 1936.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 1935.10 | 1933.59 | 1936.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1930.30 | 1932.93 | 1935.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 10:30:00 | 1924.70 | 1931.12 | 1934.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 1948.20 | 1932.71 | 1932.72 | SL hit (close>static) qty=1.00 sl=1941.10 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 1940.80 | 1934.33 | 1933.45 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 1931.70 | 1934.00 | 1934.06 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 1941.00 | 1935.40 | 1934.69 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1922.70 | 1933.51 | 1934.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 1918.90 | 1930.59 | 1932.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 1949.10 | 1931.38 | 1932.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 13:15:00 | 1949.10 | 1931.38 | 1932.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1949.10 | 1931.38 | 1932.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 1949.10 | 1931.38 | 1932.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 14:15:00 | 1959.50 | 1937.01 | 1934.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 15:15:00 | 1966.70 | 1942.94 | 1937.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 13:15:00 | 1951.20 | 1956.44 | 1947.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 13:15:00 | 1951.20 | 1956.44 | 1947.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1951.20 | 1956.44 | 1947.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1946.90 | 1956.44 | 1947.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1969.70 | 1959.10 | 1949.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:45:00 | 1971.60 | 1959.61 | 1954.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1924.10 | 1966.06 | 1967.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1924.10 | 1966.06 | 1967.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1920.00 | 1956.85 | 1963.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 1943.20 | 1934.91 | 1946.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 1943.20 | 1934.91 | 1946.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1943.20 | 1934.91 | 1946.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 1961.00 | 1934.91 | 1946.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1933.40 | 1934.61 | 1945.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 1938.00 | 1934.61 | 1945.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1917.60 | 1898.72 | 1905.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 1914.30 | 1898.72 | 1905.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1925.10 | 1904.00 | 1907.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 1929.10 | 1904.00 | 1907.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 1925.00 | 1911.86 | 1910.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 1951.50 | 1928.60 | 1920.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 1945.00 | 1945.27 | 1934.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 12:15:00 | 1941.10 | 1945.27 | 1934.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1932.50 | 1942.72 | 1934.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 1932.50 | 1942.72 | 1934.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1932.40 | 1940.66 | 1934.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1948.70 | 1936.96 | 1933.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 1929.00 | 1944.20 | 1941.50 | SL hit (close<static) qty=1.00 sl=1929.30 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1933.90 | 1939.42 | 1939.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 1931.10 | 1937.76 | 1938.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 12:15:00 | 1912.00 | 1907.97 | 1919.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 13:00:00 | 1912.00 | 1907.97 | 1919.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1918.30 | 1908.50 | 1916.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1918.30 | 1908.50 | 1916.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1909.10 | 1908.62 | 1915.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:45:00 | 1908.40 | 1909.09 | 1914.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 1919.90 | 1911.25 | 1914.64 | SL hit (close>static) qty=1.00 sl=1918.50 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 1927.80 | 1916.94 | 1916.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 11:15:00 | 1948.90 | 1923.33 | 1919.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 1951.50 | 1957.64 | 1947.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 13:00:00 | 1951.50 | 1957.64 | 1947.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1960.20 | 1958.82 | 1951.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 10:30:00 | 1965.60 | 1960.09 | 1952.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:30:00 | 1967.00 | 1962.10 | 1953.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1928.40 | 1972.10 | 1969.63 | SL hit (close<static) qty=1.00 sl=1947.70 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1933.70 | 1964.42 | 1966.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 1925.00 | 1942.24 | 1953.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 1883.00 | 1881.96 | 1898.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:15:00 | 1872.50 | 1881.96 | 1898.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1858.30 | 1866.36 | 1879.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:00:00 | 1842.00 | 1855.50 | 1869.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 1912.80 | 1865.85 | 1870.95 | SL hit (close>static) qty=1.00 sl=1908.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 11:15:00 | 1889.20 | 1876.16 | 1875.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 12:15:00 | 1927.80 | 1886.49 | 1879.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 13:15:00 | 1922.50 | 1922.70 | 1907.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 14:00:00 | 1922.50 | 1922.70 | 1907.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1918.80 | 1920.73 | 1910.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:30:00 | 1933.30 | 1923.79 | 1912.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 1958.80 | 1964.98 | 1965.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 1958.80 | 1964.98 | 1965.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 11:15:00 | 1941.60 | 1960.31 | 1963.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1957.70 | 1949.14 | 1954.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 1957.70 | 1949.14 | 1954.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1957.70 | 1949.14 | 1954.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 1961.20 | 1949.14 | 1954.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1963.60 | 1952.03 | 1955.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 1963.60 | 1952.03 | 1955.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1960.00 | 1953.62 | 1956.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 1963.40 | 1953.62 | 1956.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 14:15:00 | 1963.50 | 1957.50 | 1957.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 1970.90 | 1961.81 | 1959.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 13:15:00 | 1970.50 | 1973.68 | 1969.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 13:15:00 | 1970.50 | 1973.68 | 1969.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1970.50 | 1973.68 | 1969.31 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1926.70 | 1961.51 | 1964.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1913.90 | 1937.82 | 1951.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1909.40 | 1905.38 | 1920.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1909.40 | 1905.38 | 1920.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1920.30 | 1906.02 | 1913.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1920.30 | 1906.02 | 1913.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1924.40 | 1909.69 | 1914.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 1923.90 | 1909.69 | 1914.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1909.00 | 1912.47 | 1915.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 1916.00 | 1912.47 | 1915.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1910.90 | 1899.36 | 1904.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 1910.90 | 1899.36 | 1904.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1925.00 | 1904.49 | 1906.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 1925.00 | 1904.49 | 1906.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 1930.30 | 1909.65 | 1908.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 12:15:00 | 1933.00 | 1914.32 | 1910.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 1938.70 | 1940.85 | 1932.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 11:00:00 | 1938.70 | 1940.85 | 1932.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1938.10 | 1939.97 | 1933.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 1938.10 | 1939.97 | 1933.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1951.90 | 1944.61 | 1938.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:45:00 | 1955.90 | 1948.53 | 1941.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 15:00:00 | 1956.40 | 1950.37 | 1946.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 2023.00 | 2038.43 | 2040.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 2023.00 | 2038.43 | 2040.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 2010.70 | 2032.88 | 2037.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 2007.10 | 2006.78 | 2015.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 2007.10 | 2006.78 | 2015.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2011.70 | 2002.20 | 2009.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:15:00 | 1998.40 | 2004.25 | 2009.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 1983.00 | 1950.52 | 1946.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 1983.00 | 1950.52 | 1946.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 1985.80 | 1957.57 | 1950.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 1969.60 | 1976.82 | 1967.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 14:00:00 | 1969.60 | 1976.82 | 1967.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1933.80 | 1967.47 | 1965.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 1933.80 | 1967.47 | 1965.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 1938.10 | 1961.60 | 1962.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 09:15:00 | 1923.10 | 1940.54 | 1950.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1983.00 | 1927.49 | 1930.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1983.00 | 1927.49 | 1930.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1983.00 | 1927.49 | 1930.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 1983.00 | 1927.49 | 1930.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1970.00 | 1935.99 | 1934.29 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1935.10 | 1952.44 | 1954.17 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 1953.20 | 1948.00 | 1947.83 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 1933.00 | 1945.42 | 1946.71 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1949.00 | 1944.78 | 1944.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 12:15:00 | 1955.00 | 1947.40 | 1945.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1937.30 | 1945.73 | 1945.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1937.30 | 1945.73 | 1945.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1937.30 | 1945.73 | 1945.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1937.30 | 1945.73 | 1945.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1945.00 | 1945.59 | 1945.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1935.90 | 1945.59 | 1945.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 1938.00 | 1944.07 | 1944.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 1926.70 | 1937.58 | 1941.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1919.20 | 1915.22 | 1923.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 1919.20 | 1915.22 | 1923.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1919.20 | 1915.22 | 1923.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 1919.20 | 1915.22 | 1923.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1927.30 | 1917.63 | 1923.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1927.60 | 1917.63 | 1923.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1930.70 | 1920.25 | 1924.12 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1949.00 | 1929.73 | 1927.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 1956.20 | 1944.31 | 1936.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1934.70 | 1942.39 | 1936.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1934.70 | 1942.39 | 1936.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1934.70 | 1942.39 | 1936.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1934.70 | 1942.39 | 1936.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1935.50 | 1941.01 | 1936.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 1931.70 | 1941.01 | 1936.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1935.40 | 1939.89 | 1936.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:30:00 | 1934.00 | 1939.89 | 1936.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 1944.00 | 1940.71 | 1936.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1949.10 | 1942.45 | 1938.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 10:15:00 | 1955.00 | 1942.14 | 1938.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 13:15:00 | 1955.90 | 1973.91 | 1974.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1955.90 | 1973.91 | 1974.20 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 09:15:00 | 1996.00 | 1975.92 | 1974.77 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 14:15:00 | 1970.00 | 1974.45 | 1974.76 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 2009.50 | 1980.59 | 1977.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 2022.50 | 1991.28 | 1982.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 1985.90 | 1998.35 | 1989.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 14:15:00 | 1985.90 | 1998.35 | 1989.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1985.90 | 1998.35 | 1989.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 1985.90 | 1998.35 | 1989.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1985.00 | 1995.68 | 1988.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 1976.90 | 1995.68 | 1988.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1980.30 | 1992.60 | 1987.90 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 1979.90 | 1984.89 | 1985.23 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1990.00 | 1985.56 | 1985.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 12:15:00 | 2014.10 | 1993.85 | 1989.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 11:15:00 | 2046.60 | 2048.70 | 2033.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 11:45:00 | 2046.60 | 2048.70 | 2033.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 2050.70 | 2052.28 | 2045.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 2046.50 | 2052.28 | 2045.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2048.50 | 2051.85 | 2046.22 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 2040.30 | 2044.77 | 2045.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 2030.00 | 2041.06 | 2043.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 2036.80 | 2035.53 | 2039.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:30:00 | 2036.00 | 2035.53 | 2039.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 2035.00 | 2035.42 | 2039.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 2035.00 | 2035.42 | 2039.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 2034.70 | 2035.37 | 2038.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:30:00 | 2036.10 | 2035.37 | 2038.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 2035.60 | 2032.51 | 2035.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:30:00 | 2036.60 | 2032.51 | 2035.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 2028.90 | 2031.79 | 2034.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 2002.30 | 2031.63 | 2034.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 2037.90 | 2008.26 | 2016.40 | SL hit (close>static) qty=1.00 sl=2035.60 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 2043.00 | 2024.75 | 2022.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 2056.90 | 2040.56 | 2032.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 13:15:00 | 2062.20 | 2065.80 | 2054.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 14:00:00 | 2062.20 | 2065.80 | 2054.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2075.00 | 2078.26 | 2070.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 2075.00 | 2078.26 | 2070.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 2071.50 | 2076.91 | 2070.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 2070.10 | 2076.91 | 2070.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 2081.00 | 2077.73 | 2071.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 2070.10 | 2077.73 | 2071.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 2076.10 | 2077.40 | 2071.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:45:00 | 2076.90 | 2077.40 | 2071.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 2085.10 | 2080.00 | 2075.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:15:00 | 2088.20 | 2081.33 | 2076.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:15:00 | 2089.50 | 2082.96 | 2078.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 2069.30 | 2081.28 | 2078.39 | SL hit (close<static) qty=1.00 sl=2074.50 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 2071.90 | 2086.13 | 2086.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 2067.20 | 2082.34 | 2085.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 2070.20 | 2062.60 | 2069.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 2070.20 | 2062.60 | 2069.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 2070.20 | 2062.60 | 2069.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 2070.20 | 2062.60 | 2069.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 2059.70 | 2062.02 | 2068.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 2067.60 | 2062.02 | 2068.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 2056.40 | 2060.46 | 2066.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:30:00 | 2065.00 | 2060.46 | 2066.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 2070.20 | 2060.77 | 2064.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 2071.20 | 2060.77 | 2064.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 2077.60 | 2064.13 | 2065.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 2077.60 | 2064.13 | 2065.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 2077.80 | 2066.87 | 2066.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 2097.10 | 2078.10 | 2072.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 2086.40 | 2097.91 | 2087.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 2086.40 | 2097.91 | 2087.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2086.40 | 2097.91 | 2087.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 2086.40 | 2097.91 | 2087.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2091.50 | 2096.62 | 2088.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:00:00 | 2096.40 | 2096.58 | 2088.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 2095.70 | 2092.32 | 2089.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 2085.90 | 2090.74 | 2089.93 | SL hit (close<static) qty=1.00 sl=2086.40 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 2111.30 | 2118.33 | 2118.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 15:15:00 | 2108.50 | 2115.53 | 2117.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 2115.00 | 2107.05 | 2109.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 13:15:00 | 2115.00 | 2107.05 | 2109.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 2115.00 | 2107.05 | 2109.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 2115.00 | 2107.05 | 2109.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 2114.00 | 2108.44 | 2110.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 2112.00 | 2108.44 | 2110.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 2100.60 | 2106.87 | 2109.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 2099.00 | 2105.80 | 2108.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 2093.80 | 2103.40 | 2107.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 2110.00 | 2092.26 | 2091.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 2110.00 | 2092.26 | 2091.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 2120.00 | 2105.85 | 2100.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 2100.40 | 2105.20 | 2101.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 12:15:00 | 2100.40 | 2105.20 | 2101.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 2100.40 | 2105.20 | 2101.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:45:00 | 2096.50 | 2105.20 | 2101.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 2106.30 | 2105.42 | 2101.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:15:00 | 2109.70 | 2105.42 | 2101.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 2098.90 | 2104.40 | 2102.21 | SL hit (close<static) qty=1.00 sl=2099.50 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 2076.70 | 2096.85 | 2099.27 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 2122.50 | 2100.93 | 2099.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 13:15:00 | 2134.00 | 2107.55 | 2102.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 2191.30 | 2192.85 | 2165.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 2191.30 | 2192.85 | 2165.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 2178.30 | 2190.48 | 2176.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 2164.90 | 2190.48 | 2176.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 2184.80 | 2189.34 | 2177.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 2177.40 | 2189.34 | 2177.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 2179.90 | 2184.92 | 2178.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:30:00 | 2179.70 | 2184.92 | 2178.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 2178.00 | 2183.54 | 2178.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 2183.40 | 2183.54 | 2178.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 2178.00 | 2182.43 | 2178.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 2144.60 | 2182.43 | 2178.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2157.40 | 2177.42 | 2176.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 2156.00 | 2177.42 | 2176.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 2166.10 | 2175.16 | 2175.27 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 2182.50 | 2176.63 | 2175.83 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 2159.40 | 2173.06 | 2174.84 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 2189.40 | 2175.86 | 2175.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 2197.80 | 2180.25 | 2177.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 2194.30 | 2198.76 | 2190.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:00:00 | 2194.30 | 2198.76 | 2190.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 2180.00 | 2195.01 | 2189.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 2180.00 | 2195.01 | 2189.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2177.60 | 2191.53 | 2188.63 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 2168.40 | 2184.11 | 2185.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 2135.70 | 2160.63 | 2168.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2187.40 | 2159.92 | 2165.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2187.40 | 2159.92 | 2165.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2187.40 | 2159.92 | 2165.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 2187.40 | 2159.92 | 2165.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2181.40 | 2164.21 | 2167.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 2171.10 | 2168.88 | 2169.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 2142.00 | 2133.19 | 2133.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 2142.00 | 2133.19 | 2133.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 2147.10 | 2135.97 | 2134.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2130.00 | 2138.25 | 2136.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 2130.00 | 2138.25 | 2136.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2130.00 | 2138.25 | 2136.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 2128.80 | 2138.25 | 2136.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 2144.00 | 2139.40 | 2136.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:15:00 | 2131.30 | 2139.40 | 2136.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 2114.50 | 2134.42 | 2134.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 2108.40 | 2126.68 | 2131.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 14:15:00 | 2138.60 | 2129.07 | 2131.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 2138.60 | 2129.07 | 2131.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 2138.60 | 2129.07 | 2131.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 2138.60 | 2129.07 | 2131.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 2131.00 | 2129.45 | 2131.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 2118.00 | 2129.45 | 2131.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2112.60 | 2126.08 | 2129.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 2097.10 | 2126.08 | 2129.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2168.70 | 2125.08 | 2124.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2168.70 | 2125.08 | 2124.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 10:15:00 | 2229.50 | 2200.56 | 2180.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 2173.80 | 2203.36 | 2191.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 2173.80 | 2203.36 | 2191.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 2173.80 | 2203.36 | 2191.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 2173.80 | 2203.36 | 2191.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2178.00 | 2198.29 | 2190.47 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 14:15:00 | 2172.80 | 2185.08 | 2186.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 15:15:00 | 2162.10 | 2180.48 | 2183.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2190.00 | 2182.39 | 2184.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 2190.00 | 2182.39 | 2184.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2190.00 | 2182.39 | 2184.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 2190.00 | 2182.39 | 2184.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 2199.60 | 2185.83 | 2185.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 2205.30 | 2189.72 | 2187.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 12:15:00 | 2185.00 | 2188.78 | 2187.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 12:15:00 | 2185.00 | 2188.78 | 2187.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 2185.00 | 2188.78 | 2187.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 2186.20 | 2188.78 | 2187.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 2190.30 | 2189.08 | 2187.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 2208.70 | 2189.15 | 2187.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:45:00 | 2201.90 | 2205.38 | 2198.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:30:00 | 2204.60 | 2210.16 | 2207.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:30:00 | 2205.40 | 2205.33 | 2205.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 2210.30 | 2206.33 | 2205.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 2203.50 | 2206.33 | 2205.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 2179.90 | 2202.43 | 2204.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 2179.90 | 2202.43 | 2204.15 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 2220.90 | 2205.52 | 2203.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 2254.00 | 2223.16 | 2214.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 10:15:00 | 2239.50 | 2240.84 | 2230.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:30:00 | 2242.60 | 2240.84 | 2230.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 2242.50 | 2241.17 | 2231.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:30:00 | 2231.80 | 2241.17 | 2231.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2227.70 | 2238.76 | 2232.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 2232.70 | 2238.76 | 2232.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 2230.80 | 2237.17 | 2232.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 2226.30 | 2237.17 | 2232.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2222.20 | 2234.18 | 2231.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 2240.20 | 2233.20 | 2231.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 2242.40 | 2233.20 | 2231.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 2218.40 | 2230.13 | 2231.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 2218.40 | 2230.13 | 2231.24 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 2248.80 | 2232.06 | 2230.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 10:15:00 | 2252.30 | 2239.97 | 2235.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2305.00 | 2308.01 | 2290.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:00:00 | 2305.00 | 2308.01 | 2290.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 2322.00 | 2309.91 | 2295.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 2329.10 | 2312.33 | 2298.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2274.80 | 2301.34 | 2296.31 | SL hit (close<static) qty=1.00 sl=2291.70 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 2278.30 | 2293.31 | 2294.51 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 15:15:00 | 2307.00 | 2295.68 | 2294.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 2344.70 | 2305.48 | 2299.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 2331.80 | 2338.56 | 2327.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 2288.00 | 2338.56 | 2327.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2303.20 | 2331.49 | 2325.17 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 2300.00 | 2317.74 | 2319.73 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 2338.30 | 2320.22 | 2319.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 2367.10 | 2339.13 | 2329.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 2339.90 | 2348.92 | 2339.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 2339.90 | 2348.92 | 2339.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 2339.90 | 2348.92 | 2339.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 2339.90 | 2348.92 | 2339.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 2337.00 | 2346.54 | 2339.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 2312.80 | 2346.54 | 2339.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2317.10 | 2340.65 | 2337.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 2319.90 | 2340.65 | 2337.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 2334.50 | 2339.42 | 2336.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 2340.00 | 2340.26 | 2337.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:30:00 | 2342.90 | 2344.07 | 2341.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 2345.20 | 2344.07 | 2341.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 2317.00 | 2336.64 | 2338.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 2317.00 | 2336.64 | 2338.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 15:15:00 | 2306.00 | 2325.81 | 2332.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 2304.10 | 2296.15 | 2308.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 2304.10 | 2296.15 | 2308.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2304.10 | 2296.15 | 2308.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 2304.80 | 2296.15 | 2308.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 2296.50 | 2296.22 | 2307.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 2288.70 | 2296.22 | 2307.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:45:00 | 2294.80 | 2295.99 | 2305.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:15:00 | 2293.40 | 2295.99 | 2305.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 2311.50 | 2304.45 | 2305.33 | SL hit (close>static) qty=1.00 sl=2310.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 2319.00 | 2292.68 | 2291.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 2327.50 | 2299.65 | 2294.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 2285.90 | 2303.68 | 2298.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 2285.90 | 2303.68 | 2298.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 2285.90 | 2303.68 | 2298.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 2285.90 | 2303.68 | 2298.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 2293.80 | 2301.70 | 2298.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 13:15:00 | 2300.60 | 2299.03 | 2297.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 2317.80 | 2298.95 | 2298.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 2291.00 | 2296.28 | 2296.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 2291.00 | 2296.28 | 2296.99 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 2312.70 | 2299.57 | 2298.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 2335.10 | 2306.67 | 2301.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 2344.80 | 2344.90 | 2333.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 12:00:00 | 2344.80 | 2344.90 | 2333.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 2334.30 | 2342.70 | 2335.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 2334.30 | 2342.70 | 2335.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 2331.90 | 2340.54 | 2334.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 2316.50 | 2340.54 | 2334.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 2307.00 | 2333.83 | 2332.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 2307.00 | 2333.83 | 2332.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 2304.30 | 2327.93 | 2329.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 10:15:00 | 2289.00 | 2314.23 | 2321.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 2271.70 | 2261.65 | 2282.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 2271.70 | 2261.65 | 2282.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 2265.00 | 2264.65 | 2279.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 2251.10 | 2264.65 | 2279.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 2250.60 | 2261.84 | 2277.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 2287.00 | 2268.17 | 2271.61 | SL hit (close>static) qty=1.00 sl=2286.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 2298.70 | 2274.28 | 2274.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 2312.90 | 2297.51 | 2291.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 11:15:00 | 2320.00 | 2322.60 | 2311.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 11:45:00 | 2317.40 | 2322.60 | 2311.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 2313.60 | 2321.69 | 2313.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 2313.60 | 2321.69 | 2313.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 2310.90 | 2319.54 | 2313.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 2334.00 | 2319.54 | 2313.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:00:00 | 2319.40 | 2332.65 | 2326.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 11:00:00 | 2319.00 | 2329.92 | 2325.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 2298.40 | 2323.62 | 2322.98 | SL hit (close<static) qty=1.00 sl=2310.20 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 2304.20 | 2319.73 | 2321.27 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2333.00 | 2323.03 | 2322.22 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 2304.00 | 2318.77 | 2320.40 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 2332.80 | 2321.59 | 2320.77 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 13:15:00 | 2314.70 | 2323.27 | 2323.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 2296.30 | 2314.81 | 2319.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 12:15:00 | 2315.00 | 2313.66 | 2317.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 12:45:00 | 2315.20 | 2313.66 | 2317.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 2303.00 | 2311.52 | 2316.16 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 2378.50 | 2324.35 | 2320.77 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 2292.20 | 2326.91 | 2327.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 2269.40 | 2315.41 | 2322.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 2331.60 | 2306.34 | 2313.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 2331.60 | 2306.34 | 2313.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 2331.60 | 2306.34 | 2313.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 2331.60 | 2306.34 | 2313.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 2318.50 | 2308.77 | 2313.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 2308.00 | 2308.77 | 2313.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:45:00 | 2315.80 | 2311.18 | 2314.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:45:00 | 2315.00 | 2312.30 | 2314.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 14:15:00 | 2323.20 | 2316.58 | 2316.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 2323.20 | 2316.58 | 2316.30 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 2278.50 | 2308.76 | 2312.84 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 2320.40 | 2311.61 | 2310.93 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 2303.00 | 2310.02 | 2310.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 2292.40 | 2306.49 | 2308.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 2304.70 | 2301.29 | 2305.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 12:15:00 | 2304.70 | 2301.29 | 2305.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 2304.70 | 2301.29 | 2305.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:00:00 | 2304.70 | 2301.29 | 2305.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 2304.20 | 2301.87 | 2305.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:15:00 | 2305.70 | 2301.87 | 2305.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 2304.10 | 2302.32 | 2305.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 2304.10 | 2302.32 | 2305.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 2312.30 | 2304.32 | 2305.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 2312.50 | 2304.32 | 2305.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 2338.40 | 2311.13 | 2308.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 11:15:00 | 2349.70 | 2323.99 | 2315.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 2337.90 | 2338.11 | 2326.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 09:30:00 | 2350.70 | 2338.11 | 2326.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 2336.00 | 2338.74 | 2329.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 2336.00 | 2338.74 | 2329.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 2434.60 | 2449.61 | 2422.49 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 2380.10 | 2407.08 | 2409.49 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:15:00 | 1979.50 | 2025-05-13 09:15:00 | 2088.70 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2025-05-12 10:45:00 | 2019.80 | 2025-05-13 09:15:00 | 2088.70 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-05-27 11:30:00 | 1973.90 | 2025-06-05 10:15:00 | 1976.10 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-05-27 13:30:00 | 1973.50 | 2025-06-05 10:15:00 | 1976.10 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-06-17 10:45:00 | 1966.00 | 2025-06-24 11:15:00 | 1957.40 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-06-26 10:30:00 | 1924.70 | 2025-06-27 10:15:00 | 1948.20 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-04 09:45:00 | 1971.60 | 2025-07-08 09:15:00 | 1924.10 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1948.70 | 2025-07-18 10:15:00 | 1929.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-18 12:15:00 | 1943.50 | 2025-07-18 13:15:00 | 1933.90 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-07-23 13:45:00 | 1908.40 | 2025-07-23 14:15:00 | 1919.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-29 10:30:00 | 1965.60 | 2025-07-31 09:15:00 | 1928.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-07-29 11:30:00 | 1967.00 | 2025-07-31 09:15:00 | 1928.40 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-08-06 14:00:00 | 1842.00 | 2025-08-07 09:15:00 | 1912.80 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-08-11 10:30:00 | 1933.30 | 2025-08-20 10:15:00 | 1958.80 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-09-08 12:45:00 | 1955.90 | 2025-09-22 12:15:00 | 2023.00 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2025-09-09 15:00:00 | 1956.40 | 2025-09-22 12:15:00 | 2023.00 | STOP_HIT | 1.00 | 3.40% |
| SELL | retest2 | 2025-09-25 12:15:00 | 1998.40 | 2025-10-01 11:15:00 | 1983.00 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1949.10 | 2025-11-06 13:15:00 | 1955.90 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-10-31 10:15:00 | 1955.00 | 2025-11-06 13:15:00 | 1955.90 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-11-24 09:15:00 | 2002.30 | 2025-11-25 09:15:00 | 2037.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-02 13:15:00 | 2088.20 | 2025-12-03 09:15:00 | 2069.30 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-02 15:15:00 | 2089.50 | 2025-12-03 09:15:00 | 2069.30 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-04 13:00:00 | 2091.70 | 2025-12-08 12:15:00 | 2071.90 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-05 10:00:00 | 2087.60 | 2025-12-08 12:15:00 | 2071.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-12-15 12:00:00 | 2096.40 | 2025-12-16 15:15:00 | 2085.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-16 12:15:00 | 2095.70 | 2025-12-16 15:15:00 | 2085.90 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-17 09:15:00 | 2117.80 | 2025-12-23 13:15:00 | 2111.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-12-29 10:15:00 | 2099.00 | 2025-12-31 12:15:00 | 2110.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-29 11:00:00 | 2093.80 | 2025-12-31 12:15:00 | 2110.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-01-02 14:15:00 | 2109.70 | 2026-01-05 09:15:00 | 2098.90 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-01-22 13:15:00 | 2171.10 | 2026-01-30 12:15:00 | 2142.00 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2026-02-02 10:15:00 | 2097.10 | 2026-02-03 09:15:00 | 2168.70 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2026-02-09 15:15:00 | 2208.70 | 2026-02-13 09:15:00 | 2179.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-02-10 14:45:00 | 2201.90 | 2026-02-13 09:15:00 | 2179.90 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-12 11:30:00 | 2204.60 | 2026-02-13 09:15:00 | 2179.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-12 13:30:00 | 2205.40 | 2026-02-13 09:15:00 | 2179.90 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-02-19 11:45:00 | 2240.20 | 2026-02-20 09:15:00 | 2218.40 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-02-19 12:15:00 | 2242.40 | 2026-02-20 09:15:00 | 2218.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-02-27 13:45:00 | 2329.10 | 2026-03-02 09:15:00 | 2274.80 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-03-12 11:30:00 | 2340.00 | 2026-03-13 12:15:00 | 2317.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-03-13 10:30:00 | 2342.90 | 2026-03-13 12:15:00 | 2317.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-03-13 11:15:00 | 2345.20 | 2026-03-13 12:15:00 | 2317.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-03-17 11:15:00 | 2288.70 | 2026-03-18 13:15:00 | 2311.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-03-17 12:45:00 | 2294.80 | 2026-03-18 13:15:00 | 2311.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-03-17 13:15:00 | 2293.40 | 2026-03-18 13:15:00 | 2311.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-03-19 09:15:00 | 2280.00 | 2026-03-20 11:15:00 | 2311.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-03-23 13:15:00 | 2300.60 | 2026-03-24 10:15:00 | 2291.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-03-24 09:15:00 | 2317.80 | 2026-03-24 10:15:00 | 2291.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-06 09:15:00 | 2251.10 | 2026-04-07 09:15:00 | 2287.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-06 10:00:00 | 2250.60 | 2026-04-07 09:15:00 | 2287.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-04-15 09:15:00 | 2334.00 | 2026-04-16 11:15:00 | 2298.40 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-04-16 10:00:00 | 2319.40 | 2026-04-16 11:15:00 | 2298.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-04-16 11:00:00 | 2319.00 | 2026-04-16 11:15:00 | 2298.40 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-04-27 11:15:00 | 2308.00 | 2026-04-27 14:15:00 | 2323.20 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-04-27 11:45:00 | 2315.80 | 2026-04-27 14:15:00 | 2323.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-04-27 12:45:00 | 2315.00 | 2026-04-27 14:15:00 | 2323.20 | STOP_HIT | 1.00 | -0.35% |
