# Deepak Nitrite Ltd. (DEEPAKNTR)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1875.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 39 |
| ALERT2 | 40 |
| ALERT2_SKIP | 21 |
| ALERT3 | 112 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 49 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 28 / 28
- **Target hits / Stop hits / Partials:** 5 / 44 / 7
- **Avg / median % per leg:** 1.71% / 0.03%
- **Sum % (uncompounded):** 95.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 9 | 50.0% | 5 | 13 | 0 | 2.94% | 52.9% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.89% | 0.9% |
| BUY @ 3rd Alert (retest2) | 17 | 8 | 47.1% | 5 | 12 | 0 | 3.06% | 52.0% |
| SELL (all) | 38 | 19 | 50.0% | 0 | 31 | 7 | 1.13% | 42.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 19 | 50.0% | 0 | 31 | 7 | 1.13% | 42.8% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.89% | 0.9% |
| retest2 (combined) | 55 | 27 | 49.1% | 5 | 43 | 7 | 1.72% | 94.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1973.30 | 1947.05 | 1944.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1981.80 | 1954.00 | 1948.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1958.40 | 1964.31 | 1955.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:45:00 | 1957.40 | 1964.31 | 1955.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 1954.80 | 1962.41 | 1955.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:00:00 | 1954.80 | 1962.41 | 1955.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1946.00 | 1959.13 | 1954.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1946.00 | 1959.13 | 1954.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1945.00 | 1956.30 | 1953.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:15:00 | 1945.60 | 1956.30 | 1953.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1972.30 | 1958.04 | 1955.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:30:00 | 1982.00 | 1966.44 | 1960.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:00:00 | 1981.50 | 1969.46 | 1962.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1998.00 | 1971.10 | 1963.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 2060.50 | 2088.82 | 2090.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 2060.50 | 2088.82 | 2090.53 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 2129.70 | 2096.85 | 2093.56 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 2069.90 | 2090.07 | 2092.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 2058.80 | 2071.15 | 2077.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 2095.50 | 2067.40 | 2073.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 2095.50 | 2067.40 | 2073.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2095.50 | 2067.40 | 2073.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 2092.00 | 2067.40 | 2073.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2102.00 | 2074.32 | 2076.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 2109.50 | 2074.32 | 2076.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 2071.10 | 2056.42 | 2064.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:00:00 | 2071.10 | 2056.42 | 2064.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 2048.30 | 2054.80 | 2062.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 11:15:00 | 2042.40 | 2054.80 | 2062.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 2108.80 | 2048.68 | 2051.85 | SL hit (close>static) qty=1.00 sl=2077.90 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 2121.00 | 2063.14 | 2058.14 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 2026.80 | 2061.74 | 2063.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1991.00 | 2031.30 | 2046.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 2014.90 | 2007.09 | 2023.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 2014.90 | 2007.09 | 2023.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 2014.90 | 2007.09 | 2023.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 2014.90 | 2007.09 | 2023.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 2026.10 | 2010.89 | 2023.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 2026.10 | 2010.89 | 2023.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 2025.50 | 2013.81 | 2023.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:30:00 | 2030.00 | 2013.81 | 2023.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 2017.70 | 2014.59 | 2023.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 2008.00 | 2017.74 | 2022.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:00:00 | 2007.80 | 2002.33 | 2008.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1907.60 | 1934.16 | 1945.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1907.41 | 1934.16 | 1945.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 1912.30 | 1908.52 | 1919.19 | SL hit (close>ema200) qty=0.50 sl=1908.52 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 1895.10 | 1888.86 | 1888.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 15:15:00 | 1900.00 | 1892.97 | 1890.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 1971.20 | 1976.06 | 1960.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 09:45:00 | 1971.80 | 1976.06 | 1960.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1974.90 | 1980.20 | 1971.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1974.90 | 1980.20 | 1971.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1962.20 | 1976.60 | 1970.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 1961.50 | 1976.60 | 1970.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1960.10 | 1973.30 | 1969.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:45:00 | 1963.60 | 1973.30 | 1969.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1955.20 | 1969.68 | 1968.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 1955.20 | 1969.68 | 1968.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 1960.80 | 1966.53 | 1967.17 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 09:15:00 | 1982.10 | 1969.65 | 1968.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 2005.30 | 1978.13 | 1972.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 11:15:00 | 1986.90 | 1991.02 | 1983.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 11:45:00 | 1987.80 | 1991.02 | 1983.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1968.70 | 1985.98 | 1982.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 1968.70 | 1985.98 | 1982.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1970.40 | 1982.86 | 1981.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 1973.60 | 1982.86 | 1981.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 1974.90 | 1979.23 | 1979.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 1964.10 | 1976.20 | 1978.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 11:15:00 | 1965.90 | 1963.55 | 1968.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 1965.90 | 1963.55 | 1968.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1965.90 | 1963.55 | 1968.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 1967.70 | 1963.55 | 1968.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1965.00 | 1963.84 | 1968.54 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1979.10 | 1972.63 | 1971.77 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 1969.10 | 1981.16 | 1981.84 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1989.50 | 1980.69 | 1980.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 1991.00 | 1982.75 | 1981.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 1977.80 | 1982.12 | 1981.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 1977.80 | 1982.12 | 1981.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1977.80 | 1982.12 | 1981.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1977.80 | 1982.12 | 1981.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1977.90 | 1981.28 | 1980.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1987.70 | 1981.28 | 1980.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 1975.00 | 1980.02 | 1980.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 1975.00 | 1980.02 | 1980.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 1972.00 | 1978.42 | 1979.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1935.60 | 1933.46 | 1939.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 10:00:00 | 1935.60 | 1933.46 | 1939.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1940.30 | 1931.53 | 1934.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 1938.30 | 1931.53 | 1934.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1934.10 | 1932.04 | 1934.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 1940.50 | 1932.04 | 1934.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1922.50 | 1930.13 | 1933.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 1919.20 | 1927.37 | 1931.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 1920.00 | 1925.89 | 1930.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1912.30 | 1925.31 | 1929.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 1823.24 | 1840.67 | 1856.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 1824.00 | 1840.67 | 1856.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 1816.68 | 1840.67 | 1856.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 14:15:00 | 1823.70 | 1816.83 | 1825.43 | SL hit (close>ema200) qty=0.50 sl=1816.83 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 1825.10 | 1815.19 | 1814.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 1835.10 | 1821.32 | 1817.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 1852.00 | 1853.79 | 1841.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:45:00 | 1850.50 | 1853.79 | 1841.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1819.20 | 1857.13 | 1851.05 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 1832.30 | 1846.72 | 1847.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 09:15:00 | 1818.90 | 1839.04 | 1843.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 15:15:00 | 1815.00 | 1814.57 | 1818.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:15:00 | 1809.80 | 1814.57 | 1818.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1818.80 | 1813.25 | 1815.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 1818.80 | 1813.25 | 1815.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1815.20 | 1813.64 | 1815.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 1813.00 | 1813.64 | 1815.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 14:30:00 | 1812.50 | 1812.02 | 1814.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1831.30 | 1816.24 | 1815.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 1831.30 | 1816.24 | 1815.55 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 1812.50 | 1819.10 | 1819.29 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 1846.00 | 1824.48 | 1821.72 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 1811.40 | 1820.37 | 1820.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 1802.00 | 1811.99 | 1816.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 1798.30 | 1798.18 | 1805.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 11:00:00 | 1798.30 | 1798.18 | 1805.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1778.50 | 1791.83 | 1799.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:45:00 | 1774.70 | 1786.72 | 1795.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:00:00 | 1775.00 | 1784.38 | 1793.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 1768.40 | 1749.53 | 1749.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1768.40 | 1749.53 | 1749.35 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1743.10 | 1748.15 | 1748.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 13:15:00 | 1737.70 | 1746.06 | 1747.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 1748.20 | 1744.61 | 1746.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1748.20 | 1744.61 | 1746.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1748.20 | 1744.61 | 1746.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 1743.80 | 1744.88 | 1746.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1765.50 | 1750.52 | 1748.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1765.50 | 1750.52 | 1748.50 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 1746.70 | 1749.28 | 1749.45 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1759.90 | 1750.08 | 1749.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 1782.00 | 1758.79 | 1753.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 10:15:00 | 1824.10 | 1828.56 | 1806.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:45:00 | 1824.80 | 1828.56 | 1806.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1816.00 | 1821.57 | 1808.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1833.60 | 1819.51 | 1809.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1804.50 | 1823.57 | 1817.37 | SL hit (close<static) qty=1.00 sl=1806.50 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 1812.80 | 1816.94 | 1816.97 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 1821.10 | 1817.77 | 1817.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 1828.80 | 1820.82 | 1818.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 15:15:00 | 1820.80 | 1820.95 | 1819.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 09:15:00 | 1822.40 | 1820.95 | 1819.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1826.70 | 1822.10 | 1820.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 1836.00 | 1822.10 | 1820.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 10:15:00 | 1832.50 | 1844.58 | 1845.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 10:15:00 | 1832.50 | 1844.58 | 1845.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 1829.00 | 1835.49 | 1839.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 1836.00 | 1833.78 | 1837.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 14:15:00 | 1836.00 | 1833.78 | 1837.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1836.00 | 1833.78 | 1837.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:45:00 | 1834.20 | 1833.78 | 1837.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 1833.50 | 1833.72 | 1837.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 1841.00 | 1833.72 | 1837.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1831.70 | 1833.32 | 1836.66 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1846.50 | 1838.00 | 1837.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 1856.90 | 1844.52 | 1840.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 1829.40 | 1844.79 | 1843.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 11:15:00 | 1829.40 | 1844.79 | 1843.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1829.40 | 1844.79 | 1843.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 1827.50 | 1844.79 | 1843.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 1830.10 | 1841.86 | 1841.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 09:15:00 | 1816.30 | 1835.73 | 1838.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 12:15:00 | 1784.00 | 1776.73 | 1788.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 13:00:00 | 1784.00 | 1776.73 | 1788.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1788.50 | 1779.99 | 1787.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 1788.00 | 1779.99 | 1787.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1790.00 | 1781.99 | 1787.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1786.40 | 1781.99 | 1787.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:00:00 | 1781.60 | 1769.38 | 1770.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 1779.00 | 1771.30 | 1770.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1779.00 | 1771.30 | 1770.90 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 1768.50 | 1770.53 | 1770.61 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 1772.00 | 1770.83 | 1770.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 1776.20 | 1772.50 | 1771.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 1766.00 | 1771.28 | 1771.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 1766.00 | 1771.28 | 1771.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1766.00 | 1771.28 | 1771.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 1766.00 | 1771.28 | 1771.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 1765.80 | 1770.19 | 1770.68 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1775.00 | 1771.02 | 1770.78 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 1764.10 | 1769.64 | 1770.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 1761.70 | 1767.63 | 1769.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 1773.00 | 1765.48 | 1767.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1773.00 | 1765.48 | 1767.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1773.00 | 1765.48 | 1767.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1772.80 | 1765.48 | 1767.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1769.00 | 1767.23 | 1767.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 10:30:00 | 1767.20 | 1767.08 | 1767.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 1772.80 | 1768.22 | 1768.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 1772.80 | 1768.22 | 1768.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 12:15:00 | 1775.10 | 1769.60 | 1768.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1767.10 | 1769.10 | 1768.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 13:15:00 | 1767.10 | 1769.10 | 1768.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1767.10 | 1769.10 | 1768.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 1767.10 | 1769.10 | 1768.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1764.50 | 1768.18 | 1768.29 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 1772.10 | 1768.50 | 1768.39 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1766.60 | 1768.12 | 1768.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 1758.00 | 1766.10 | 1767.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1759.10 | 1752.84 | 1756.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1759.10 | 1752.84 | 1756.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1759.10 | 1752.84 | 1756.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 1747.20 | 1751.29 | 1754.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 1746.40 | 1750.55 | 1753.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:30:00 | 1747.10 | 1749.13 | 1750.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 1747.20 | 1749.13 | 1750.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1739.00 | 1743.49 | 1747.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 1741.30 | 1743.49 | 1747.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1737.80 | 1734.33 | 1739.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1737.80 | 1734.33 | 1739.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1737.90 | 1735.04 | 1739.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 1735.30 | 1735.97 | 1739.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:15:00 | 1734.00 | 1735.99 | 1738.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 15:15:00 | 1734.90 | 1736.56 | 1738.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 1732.00 | 1736.22 | 1738.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 14:15:00 | 1768.80 | 1741.09 | 1739.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 14:15:00 | 1768.80 | 1741.09 | 1739.47 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 1731.30 | 1738.41 | 1738.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1717.90 | 1732.69 | 1735.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 1727.80 | 1722.77 | 1726.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 13:15:00 | 1727.80 | 1722.77 | 1726.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 1727.80 | 1722.77 | 1726.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 1727.80 | 1722.77 | 1726.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1725.50 | 1723.32 | 1726.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 1718.00 | 1723.32 | 1726.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 1721.70 | 1723.52 | 1726.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:30:00 | 1722.80 | 1724.27 | 1726.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:30:00 | 1724.50 | 1725.25 | 1726.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1724.50 | 1725.10 | 1726.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 1729.00 | 1725.10 | 1726.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1730.00 | 1726.08 | 1726.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 1730.80 | 1726.08 | 1726.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1727.30 | 1726.32 | 1726.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-12 10:15:00 | 1729.00 | 1726.86 | 1726.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1729.00 | 1726.86 | 1726.71 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 09:15:00 | 1713.80 | 1725.23 | 1726.28 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 1740.40 | 1725.54 | 1725.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 1746.00 | 1736.97 | 1731.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 10:15:00 | 1730.00 | 1735.57 | 1731.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 10:15:00 | 1730.00 | 1735.57 | 1731.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1730.00 | 1735.57 | 1731.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 1730.00 | 1735.57 | 1731.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1731.30 | 1734.72 | 1731.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 1731.30 | 1734.72 | 1731.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1730.00 | 1733.78 | 1731.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 1730.00 | 1733.78 | 1731.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1731.70 | 1733.36 | 1731.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:30:00 | 1730.60 | 1733.36 | 1731.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1730.00 | 1732.69 | 1731.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 1730.00 | 1732.69 | 1731.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1737.00 | 1733.55 | 1731.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 1726.10 | 1733.55 | 1731.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1717.00 | 1730.24 | 1730.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 11:15:00 | 1711.10 | 1724.31 | 1727.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1711.00 | 1710.31 | 1717.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 1711.00 | 1710.31 | 1717.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 1716.30 | 1712.02 | 1716.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 1716.30 | 1712.02 | 1716.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 1706.50 | 1710.91 | 1715.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 1727.70 | 1710.91 | 1715.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1727.50 | 1714.23 | 1716.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1727.50 | 1714.23 | 1716.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1725.60 | 1716.50 | 1717.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 1727.50 | 1716.50 | 1717.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1727.50 | 1718.70 | 1718.44 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 1703.90 | 1718.47 | 1718.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 1699.00 | 1710.65 | 1714.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 1580.10 | 1579.31 | 1601.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:15:00 | 1584.50 | 1579.31 | 1601.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1571.40 | 1563.71 | 1572.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 1573.70 | 1563.71 | 1572.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1567.10 | 1564.38 | 1571.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:30:00 | 1561.90 | 1563.91 | 1570.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 1561.50 | 1537.95 | 1536.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 1561.50 | 1537.95 | 1536.47 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 1541.50 | 1544.55 | 1544.60 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1548.00 | 1545.24 | 1544.91 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1533.10 | 1542.81 | 1543.84 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 1549.20 | 1536.43 | 1534.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 1551.40 | 1539.42 | 1536.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1551.60 | 1557.76 | 1548.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 12:00:00 | 1567.10 | 1560.80 | 1551.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1587.50 | 1601.15 | 1587.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1587.50 | 1601.15 | 1587.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1581.00 | 1597.12 | 1586.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-18 13:15:00 | 1581.00 | 1597.12 | 1586.69 | SL hit (close<ema400) qty=1.00 sl=1586.69 alert=retest1 |

### Cycle 54 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 1732.80 | 1743.65 | 1744.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 13:15:00 | 1730.00 | 1738.03 | 1741.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 1706.80 | 1706.37 | 1716.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:00:00 | 1706.80 | 1706.37 | 1716.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1710.90 | 1707.28 | 1715.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1700.00 | 1707.42 | 1714.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 09:15:00 | 1615.00 | 1638.38 | 1660.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 1639.50 | 1633.66 | 1649.33 | SL hit (close>ema200) qty=0.50 sl=1633.66 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 10:15:00 | 1596.40 | 1557.80 | 1557.69 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1568.80 | 1584.14 | 1585.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 1561.80 | 1576.75 | 1581.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 1565.90 | 1555.78 | 1565.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 1565.90 | 1555.78 | 1565.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1565.90 | 1555.78 | 1565.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1560.70 | 1555.78 | 1565.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1558.60 | 1556.34 | 1564.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1545.90 | 1556.34 | 1564.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1563.00 | 1557.67 | 1564.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1563.00 | 1557.67 | 1564.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1568.40 | 1559.82 | 1564.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 1568.40 | 1559.82 | 1564.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 1566.50 | 1561.15 | 1565.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:00:00 | 1559.70 | 1564.48 | 1565.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 1575.90 | 1567.49 | 1566.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 1575.90 | 1567.49 | 1566.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 1584.90 | 1570.97 | 1568.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1624.70 | 1629.43 | 1608.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:30:00 | 1623.60 | 1629.43 | 1608.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1604.00 | 1623.33 | 1608.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1604.00 | 1623.33 | 1608.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1615.40 | 1621.75 | 1609.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1611.00 | 1621.75 | 1609.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1602.90 | 1617.98 | 1608.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 1602.90 | 1617.98 | 1608.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1591.50 | 1612.68 | 1607.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 1591.50 | 1612.68 | 1607.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 1573.90 | 1600.94 | 1603.03 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1621.10 | 1604.39 | 1603.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1660.50 | 1618.35 | 1610.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 1627.00 | 1643.52 | 1630.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 1627.00 | 1643.52 | 1630.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1627.00 | 1643.52 | 1630.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 1627.00 | 1643.52 | 1630.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 1619.50 | 1638.71 | 1629.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 1619.50 | 1638.71 | 1629.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1626.50 | 1634.24 | 1629.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:45:00 | 1626.00 | 1634.24 | 1629.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1630.60 | 1633.51 | 1629.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:30:00 | 1626.90 | 1633.51 | 1629.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1625.00 | 1631.81 | 1628.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 1625.00 | 1631.81 | 1628.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 1629.40 | 1631.33 | 1629.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 1631.00 | 1631.33 | 1629.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1618.40 | 1628.74 | 1628.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 1616.60 | 1628.74 | 1628.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 1609.70 | 1624.93 | 1626.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1594.00 | 1611.12 | 1618.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 1612.80 | 1611.45 | 1617.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:45:00 | 1611.80 | 1611.45 | 1617.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1607.70 | 1610.70 | 1616.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 1616.70 | 1610.70 | 1616.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1629.40 | 1614.44 | 1617.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 1629.40 | 1614.44 | 1617.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1626.00 | 1616.75 | 1618.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:45:00 | 1627.60 | 1616.75 | 1618.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1610.00 | 1615.87 | 1617.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 1647.80 | 1615.87 | 1617.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1637.00 | 1620.10 | 1619.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1654.00 | 1626.88 | 1622.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 10:15:00 | 1671.50 | 1690.89 | 1673.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 1671.50 | 1690.89 | 1673.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1671.50 | 1690.89 | 1673.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1671.50 | 1690.89 | 1673.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1672.50 | 1687.21 | 1673.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 1668.70 | 1687.21 | 1673.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1672.00 | 1684.17 | 1673.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:30:00 | 1678.80 | 1683.14 | 1673.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 1680.10 | 1680.89 | 1673.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1652.20 | 1675.02 | 1672.35 | SL hit (close<static) qty=1.00 sl=1668.80 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 1655.90 | 1668.49 | 1669.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1625.20 | 1656.28 | 1662.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 1658.40 | 1655.25 | 1661.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 1658.40 | 1655.25 | 1661.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1645.00 | 1652.81 | 1658.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 1634.00 | 1652.81 | 1658.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 1654.90 | 1642.85 | 1642.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1654.90 | 1642.85 | 1642.84 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 1632.30 | 1641.03 | 1642.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 1630.00 | 1638.82 | 1641.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 1642.30 | 1638.12 | 1640.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 1642.30 | 1638.12 | 1640.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1642.30 | 1638.12 | 1640.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1642.30 | 1638.12 | 1640.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 1655.00 | 1641.50 | 1641.44 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1624.00 | 1639.30 | 1640.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 1620.10 | 1633.08 | 1637.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 13:15:00 | 1615.60 | 1614.97 | 1622.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 13:45:00 | 1614.40 | 1614.97 | 1622.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 1596.00 | 1582.30 | 1589.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 1596.00 | 1582.30 | 1589.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 1593.30 | 1584.50 | 1589.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:45:00 | 1596.00 | 1584.50 | 1589.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1588.60 | 1581.61 | 1585.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 1588.60 | 1581.61 | 1585.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1585.00 | 1582.28 | 1585.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 1578.00 | 1582.28 | 1585.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1499.10 | 1539.62 | 1558.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1536.50 | 1506.82 | 1526.77 | SL hit (close>ema200) qty=0.50 sl=1506.82 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1372.40 | 1354.65 | 1353.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1378.10 | 1359.34 | 1355.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 1361.00 | 1361.25 | 1357.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 15:15:00 | 1361.00 | 1361.25 | 1357.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1361.00 | 1361.25 | 1357.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1370.00 | 1361.25 | 1357.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 1341.90 | 1357.38 | 1356.00 | SL hit (close<static) qty=1.00 sl=1355.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1340.90 | 1354.08 | 1354.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1311.10 | 1338.76 | 1345.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1338.60 | 1310.24 | 1323.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1338.60 | 1310.24 | 1323.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1338.60 | 1310.24 | 1323.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1340.40 | 1310.24 | 1323.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1335.40 | 1315.27 | 1324.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 1342.60 | 1315.27 | 1324.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1351.80 | 1332.99 | 1331.43 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1313.60 | 1330.25 | 1331.23 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 1346.00 | 1333.40 | 1332.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 1363.30 | 1339.38 | 1335.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 1354.60 | 1357.13 | 1346.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 1354.60 | 1357.13 | 1346.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1354.60 | 1357.13 | 1346.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 15:00:00 | 1391.10 | 1372.01 | 1358.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 1395.50 | 1377.88 | 1363.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:45:00 | 1392.10 | 1380.87 | 1366.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:00:00 | 1390.10 | 1386.35 | 1372.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1424.00 | 1430.67 | 1418.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 1425.10 | 1430.67 | 1418.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 1424.00 | 1429.34 | 1419.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1461.60 | 1429.34 | 1419.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 1530.21 | 1481.66 | 1463.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1678.90 | 1688.61 | 1688.81 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 1723.10 | 1692.68 | 1689.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 1735.00 | 1701.14 | 1693.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 09:15:00 | 1729.30 | 1732.69 | 1721.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 10:00:00 | 1729.30 | 1732.69 | 1721.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1751.00 | 1747.86 | 1736.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 1765.10 | 1749.73 | 1739.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 13:30:00 | 1982.00 | 2025-05-21 11:15:00 | 2060.50 | STOP_HIT | 1.00 | 3.96% |
| BUY | retest2 | 2025-05-14 15:00:00 | 1981.50 | 2025-05-21 11:15:00 | 2060.50 | STOP_HIT | 1.00 | 3.99% |
| BUY | retest2 | 2025-05-15 09:15:00 | 1998.00 | 2025-05-21 11:15:00 | 2060.50 | STOP_HIT | 1.00 | 3.13% |
| SELL | retest2 | 2025-05-28 11:15:00 | 2042.40 | 2025-05-29 10:15:00 | 2108.80 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-06-04 09:15:00 | 2008.00 | 2025-06-13 09:15:00 | 1907.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 11:00:00 | 2007.80 | 2025-06-13 09:15:00 | 1907.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-04 09:15:00 | 2008.00 | 2025-06-16 14:15:00 | 1912.30 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-06-05 11:00:00 | 2007.80 | 2025-06-16 14:15:00 | 1912.30 | STOP_HIT | 0.50 | 4.76% |
| BUY | retest2 | 2025-07-16 09:15:00 | 1987.70 | 2025-07-16 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-24 12:45:00 | 1919.20 | 2025-08-01 13:15:00 | 1823.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 14:00:00 | 1920.00 | 2025-08-01 13:15:00 | 1824.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:15:00 | 1912.30 | 2025-08-01 13:15:00 | 1816.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 12:45:00 | 1919.20 | 2025-08-05 14:15:00 | 1823.70 | STOP_HIT | 0.50 | 4.98% |
| SELL | retest2 | 2025-07-24 14:00:00 | 1920.00 | 2025-08-05 14:15:00 | 1823.70 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest2 | 2025-07-25 09:15:00 | 1912.30 | 2025-08-05 14:15:00 | 1823.70 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2025-08-22 11:15:00 | 1813.00 | 2025-08-25 10:15:00 | 1831.30 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-22 14:30:00 | 1812.50 | 2025-08-25 10:15:00 | 1831.30 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-02 11:45:00 | 1774.70 | 2025-09-10 10:15:00 | 1768.40 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-09-02 13:00:00 | 1775.00 | 2025-09-10 10:15:00 | 1768.40 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-09-11 11:30:00 | 1743.80 | 2025-09-12 09:15:00 | 1765.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-09-19 09:15:00 | 1833.60 | 2025-09-19 14:15:00 | 1804.50 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-09-22 09:45:00 | 1819.70 | 2025-09-23 09:15:00 | 1812.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-09-22 10:45:00 | 1823.00 | 2025-09-23 09:15:00 | 1812.80 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-22 12:30:00 | 1819.90 | 2025-09-23 09:15:00 | 1812.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-09-24 10:15:00 | 1836.00 | 2025-09-29 10:15:00 | 1832.50 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1786.40 | 2025-10-16 10:15:00 | 1779.00 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-10-16 10:00:00 | 1781.60 | 2025-10-16 10:15:00 | 1779.00 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-10-23 10:30:00 | 1767.20 | 2025-10-23 11:15:00 | 1772.80 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-10-28 14:00:00 | 1747.20 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-29 09:45:00 | 1746.40 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-30 11:30:00 | 1747.10 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-30 12:00:00 | 1747.20 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-03 13:00:00 | 1735.30 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-11-03 14:15:00 | 1734.00 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-11-03 15:15:00 | 1734.90 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-11-04 10:15:00 | 1732.00 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-11-10 15:15:00 | 1718.00 | 2025-11-12 10:15:00 | 1729.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-11 09:30:00 | 1721.70 | 2025-11-12 10:15:00 | 1729.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-11-11 11:30:00 | 1722.80 | 2025-11-12 10:15:00 | 1729.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-11-11 13:30:00 | 1724.50 | 2025-11-12 10:15:00 | 1729.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-12-01 12:30:00 | 1561.90 | 2025-12-05 09:15:00 | 1561.50 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest1 | 2025-12-16 12:00:00 | 1567.10 | 2025-12-18 13:15:00 | 1581.00 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1700.00 | 2026-01-07 09:15:00 | 1615.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1700.00 | 2026-01-07 14:15:00 | 1639.50 | STOP_HIT | 0.50 | 3.56% |
| SELL | retest2 | 2026-01-29 10:00:00 | 1559.70 | 2026-01-29 13:15:00 | 1575.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-11 13:30:00 | 1678.80 | 2026-02-12 09:15:00 | 1652.20 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-02-11 15:15:00 | 1680.10 | 2026-02-12 09:15:00 | 1652.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-13 15:15:00 | 1634.00 | 2026-02-17 13:15:00 | 1654.90 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-02-27 15:15:00 | 1578.00 | 2026-03-04 09:15:00 | 1499.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:15:00 | 1578.00 | 2026-03-05 09:15:00 | 1536.50 | STOP_HIT | 0.50 | 2.63% |
| BUY | retest2 | 2026-03-27 09:15:00 | 1370.00 | 2026-03-27 09:15:00 | 1341.90 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-04-06 15:00:00 | 1391.10 | 2026-04-15 09:15:00 | 1530.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:15:00 | 1395.50 | 2026-04-15 09:15:00 | 1535.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:45:00 | 1392.10 | 2026-04-15 09:15:00 | 1531.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 14:00:00 | 1390.10 | 2026-04-15 09:15:00 | 1529.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1461.60 | 2026-04-21 11:15:00 | 1607.76 | TARGET_HIT | 1.00 | 10.00% |
