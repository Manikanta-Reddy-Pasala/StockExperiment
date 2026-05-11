# Lupin Ltd. (LUPIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2373.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 32
- **Target hits / Stop hits / Partials:** 3 / 41 / 6
- **Avg / median % per leg:** 0.29% / -1.04%
- **Sum % (uncompounded):** 14.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 2 | 11 | 0 | 0.67% | 8.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 5 | 38.5% | 2 | 11 | 0 | 0.67% | 8.7% |
| SELL (all) | 37 | 13 | 35.1% | 1 | 30 | 6 | 0.15% | 5.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 13 | 35.1% | 1 | 30 | 6 | 0.15% | 5.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 50 | 18 | 36.0% | 3 | 41 | 6 | 0.29% | 14.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 11:15:00 | 1585.15 | 1598.06 | 1598.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 1581.00 | 1597.74 | 1597.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 1608.95 | 1597.60 | 1597.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 1608.95 | 1597.60 | 1597.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1608.95 | 1597.60 | 1597.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 1609.10 | 1597.60 | 1597.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 1615.15 | 1597.78 | 1597.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 1616.00 | 1597.78 | 1597.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 12:15:00 | 1620.80 | 1598.27 | 1598.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 13:15:00 | 1633.45 | 1598.62 | 1598.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 12:15:00 | 2153.60 | 2163.58 | 2037.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 13:00:00 | 2153.60 | 2163.58 | 2037.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 2125.85 | 2187.34 | 2123.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:45:00 | 2127.95 | 2187.34 | 2123.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 2122.40 | 2186.69 | 2123.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 2153.50 | 2177.37 | 2122.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 11:00:00 | 2133.00 | 2176.42 | 2122.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 13:30:00 | 2140.00 | 2175.06 | 2122.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 11:15:00 | 2102.50 | 2176.90 | 2133.61 | SL hit (close<static) qty=1.00 sl=2112.65 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 2007.20 | 2110.77 | 2110.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 1999.00 | 2104.69 | 2107.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 13:15:00 | 2096.65 | 2092.91 | 2100.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 13:45:00 | 2095.00 | 2092.91 | 2100.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 2100.85 | 2092.99 | 2100.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 2100.85 | 2092.99 | 2100.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 2108.00 | 2093.14 | 2100.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 2107.95 | 2093.14 | 2100.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2088.50 | 2093.10 | 2100.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 10:15:00 | 2087.05 | 2093.10 | 2100.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 13:15:00 | 2117.95 | 2093.31 | 2100.72 | SL hit (close>static) qty=1.00 sl=2114.80 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 2177.30 | 2105.53 | 2105.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 10:15:00 | 2192.85 | 2114.17 | 2109.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 2207.95 | 2220.22 | 2173.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-10 10:00:00 | 2207.95 | 2220.22 | 2173.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 2170.00 | 2218.56 | 2174.69 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 15:15:00 | 2070.00 | 2149.77 | 2149.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 2053.20 | 2144.18 | 2146.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 2177.90 | 2125.41 | 2136.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 2177.90 | 2125.41 | 2136.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 2177.90 | 2125.41 | 2136.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 2177.90 | 2125.41 | 2136.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 2197.90 | 2126.13 | 2136.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 2197.90 | 2126.13 | 2136.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2113.00 | 2128.44 | 2136.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 2133.55 | 2128.44 | 2136.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2038.85 | 2018.25 | 2062.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 11:00:00 | 2022.70 | 2018.29 | 2062.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 11:15:00 | 2094.00 | 2014.15 | 2050.06 | SL hit (close>static) qty=1.00 sl=2080.10 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 2085.80 | 2043.43 | 2043.43 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1985.60 | 2043.56 | 2043.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1977.90 | 2042.91 | 2043.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1998.10 | 1998.10 | 2016.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:00:00 | 1998.10 | 1998.10 | 2016.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 2004.80 | 1997.98 | 2014.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 13:15:00 | 2001.20 | 1997.98 | 2014.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 14:15:00 | 2002.20 | 1998.06 | 2014.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 11:15:00 | 2001.50 | 1998.47 | 2014.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 2024.40 | 1998.96 | 2014.48 | SL hit (close>static) qty=1.00 sl=2017.60 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 2040.50 | 1953.56 | 1953.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 14:15:00 | 2047.30 | 1954.50 | 1953.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 13:15:00 | 1969.70 | 1983.67 | 1970.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 13:15:00 | 1969.70 | 1983.67 | 1970.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1969.70 | 1983.67 | 1970.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 1969.70 | 1983.67 | 1970.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1962.80 | 1983.46 | 1970.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 1962.80 | 1983.46 | 1970.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1963.00 | 1983.26 | 1970.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 1918.40 | 1983.26 | 1970.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1933.80 | 1973.69 | 1967.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 1933.80 | 1973.69 | 1967.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1938.10 | 1973.34 | 1967.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 1971.00 | 1964.56 | 1963.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:30:00 | 1944.20 | 1964.37 | 1963.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 1948.30 | 1964.21 | 1963.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:45:00 | 1944.80 | 1962.42 | 1962.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1949.90 | 1962.12 | 1962.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 11:15:00 | 1949.90 | 1962.12 | 1962.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 1941.10 | 1961.75 | 1961.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 12:15:00 | 1954.20 | 1950.56 | 1955.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 12:15:00 | 1954.20 | 1950.56 | 1955.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1954.20 | 1950.56 | 1955.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 1954.20 | 1950.56 | 1955.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1953.00 | 1950.59 | 1955.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:15:00 | 1957.50 | 1950.59 | 1955.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1953.50 | 1950.62 | 1955.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1940.00 | 1950.67 | 1955.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 1970.00 | 1950.08 | 1955.14 | SL hit (close>static) qty=1.00 sl=1960.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 2020.00 | 1959.81 | 1959.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 2034.30 | 1964.82 | 1962.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 11:15:00 | 1991.60 | 1995.52 | 1980.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 12:00:00 | 1991.60 | 1995.52 | 1980.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 2098.90 | 2135.69 | 2099.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 2105.60 | 2135.69 | 2099.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 2103.40 | 2135.37 | 2099.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 2114.10 | 2135.37 | 2099.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 2112.10 | 2135.09 | 2101.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 2119.10 | 2134.82 | 2101.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 2085.30 | 2134.11 | 2102.13 | SL hit (close<static) qty=1.00 sl=2088.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-25 09:15:00 | 2153.50 | 2024-11-05 11:15:00 | 2102.50 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-10-25 11:00:00 | 2133.00 | 2024-11-05 11:15:00 | 2102.50 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-10-25 13:30:00 | 2140.00 | 2024-11-05 11:15:00 | 2102.50 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-11-05 12:45:00 | 2134.30 | 2024-11-07 14:15:00 | 2112.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-12-05 10:15:00 | 2087.05 | 2024-12-05 13:15:00 | 2117.95 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-12-13 09:30:00 | 2076.55 | 2024-12-19 11:15:00 | 2135.05 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2024-12-18 12:45:00 | 2087.50 | 2024-12-19 11:15:00 | 2135.05 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-03-10 11:00:00 | 2022.70 | 2025-03-20 11:15:00 | 2094.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-03-27 10:00:00 | 2024.00 | 2025-04-03 09:15:00 | 2100.00 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-03-28 15:15:00 | 2024.00 | 2025-04-03 09:15:00 | 2100.00 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-04-04 09:30:00 | 2012.40 | 2025-04-07 09:15:00 | 1811.16 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-08 14:45:00 | 2009.00 | 2025-05-09 13:15:00 | 2042.60 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-05-08 15:15:00 | 2008.00 | 2025-05-09 13:15:00 | 2042.60 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-05-09 12:30:00 | 2011.40 | 2025-05-09 13:15:00 | 2042.60 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-05-12 09:15:00 | 1979.50 | 2025-05-12 14:15:00 | 2042.60 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-06-10 13:15:00 | 2001.20 | 2025-06-11 13:15:00 | 2024.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-10 14:15:00 | 2002.20 | 2025-06-11 13:15:00 | 2024.40 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-06-11 11:15:00 | 2001.50 | 2025-06-11 13:15:00 | 2024.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-13 11:00:00 | 2000.10 | 2025-07-10 09:15:00 | 1900.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 11:00:00 | 2000.10 | 2025-07-15 11:15:00 | 1951.50 | STOP_HIT | 0.50 | 2.43% |
| SELL | retest2 | 2025-06-17 10:45:00 | 1966.00 | 2025-07-29 14:15:00 | 1985.60 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-07-02 10:00:00 | 1966.40 | 2025-07-29 14:15:00 | 1985.60 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-02 12:15:00 | 1971.10 | 2025-07-29 14:15:00 | 1985.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-07-07 12:45:00 | 1971.30 | 2025-08-01 11:15:00 | 1872.54 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1951.40 | 2025-08-01 11:15:00 | 1872.73 | PARTIAL | 0.50 | 4.03% |
| SELL | retest2 | 2025-07-28 15:00:00 | 1957.90 | 2025-08-01 14:15:00 | 1867.70 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2025-07-29 09:15:00 | 1953.60 | 2025-08-01 14:15:00 | 1868.08 | PARTIAL | 0.50 | 4.38% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1957.60 | 2025-08-04 09:15:00 | 1859.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 12:45:00 | 1971.30 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1951.40 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 1.21% |
| SELL | retest2 | 2025-07-28 15:00:00 | 1957.90 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 1.54% |
| SELL | retest2 | 2025-07-29 09:15:00 | 1953.60 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1957.60 | 2025-08-07 12:15:00 | 1927.80 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1921.70 | 2025-08-12 12:15:00 | 1956.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-11 11:00:00 | 1936.00 | 2025-08-12 12:15:00 | 1956.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-08-12 09:30:00 | 1933.80 | 2025-08-12 12:15:00 | 1956.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-08-20 13:00:00 | 1935.00 | 2025-08-21 10:15:00 | 1957.70 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-05 12:30:00 | 1935.10 | 2025-09-08 12:15:00 | 1955.10 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1936.00 | 2025-09-08 12:15:00 | 1955.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-09 09:15:00 | 1971.00 | 2025-10-15 11:15:00 | 1949.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-10 09:30:00 | 1944.20 | 2025-10-15 11:15:00 | 1949.90 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-10-10 11:00:00 | 1948.30 | 2025-10-15 11:15:00 | 1949.90 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-10-15 09:45:00 | 1944.80 | 2025-10-15 11:15:00 | 1949.90 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1940.00 | 2025-10-31 10:15:00 | 1970.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-03 10:45:00 | 1951.80 | 2025-11-03 11:15:00 | 1990.50 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-01-29 11:15:00 | 2114.10 | 2026-02-02 10:15:00 | 2085.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-01 12:30:00 | 2112.10 | 2026-02-02 10:15:00 | 2085.30 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-02-01 14:15:00 | 2119.10 | 2026-02-02 10:15:00 | 2085.30 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-02 14:30:00 | 2111.80 | 2026-02-26 09:15:00 | 2322.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:30:00 | 2238.30 | 2026-05-07 09:15:00 | 2462.13 | TARGET_HIT | 1.00 | 10.00% |
