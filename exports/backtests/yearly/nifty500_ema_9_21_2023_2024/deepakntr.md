# Deepak Nitrite Ltd. (DEEPAKNTR)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 1875.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 213 |
| ALERT1 | 129 |
| ALERT2 | 129 |
| ALERT2_SKIP | 56 |
| ALERT3 | 385 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 172 |
| PARTIAL | 21 |
| TARGET_HIT | 14 |
| STOP_HIT | 164 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 198 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 78 / 120
- **Target hits / Stop hits / Partials:** 14 / 163 / 21
- **Avg / median % per leg:** 0.99% / -0.44%
- **Sum % (uncompounded):** 195.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 88 | 27 | 30.7% | 7 | 81 | 0 | 0.33% | 28.7% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | 0.71% | 2.1% |
| BUY @ 3rd Alert (retest2) | 85 | 25 | 29.4% | 7 | 78 | 0 | 0.31% | 26.5% |
| SELL (all) | 110 | 51 | 46.4% | 7 | 82 | 21 | 1.52% | 166.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.94% | -2.8% |
| SELL @ 3rd Alert (retest2) | 107 | 51 | 47.7% | 7 | 79 | 21 | 1.59% | 169.7% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 6 | 0 | -0.11% | -0.7% |
| retest2 (combined) | 192 | 76 | 39.6% | 14 | 157 | 21 | 1.02% | 196.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 10:15:00 | 1947.85 | 1936.57 | 1935.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 14:15:00 | 1973.00 | 1946.74 | 1940.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 14:15:00 | 1964.90 | 1971.40 | 1959.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-18 15:00:00 | 1964.90 | 1971.40 | 1959.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 1952.50 | 1967.39 | 1959.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:00:00 | 1952.50 | 1967.39 | 1959.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 1949.50 | 1963.82 | 1959.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 11:00:00 | 1949.50 | 1963.82 | 1959.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 13:15:00 | 1941.80 | 1954.06 | 1955.32 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 11:15:00 | 1958.00 | 1950.85 | 1950.57 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 09:15:00 | 1944.00 | 1949.86 | 1950.40 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 2031.60 | 1966.21 | 1957.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 11:15:00 | 2119.20 | 1996.81 | 1972.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 10:15:00 | 2068.35 | 2077.04 | 2032.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 11:00:00 | 2068.35 | 2077.04 | 2032.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 2083.85 | 2081.32 | 2072.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 09:15:00 | 2106.30 | 2082.43 | 2080.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 11:15:00 | 2080.00 | 2108.02 | 2111.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 2080.00 | 2108.02 | 2111.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 12:15:00 | 2051.00 | 2079.21 | 2092.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 2051.80 | 2037.27 | 2052.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 2051.80 | 2037.27 | 2052.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 2051.80 | 2037.27 | 2052.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:45:00 | 2052.20 | 2037.27 | 2052.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 2045.00 | 2038.82 | 2052.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:30:00 | 2051.85 | 2038.82 | 2052.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 11:15:00 | 2061.15 | 2043.28 | 2053.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 12:00:00 | 2061.15 | 2043.28 | 2053.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 12:15:00 | 2058.85 | 2046.40 | 2053.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 12:30:00 | 2062.55 | 2046.40 | 2053.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 09:15:00 | 2078.15 | 2060.09 | 2058.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 10:15:00 | 2128.35 | 2073.74 | 2064.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 15:15:00 | 2172.90 | 2173.31 | 2156.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:15:00 | 2199.70 | 2173.31 | 2156.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 2243.00 | 2255.04 | 2235.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 2243.00 | 2255.04 | 2235.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 2231.95 | 2254.02 | 2241.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-23 09:15:00 | 2231.95 | 2254.02 | 2241.74 | SL hit (close<ema400) qty=1.00 sl=2241.74 alert=retest1 |

### Cycle 8 — SELL (started 2023-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 14:15:00 | 2214.90 | 2231.64 | 2233.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 15:15:00 | 2206.00 | 2226.52 | 2231.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 12:15:00 | 2219.55 | 2212.28 | 2221.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 12:15:00 | 2219.55 | 2212.28 | 2221.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 2219.55 | 2212.28 | 2221.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 13:00:00 | 2219.55 | 2212.28 | 2221.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 2221.55 | 2214.13 | 2221.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:00:00 | 2221.55 | 2214.13 | 2221.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 2232.80 | 2217.87 | 2222.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:45:00 | 2237.00 | 2217.87 | 2222.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 2227.10 | 2219.71 | 2222.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 2228.40 | 2219.71 | 2222.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 2213.95 | 2220.87 | 2222.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:30:00 | 2219.90 | 2220.87 | 2222.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 2193.80 | 2205.83 | 2213.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 10:30:00 | 2183.90 | 2204.21 | 2212.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 12:00:00 | 2192.80 | 2201.93 | 2210.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 14:15:00 | 2193.00 | 2199.76 | 2208.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 09:45:00 | 2186.50 | 2191.17 | 2201.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 2169.25 | 2176.98 | 2187.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 09:30:00 | 2173.00 | 2176.98 | 2187.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 2169.00 | 2168.91 | 2177.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 13:45:00 | 2159.25 | 2170.24 | 2173.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 12:15:00 | 2074.70 | 2113.63 | 2132.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 12:15:00 | 2083.16 | 2113.63 | 2132.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 12:15:00 | 2083.35 | 2113.63 | 2132.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 12:15:00 | 2077.17 | 2113.63 | 2132.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 09:15:00 | 2051.29 | 2077.44 | 2108.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-07-11 09:15:00 | 1965.51 | 2005.76 | 2050.60 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 1974.00 | 1947.44 | 1946.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 11:15:00 | 1982.00 | 1954.35 | 1949.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 10:15:00 | 1975.60 | 1976.00 | 1964.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-18 11:00:00 | 1975.60 | 1976.00 | 1964.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 1959.85 | 1972.77 | 1964.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 11:45:00 | 1962.60 | 1972.77 | 1964.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 1967.75 | 1971.77 | 1964.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 13:30:00 | 1974.50 | 1972.61 | 1965.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 14:00:00 | 1976.00 | 1972.61 | 1965.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 10:45:00 | 1975.90 | 1983.34 | 1981.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 11:15:00 | 1966.65 | 1980.00 | 1980.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 11:15:00 | 1966.65 | 1980.00 | 1980.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 12:15:00 | 1959.25 | 1975.85 | 1978.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 13:15:00 | 1977.00 | 1976.08 | 1978.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 13:15:00 | 1977.00 | 1976.08 | 1978.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 13:15:00 | 1977.00 | 1976.08 | 1978.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 13:45:00 | 1986.40 | 1976.08 | 1978.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 1978.40 | 1976.55 | 1978.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 14:30:00 | 1979.50 | 1976.55 | 1978.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 1980.00 | 1977.24 | 1978.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:15:00 | 1967.90 | 1977.24 | 1978.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1967.00 | 1975.19 | 1977.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 10:45:00 | 1962.50 | 1974.23 | 1976.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 12:15:00 | 1988.00 | 1978.35 | 1978.36 | SL hit (close>static) qty=1.00 sl=1983.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 13:15:00 | 1992.00 | 1981.08 | 1979.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 10:15:00 | 2005.55 | 1991.59 | 1985.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 12:15:00 | 2000.00 | 2004.06 | 1997.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-26 13:00:00 | 2000.00 | 2004.06 | 1997.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 2002.05 | 2003.88 | 1999.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:30:00 | 1994.70 | 2003.88 | 1999.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 2001.50 | 2003.40 | 1999.74 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 14:15:00 | 1977.95 | 1996.58 | 1997.64 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 12:15:00 | 2008.75 | 1996.85 | 1996.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 14:15:00 | 2013.65 | 2001.81 | 1999.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 12:15:00 | 2000.20 | 2005.91 | 2002.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 12:15:00 | 2000.20 | 2005.91 | 2002.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 2000.20 | 2005.91 | 2002.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 12:30:00 | 1997.00 | 2005.91 | 2002.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 13:15:00 | 1999.75 | 2004.68 | 2002.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 13:45:00 | 1997.80 | 2004.68 | 2002.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 2037.65 | 2032.28 | 2021.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:30:00 | 2033.70 | 2032.28 | 2021.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 2023.25 | 2030.48 | 2021.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 2023.25 | 2030.48 | 2021.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 2023.65 | 2029.11 | 2021.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 2024.60 | 2029.11 | 2021.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 2017.85 | 2026.86 | 2021.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:00:00 | 2017.85 | 2026.86 | 2021.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 1996.30 | 2020.75 | 2018.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:30:00 | 1994.60 | 2020.75 | 2018.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 2003.40 | 2015.03 | 2016.59 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 15:15:00 | 2021.00 | 2016.18 | 2016.09 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 09:15:00 | 2014.00 | 2015.75 | 2015.90 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 2032.05 | 2019.01 | 2017.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 12:15:00 | 2046.75 | 2034.88 | 2028.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 15:15:00 | 2038.05 | 2039.68 | 2032.47 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 09:15:00 | 2055.35 | 2039.68 | 2032.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 2060.90 | 2044.79 | 2036.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 12:15:00 | 2065.35 | 2044.79 | 2036.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 14:30:00 | 2070.00 | 2058.43 | 2045.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 11:45:00 | 2065.00 | 2072.45 | 2057.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 13:00:00 | 2070.20 | 2072.00 | 2058.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 15:15:00 | 2065.00 | 2069.25 | 2060.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 09:15:00 | 2048.00 | 2069.25 | 2060.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 2050.70 | 2065.54 | 2059.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-10 09:15:00 | 2050.70 | 2065.54 | 2059.80 | SL hit (close<ema400) qty=1.00 sl=2059.80 alert=retest1 |

### Cycle 18 — SELL (started 2023-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 12:15:00 | 2048.70 | 2067.56 | 2068.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 09:15:00 | 2037.70 | 2055.32 | 2061.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 14:15:00 | 1992.30 | 1990.17 | 2009.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-18 15:00:00 | 1992.30 | 1990.17 | 2009.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 1998.30 | 1993.11 | 2006.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:00:00 | 1998.30 | 1993.11 | 2006.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 2001.00 | 1994.69 | 2005.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:00:00 | 2001.00 | 1994.69 | 2005.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 2011.00 | 1997.95 | 2006.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 13:00:00 | 2011.00 | 1997.95 | 2006.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 2001.70 | 1998.70 | 2005.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-22 09:45:00 | 1995.50 | 2000.46 | 2005.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 09:15:00 | 2024.60 | 2000.07 | 1999.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 09:15:00 | 2024.60 | 2000.07 | 1999.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 10:15:00 | 2029.95 | 2006.05 | 2002.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 09:15:00 | 2012.30 | 2020.86 | 2013.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 2012.30 | 2020.86 | 2013.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 2012.30 | 2020.86 | 2013.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:15:00 | 1999.00 | 2020.86 | 2013.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 1997.00 | 2016.09 | 2011.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 1996.80 | 2016.09 | 2011.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 2011.10 | 2013.40 | 2011.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:15:00 | 2017.15 | 2013.40 | 2011.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 2020.00 | 2014.72 | 2012.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 11:00:00 | 2031.20 | 2018.01 | 2014.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-04 09:15:00 | 2234.32 | 2221.86 | 2202.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 2289.80 | 2320.54 | 2321.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 11:15:00 | 2252.50 | 2306.93 | 2314.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 2269.20 | 2262.06 | 2281.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 13:00:00 | 2269.20 | 2262.06 | 2281.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 2280.95 | 2266.87 | 2280.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 15:00:00 | 2280.95 | 2266.87 | 2280.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 2281.70 | 2269.84 | 2280.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 2300.00 | 2269.84 | 2280.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 2288.00 | 2273.47 | 2281.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 10:30:00 | 2278.40 | 2273.14 | 2280.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 09:15:00 | 2164.48 | 2189.09 | 2212.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-25 09:15:00 | 2152.05 | 2151.66 | 2178.45 | SL hit (close>ema200) qty=0.50 sl=2151.66 alert=retest2 |

### Cycle 21 — BUY (started 2023-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 14:15:00 | 2111.35 | 2107.59 | 2107.13 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 2077.85 | 2101.70 | 2104.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 2065.00 | 2077.82 | 2089.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 2110.60 | 2079.58 | 2083.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 09:15:00 | 2110.60 | 2079.58 | 2083.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 2110.60 | 2079.58 | 2083.12 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 2093.00 | 2086.19 | 2085.75 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 12:15:00 | 2085.00 | 2086.94 | 2086.99 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-10-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 14:15:00 | 2089.10 | 2087.19 | 2087.08 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 09:15:00 | 2081.00 | 2086.16 | 2086.64 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 10:15:00 | 2097.95 | 2087.91 | 2086.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 14:15:00 | 2109.65 | 2096.15 | 2091.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 09:15:00 | 2117.80 | 2119.55 | 2109.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-18 09:45:00 | 2120.80 | 2119.55 | 2109.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 2095.00 | 2114.64 | 2108.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 2095.00 | 2114.64 | 2108.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 2097.70 | 2111.25 | 2107.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:00:00 | 2097.70 | 2111.25 | 2107.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 13:15:00 | 2114.45 | 2111.72 | 2108.21 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 10:15:00 | 2095.00 | 2104.96 | 2105.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 2084.70 | 2095.34 | 2100.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1980.00 | 1964.75 | 1987.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-27 10:00:00 | 1980.00 | 1964.75 | 1987.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 2000.95 | 1971.99 | 1989.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 2000.95 | 1971.99 | 1989.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 1996.65 | 1976.92 | 1989.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 12:45:00 | 1986.05 | 1978.70 | 1989.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 15:00:00 | 1986.00 | 1981.15 | 1988.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-30 09:15:00 | 1982.20 | 1983.12 | 1988.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-30 10:45:00 | 1989.00 | 1982.19 | 1987.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 11:15:00 | 1978.00 | 1981.35 | 1986.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 11:30:00 | 1987.40 | 1981.35 | 1986.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 13:15:00 | 1993.00 | 1984.10 | 1986.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 13:45:00 | 1992.65 | 1984.10 | 1986.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 14:15:00 | 1991.80 | 1985.64 | 1987.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 14:30:00 | 1995.00 | 1985.64 | 1987.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 2000.55 | 1988.54 | 1988.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 2000.55 | 1988.54 | 1988.41 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 09:15:00 | 1978.85 | 1988.27 | 1988.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 1969.55 | 1980.63 | 1984.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 1975.30 | 1973.11 | 1979.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 1975.30 | 1973.11 | 1979.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 1975.30 | 1973.11 | 1979.37 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 14:15:00 | 2001.90 | 1983.86 | 1982.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 2007.75 | 1991.05 | 1985.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 12:15:00 | 2097.65 | 2106.40 | 2074.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 12:45:00 | 2100.50 | 2106.40 | 2074.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 2107.75 | 2130.71 | 2112.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 2110.15 | 2130.71 | 2112.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 2090.00 | 2122.57 | 2110.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:30:00 | 2084.90 | 2122.57 | 2110.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-11-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 13:15:00 | 2075.60 | 2099.97 | 2101.80 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 13:15:00 | 2108.95 | 2100.22 | 2099.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-12 18:15:00 | 2119.00 | 2107.40 | 2103.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 09:15:00 | 2107.00 | 2107.32 | 2103.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-13 10:15:00 | 2105.80 | 2107.32 | 2103.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 2101.90 | 2106.24 | 2103.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 12:30:00 | 2109.60 | 2106.78 | 2104.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 09:15:00 | 2117.70 | 2109.30 | 2106.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-20 13:15:00 | 2118.95 | 2134.65 | 2134.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 13:15:00 | 2118.95 | 2134.65 | 2134.88 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 09:15:00 | 2156.55 | 2134.69 | 2134.42 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 13:15:00 | 2129.25 | 2134.52 | 2134.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 14:15:00 | 2124.40 | 2132.50 | 2133.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 2140.45 | 2132.90 | 2133.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 2140.45 | 2132.90 | 2133.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 2140.45 | 2132.90 | 2133.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:30:00 | 2144.15 | 2132.90 | 2133.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 2136.60 | 2133.64 | 2133.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 11:15:00 | 2149.15 | 2133.64 | 2133.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 2132.40 | 2131.45 | 2132.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 14:45:00 | 2132.00 | 2131.45 | 2132.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 15:15:00 | 2136.00 | 2132.36 | 2132.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:15:00 | 2165.75 | 2132.36 | 2132.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 09:15:00 | 2205.55 | 2147.00 | 2139.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 11:15:00 | 2227.50 | 2203.93 | 2179.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 14:15:00 | 2196.05 | 2205.12 | 2186.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-24 15:00:00 | 2196.05 | 2205.12 | 2186.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 2178.15 | 2197.04 | 2187.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 2178.15 | 2197.04 | 2187.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 2175.50 | 2192.74 | 2186.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:45:00 | 2170.85 | 2192.74 | 2186.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 15:15:00 | 2175.00 | 2181.81 | 2182.26 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 09:15:00 | 2187.00 | 2182.85 | 2182.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 13:15:00 | 2205.00 | 2191.45 | 2187.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 2187.70 | 2194.43 | 2189.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 2187.70 | 2194.43 | 2189.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 2187.70 | 2194.43 | 2189.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 2182.90 | 2194.43 | 2189.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 2198.40 | 2195.23 | 2190.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:30:00 | 2191.05 | 2195.23 | 2190.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 2189.10 | 2195.68 | 2192.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 14:00:00 | 2189.10 | 2195.68 | 2192.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 14:15:00 | 2197.75 | 2196.10 | 2192.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 09:15:00 | 2217.40 | 2196.88 | 2193.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 11:15:00 | 2209.80 | 2197.16 | 2194.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 11:15:00 | 2219.90 | 2242.40 | 2243.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 2219.90 | 2242.40 | 2243.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 2196.80 | 2233.28 | 2239.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 2224.25 | 2222.78 | 2231.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 2224.25 | 2222.78 | 2231.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 2224.25 | 2222.78 | 2231.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 14:30:00 | 2205.00 | 2226.81 | 2228.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 09:45:00 | 2209.30 | 2225.16 | 2227.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 10:30:00 | 2207.85 | 2225.99 | 2227.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 13:15:00 | 2239.95 | 2228.21 | 2228.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 13:15:00 | 2239.95 | 2228.21 | 2228.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 09:15:00 | 2253.40 | 2238.49 | 2233.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 13:15:00 | 2302.85 | 2303.75 | 2278.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-15 14:00:00 | 2302.85 | 2303.75 | 2278.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 2323.40 | 2307.31 | 2286.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 10:30:00 | 2337.00 | 2311.66 | 2290.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 11:30:00 | 2331.00 | 2313.34 | 2293.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-19 09:15:00 | 2269.95 | 2306.04 | 2297.78 | SL hit (close<static) qty=1.00 sl=2285.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 14:15:00 | 2288.60 | 2294.22 | 2294.36 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 2363.05 | 2306.51 | 2299.83 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 2217.00 | 2295.84 | 2300.09 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 2388.00 | 2304.25 | 2295.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 2429.75 | 2364.30 | 2332.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 2442.90 | 2448.16 | 2410.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:30:00 | 2444.40 | 2448.16 | 2410.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 2431.30 | 2447.05 | 2433.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 15:00:00 | 2431.30 | 2447.05 | 2433.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 2443.00 | 2446.24 | 2434.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 10:00:00 | 2454.25 | 2447.84 | 2435.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:00:00 | 2455.80 | 2476.23 | 2469.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:45:00 | 2448.95 | 2470.75 | 2467.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 11:30:00 | 2447.25 | 2468.19 | 2466.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 2458.00 | 2469.78 | 2468.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:30:00 | 2442.20 | 2469.78 | 2468.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 2471.15 | 2470.05 | 2468.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 11:15:00 | 2475.95 | 2470.05 | 2468.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 12:45:00 | 2474.60 | 2472.65 | 2470.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 13:15:00 | 2474.70 | 2472.65 | 2470.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 14:45:00 | 2476.00 | 2474.31 | 2471.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 15:15:00 | 2473.00 | 2474.05 | 2471.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 09:15:00 | 2483.20 | 2474.05 | 2471.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 10:00:00 | 2506.10 | 2480.46 | 2474.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 12:15:00 | 2465.00 | 2479.60 | 2479.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 12:15:00 | 2465.00 | 2479.60 | 2479.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 13:15:00 | 2437.95 | 2471.27 | 2476.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 10:15:00 | 2440.60 | 2440.55 | 2451.82 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 11:45:00 | 2433.10 | 2438.69 | 2449.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 13:45:00 | 2423.15 | 2435.42 | 2446.51 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 2440.85 | 2436.51 | 2446.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 15:00:00 | 2440.85 | 2436.51 | 2446.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 2410.00 | 2432.08 | 2442.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 13:30:00 | 2400.00 | 2420.46 | 2433.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 09:15:00 | 2449.00 | 2428.45 | 2433.78 | SL hit (close>ema400) qty=1.00 sl=2433.78 alert=retest1 |

### Cycle 47 — BUY (started 2024-01-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 11:15:00 | 2471.35 | 2440.42 | 2438.51 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 14:15:00 | 2433.60 | 2442.83 | 2443.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 09:15:00 | 2419.60 | 2436.13 | 2440.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 12:15:00 | 2435.00 | 2432.05 | 2437.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 12:15:00 | 2435.00 | 2432.05 | 2437.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 12:15:00 | 2435.00 | 2432.05 | 2437.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 13:00:00 | 2435.00 | 2432.05 | 2437.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 2425.05 | 2430.59 | 2435.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 14:30:00 | 2434.25 | 2430.59 | 2435.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 2429.00 | 2430.77 | 2434.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 10:45:00 | 2423.55 | 2429.55 | 2433.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 11:45:00 | 2420.35 | 2427.47 | 2432.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 2302.37 | 2348.20 | 2380.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 2299.33 | 2348.20 | 2380.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-19 09:15:00 | 2383.00 | 2324.41 | 2347.85 | SL hit (close>ema200) qty=0.50 sl=2324.41 alert=retest2 |

### Cycle 49 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 2293.15 | 2253.73 | 2251.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 13:15:00 | 2299.20 | 2269.59 | 2259.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 10:15:00 | 2283.60 | 2289.47 | 2273.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 11:00:00 | 2283.60 | 2289.47 | 2273.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 2249.00 | 2281.37 | 2271.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 2249.00 | 2281.37 | 2271.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 2246.60 | 2274.42 | 2269.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:30:00 | 2244.90 | 2274.42 | 2269.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 14:15:00 | 2250.00 | 2264.91 | 2265.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 11:15:00 | 2231.05 | 2249.06 | 2256.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 2242.95 | 2233.94 | 2244.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 2242.95 | 2233.94 | 2244.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 2242.95 | 2233.94 | 2244.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:45:00 | 2253.00 | 2233.94 | 2244.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 2255.90 | 2238.33 | 2245.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 10:45:00 | 2253.25 | 2238.33 | 2245.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 11:15:00 | 2257.55 | 2242.18 | 2246.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 12:00:00 | 2257.55 | 2242.18 | 2246.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 14:15:00 | 2246.90 | 2245.95 | 2247.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 14:30:00 | 2249.15 | 2245.95 | 2247.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 15:15:00 | 2250.00 | 2246.76 | 2247.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:15:00 | 2253.00 | 2246.76 | 2247.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 2243.40 | 2246.09 | 2247.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:15:00 | 2231.00 | 2243.64 | 2246.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 14:45:00 | 2232.20 | 2239.21 | 2242.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 10:30:00 | 2225.45 | 2233.08 | 2239.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 09:45:00 | 2228.00 | 2213.42 | 2219.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 2205.10 | 2211.75 | 2218.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 10:30:00 | 2211.00 | 2211.75 | 2218.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 2213.30 | 2208.60 | 2214.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 14:00:00 | 2213.30 | 2208.60 | 2214.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 2190.55 | 2204.99 | 2212.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 14:30:00 | 2209.30 | 2204.99 | 2212.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 2222.05 | 2206.39 | 2211.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:00:00 | 2222.05 | 2206.39 | 2211.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 2206.95 | 2206.50 | 2210.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:30:00 | 2223.90 | 2206.50 | 2210.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 2221.00 | 2209.40 | 2211.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:00:00 | 2221.00 | 2209.40 | 2211.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-13 13:15:00 | 2228.60 | 2213.24 | 2213.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 13:15:00 | 2228.60 | 2213.24 | 2213.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 10:15:00 | 2249.90 | 2228.03 | 2220.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 2302.35 | 2302.76 | 2285.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 09:45:00 | 2296.35 | 2302.76 | 2285.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 2287.25 | 2299.66 | 2285.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:45:00 | 2281.55 | 2299.66 | 2285.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 2288.40 | 2297.40 | 2285.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 2318.75 | 2290.48 | 2285.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-20 09:15:00 | 2282.80 | 2288.94 | 2285.51 | SL hit (close<static) qty=1.00 sl=2285.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 14:15:00 | 2308.00 | 2323.22 | 2325.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 10:15:00 | 2303.75 | 2314.34 | 2320.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 2215.65 | 2204.28 | 2226.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 2215.65 | 2204.28 | 2226.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 2215.30 | 2207.88 | 2224.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 11:15:00 | 2206.00 | 2215.85 | 2220.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 11:45:00 | 2205.40 | 2213.19 | 2218.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 11:00:00 | 2202.15 | 2201.64 | 2209.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 12:00:00 | 2201.40 | 2201.59 | 2208.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 2202.30 | 2200.61 | 2206.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:30:00 | 2202.90 | 2200.61 | 2206.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 2191.95 | 2179.28 | 2189.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 2191.95 | 2179.28 | 2189.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 2190.05 | 2181.43 | 2189.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 2233.10 | 2181.43 | 2189.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 2237.20 | 2192.58 | 2193.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-07 09:15:00 | 2237.20 | 2192.58 | 2193.95 | SL hit (close>static) qty=1.00 sl=2234.80 alert=retest2 |

### Cycle 53 — BUY (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 10:15:00 | 2232.40 | 2200.55 | 2197.44 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 2187.05 | 2197.69 | 2199.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 2166.00 | 2188.13 | 2194.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 2096.35 | 2092.17 | 2119.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:45:00 | 2100.65 | 2092.17 | 2119.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 2093.20 | 2094.24 | 2108.44 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 09:15:00 | 2114.85 | 2108.32 | 2107.84 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 09:15:00 | 2070.10 | 2105.70 | 2107.74 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 2128.50 | 2103.70 | 2103.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 10:15:00 | 2134.55 | 2109.87 | 2106.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 2144.30 | 2152.93 | 2140.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 10:00:00 | 2144.30 | 2152.93 | 2140.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 2148.05 | 2151.96 | 2140.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 10:30:00 | 2140.75 | 2151.96 | 2140.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 2137.00 | 2147.48 | 2142.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 15:00:00 | 2137.00 | 2147.48 | 2142.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 2138.00 | 2145.58 | 2141.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 2145.65 | 2145.58 | 2141.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 09:15:00 | 2132.15 | 2142.89 | 2140.84 | SL hit (close<static) qty=1.00 sl=2134.55 alert=retest2 |

### Cycle 58 — SELL (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 11:15:00 | 2131.90 | 2145.70 | 2146.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 14:15:00 | 2123.20 | 2139.70 | 2143.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 11:15:00 | 2147.85 | 2136.80 | 2140.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 11:15:00 | 2147.85 | 2136.80 | 2140.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 11:15:00 | 2147.85 | 2136.80 | 2140.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 12:00:00 | 2147.85 | 2136.80 | 2140.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 12:15:00 | 2149.25 | 2139.29 | 2140.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 13:00:00 | 2149.25 | 2139.29 | 2140.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 2144.30 | 2140.13 | 2140.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 14:30:00 | 2142.65 | 2140.13 | 2140.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 15:15:00 | 2145.00 | 2141.11 | 2141.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 09:15:00 | 2156.00 | 2141.11 | 2141.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 09:15:00 | 2173.25 | 2147.54 | 2144.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 10:15:00 | 2186.70 | 2155.37 | 2148.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 10:15:00 | 2184.85 | 2186.09 | 2171.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 10:30:00 | 2183.30 | 2186.09 | 2171.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 2186.30 | 2187.47 | 2178.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 09:45:00 | 2183.35 | 2187.47 | 2178.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 2169.10 | 2183.79 | 2177.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 11:00:00 | 2169.10 | 2183.79 | 2177.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 2202.05 | 2187.44 | 2180.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 12:30:00 | 2221.35 | 2195.40 | 2184.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 15:15:00 | 2217.00 | 2203.62 | 2190.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 15:00:00 | 2219.05 | 2207.48 | 2198.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 10:15:00 | 2222.05 | 2210.69 | 2201.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 2209.25 | 2217.93 | 2209.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 13:30:00 | 2208.30 | 2217.93 | 2209.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 2207.75 | 2215.89 | 2209.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 15:00:00 | 2207.75 | 2215.89 | 2209.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 15:15:00 | 2208.00 | 2214.31 | 2208.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 09:15:00 | 2216.55 | 2214.31 | 2208.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 13:15:00 | 2190.00 | 2213.10 | 2211.04 | SL hit (close<static) qty=1.00 sl=2201.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 14:15:00 | 2193.50 | 2209.18 | 2209.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 15:15:00 | 2187.00 | 2204.74 | 2207.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 2216.60 | 2207.12 | 2208.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 2216.60 | 2207.12 | 2208.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 2216.60 | 2207.12 | 2208.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:00:00 | 2216.60 | 2207.12 | 2208.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 10:15:00 | 2257.35 | 2217.16 | 2212.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 11:15:00 | 2287.35 | 2231.20 | 2219.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 10:15:00 | 2303.00 | 2307.18 | 2284.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-15 11:00:00 | 2303.00 | 2307.18 | 2284.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 2295.00 | 2302.31 | 2290.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:15:00 | 2291.95 | 2302.31 | 2290.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 2304.50 | 2302.75 | 2291.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 10:45:00 | 2315.00 | 2307.09 | 2294.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 10:15:00 | 2248.75 | 2304.79 | 2310.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 2248.75 | 2304.79 | 2310.63 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 2328.05 | 2306.84 | 2304.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 10:15:00 | 2347.10 | 2314.89 | 2308.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 11:15:00 | 2457.80 | 2462.48 | 2437.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 11:30:00 | 2451.95 | 2462.48 | 2437.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 2443.70 | 2456.79 | 2441.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 14:45:00 | 2439.55 | 2456.79 | 2441.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 2442.65 | 2453.96 | 2441.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:15:00 | 2427.95 | 2453.96 | 2441.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 2443.90 | 2451.95 | 2441.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 11:15:00 | 2461.55 | 2449.57 | 2441.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 11:00:00 | 2458.65 | 2447.59 | 2444.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 12:00:00 | 2459.55 | 2449.98 | 2445.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 12:15:00 | 2430.20 | 2448.10 | 2450.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-05-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 12:15:00 | 2430.20 | 2448.10 | 2450.10 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 09:15:00 | 2558.45 | 2468.06 | 2458.42 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 09:15:00 | 2480.00 | 2501.23 | 2502.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 10:15:00 | 2457.00 | 2492.38 | 2498.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 2460.65 | 2453.62 | 2472.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 10:00:00 | 2460.65 | 2453.62 | 2472.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 2492.00 | 2459.52 | 2467.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 2492.00 | 2459.52 | 2467.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 2499.75 | 2467.57 | 2470.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 2482.25 | 2467.57 | 2470.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 12:15:00 | 2471.55 | 2462.73 | 2466.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 13:00:00 | 2471.55 | 2462.73 | 2466.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 2467.00 | 2463.58 | 2466.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 13:45:00 | 2470.35 | 2463.58 | 2466.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 2475.50 | 2465.97 | 2467.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:00:00 | 2475.50 | 2465.97 | 2467.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 2479.45 | 2468.66 | 2468.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 09:15:00 | 2471.55 | 2468.66 | 2468.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 10:15:00 | 2485.00 | 2470.18 | 2469.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 2485.00 | 2470.18 | 2469.30 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 2452.40 | 2467.55 | 2469.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 11:15:00 | 2437.45 | 2458.25 | 2464.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 2441.30 | 2439.33 | 2449.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 12:00:00 | 2441.30 | 2439.33 | 2449.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 2447.25 | 2439.17 | 2446.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 2447.25 | 2439.17 | 2446.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 2453.00 | 2441.94 | 2447.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 2470.65 | 2441.94 | 2447.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 2468.95 | 2447.34 | 2449.22 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 2482.90 | 2454.45 | 2452.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 2518.00 | 2479.16 | 2466.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 12:15:00 | 2512.35 | 2513.23 | 2499.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 13:00:00 | 2512.35 | 2513.23 | 2499.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 2495.15 | 2509.61 | 2499.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 2495.15 | 2509.61 | 2499.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 2471.50 | 2501.99 | 2496.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 2468.00 | 2501.99 | 2496.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 2460.00 | 2493.59 | 2493.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 2406.50 | 2493.59 | 2493.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 2387.00 | 2472.27 | 2483.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 10:15:00 | 2352.50 | 2448.32 | 2471.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 12:15:00 | 2371.50 | 2366.63 | 2401.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 12:45:00 | 2370.60 | 2366.63 | 2401.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2222.65 | 2198.89 | 2220.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 2178.20 | 2227.19 | 2227.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:45:00 | 2191.15 | 2216.84 | 2222.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 2069.29 | 2191.78 | 2209.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 2081.59 | 2191.78 | 2209.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 2197.80 | 2176.39 | 2195.17 | SL hit (close>ema200) qty=0.50 sl=2176.39 alert=retest2 |

### Cycle 71 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 2228.30 | 2207.95 | 2205.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 2267.35 | 2222.51 | 2212.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 13:15:00 | 2412.65 | 2413.19 | 2384.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 14:00:00 | 2412.65 | 2413.19 | 2384.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 2393.30 | 2409.75 | 2390.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 2393.30 | 2409.75 | 2390.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 2408.90 | 2409.58 | 2391.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 12:00:00 | 2417.05 | 2411.08 | 2394.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 14:15:00 | 2415.90 | 2414.75 | 2398.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 15:15:00 | 2505.95 | 2524.02 | 2525.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 2505.95 | 2524.02 | 2525.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 2500.55 | 2519.32 | 2523.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 11:15:00 | 2480.80 | 2479.51 | 2494.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 11:15:00 | 2480.80 | 2479.51 | 2494.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 2480.80 | 2479.51 | 2494.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 2483.60 | 2479.51 | 2494.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 2508.40 | 2486.17 | 2495.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:00:00 | 2508.40 | 2486.17 | 2495.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 2509.50 | 2490.84 | 2496.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:15:00 | 2522.95 | 2490.84 | 2496.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 2522.95 | 2497.26 | 2499.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 2500.00 | 2497.26 | 2499.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 2521.25 | 2502.06 | 2501.04 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 2481.20 | 2497.43 | 2499.24 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 2511.00 | 2501.90 | 2501.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 2553.00 | 2516.89 | 2510.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 10:15:00 | 2663.00 | 2669.91 | 2628.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 11:00:00 | 2663.00 | 2669.91 | 2628.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 2645.10 | 2659.81 | 2640.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:45:00 | 2685.15 | 2662.50 | 2645.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:45:00 | 2687.75 | 2666.12 | 2648.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:45:00 | 2680.45 | 2669.03 | 2651.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:30:00 | 2718.90 | 2678.17 | 2660.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 2698.40 | 2692.22 | 2678.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 2675.85 | 2692.22 | 2678.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 2689.25 | 2692.47 | 2681.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:30:00 | 2682.45 | 2692.47 | 2681.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 2674.80 | 2688.93 | 2681.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:30:00 | 2678.60 | 2688.93 | 2681.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 2694.80 | 2690.11 | 2682.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 2712.65 | 2691.09 | 2683.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 13:30:00 | 2729.35 | 2705.72 | 2695.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 2661.55 | 2694.61 | 2692.95 | SL hit (close<static) qty=1.00 sl=2670.05 alert=retest2 |

### Cycle 76 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 2678.00 | 2691.29 | 2691.59 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 11:15:00 | 2712.65 | 2695.56 | 2693.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 12:15:00 | 2720.20 | 2700.49 | 2695.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 15:15:00 | 2765.00 | 2769.14 | 2750.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 09:15:00 | 2772.80 | 2769.14 | 2750.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2765.55 | 2768.42 | 2752.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 2761.70 | 2768.42 | 2752.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 2800.00 | 2805.66 | 2792.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 2825.00 | 2805.66 | 2792.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 2823.15 | 2809.16 | 2794.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:15:00 | 2843.65 | 2823.21 | 2810.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 15:15:00 | 2764.90 | 2798.98 | 2802.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 2764.90 | 2798.98 | 2802.90 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 2817.65 | 2805.98 | 2805.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 12:15:00 | 2852.10 | 2815.20 | 2809.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 2823.75 | 2830.74 | 2820.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 2823.75 | 2830.74 | 2820.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 2823.75 | 2830.74 | 2820.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:00:00 | 2823.75 | 2830.74 | 2820.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 2864.50 | 2837.50 | 2824.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 10:00:00 | 2873.00 | 2841.22 | 2830.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-01 09:15:00 | 3160.30 | 3098.66 | 3057.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 2983.55 | 3078.52 | 3079.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 2975.45 | 3043.84 | 3062.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3030.30 | 3008.84 | 3036.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 3030.30 | 3008.84 | 3036.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3030.30 | 3008.84 | 3036.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:30:00 | 3024.00 | 3008.84 | 3036.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 3068.00 | 3020.67 | 3039.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 3068.00 | 3020.67 | 3039.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 3063.40 | 3029.21 | 3041.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 12:00:00 | 3063.40 | 3029.21 | 3041.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 3056.15 | 3034.60 | 3043.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 3041.05 | 3035.63 | 3042.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 3039.75 | 3035.63 | 3042.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 3035.40 | 3032.10 | 3040.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:30:00 | 3048.20 | 3029.59 | 3035.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 3076.30 | 3038.93 | 3039.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 12:15:00 | 3076.30 | 3038.93 | 3039.62 | SL hit (close>static) qty=1.00 sl=3068.95 alert=retest2 |

### Cycle 81 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 3107.25 | 3052.59 | 3045.77 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 3028.35 | 3067.09 | 3070.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 3005.50 | 3045.99 | 3059.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 2878.50 | 2875.98 | 2931.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:00:00 | 2878.50 | 2875.98 | 2931.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 2896.65 | 2877.52 | 2900.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 2896.65 | 2877.52 | 2900.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 2903.10 | 2882.64 | 2900.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:45:00 | 2904.55 | 2882.64 | 2900.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 2902.00 | 2886.51 | 2900.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 2900.50 | 2886.51 | 2900.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 2886.45 | 2886.50 | 2899.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 2875.25 | 2886.50 | 2899.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 13:30:00 | 2876.55 | 2881.67 | 2893.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 09:30:00 | 2872.30 | 2878.30 | 2888.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 2900.85 | 2888.88 | 2887.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 2900.85 | 2888.88 | 2887.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 14:15:00 | 2952.70 | 2910.15 | 2898.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 2955.20 | 2966.74 | 2941.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:45:00 | 2949.05 | 2966.74 | 2941.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 2941.60 | 2961.71 | 2941.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 2895.00 | 2961.71 | 2941.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2856.50 | 2940.67 | 2933.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 2856.50 | 2940.67 | 2933.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 2869.20 | 2926.38 | 2927.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 2821.05 | 2880.15 | 2903.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 2867.00 | 2837.14 | 2858.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 2867.00 | 2837.14 | 2858.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 2867.00 | 2837.14 | 2858.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 2877.75 | 2837.14 | 2858.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 2880.00 | 2845.72 | 2860.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 2877.00 | 2845.72 | 2860.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 2869.25 | 2864.41 | 2865.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 2854.95 | 2864.41 | 2865.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 2865.90 | 2864.71 | 2865.92 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 2891.60 | 2870.09 | 2868.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 2911.95 | 2899.42 | 2889.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 2902.00 | 2906.45 | 2896.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 12:00:00 | 2902.00 | 2906.45 | 2896.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 2913.15 | 2907.79 | 2897.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 2907.65 | 2907.79 | 2897.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 2941.60 | 2916.33 | 2905.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 2956.90 | 2927.31 | 2916.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 2928.85 | 2967.68 | 2969.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 2928.85 | 2967.68 | 2969.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 2913.15 | 2951.58 | 2961.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 11:15:00 | 2950.40 | 2949.75 | 2958.67 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 12:15:00 | 2935.00 | 2949.75 | 2958.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 2967.00 | 2947.90 | 2953.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 2967.00 | 2947.90 | 2953.19 | SL hit (close>ema400) qty=1.00 sl=2953.19 alert=retest1 |

### Cycle 87 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 2975.60 | 2958.20 | 2956.58 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 2931.95 | 2953.82 | 2955.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 2916.00 | 2946.26 | 2952.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 11:15:00 | 2937.15 | 2935.55 | 2944.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 12:00:00 | 2937.15 | 2935.55 | 2944.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 2954.60 | 2933.49 | 2939.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 2968.00 | 2933.49 | 2939.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 2945.10 | 2935.81 | 2939.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:15:00 | 2960.50 | 2935.81 | 2939.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 2934.45 | 2939.86 | 2940.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 15:15:00 | 2925.15 | 2939.86 | 2940.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:30:00 | 2926.35 | 2929.30 | 2933.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 2927.20 | 2930.44 | 2933.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 15:15:00 | 2869.00 | 2862.92 | 2862.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 15:15:00 | 2869.00 | 2862.92 | 2862.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 2890.55 | 2868.44 | 2864.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 2870.00 | 2872.60 | 2868.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 13:15:00 | 2870.00 | 2872.60 | 2868.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 2870.00 | 2872.60 | 2868.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 2870.00 | 2872.60 | 2868.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 2860.70 | 2870.22 | 2867.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 2860.70 | 2870.22 | 2867.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 2859.00 | 2867.98 | 2867.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 2856.50 | 2867.98 | 2867.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 2841.00 | 2862.58 | 2864.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 12:15:00 | 2820.40 | 2849.74 | 2858.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 2817.80 | 2812.74 | 2830.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 2817.80 | 2812.74 | 2830.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 2823.50 | 2814.89 | 2829.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 2861.15 | 2814.89 | 2829.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 2829.65 | 2817.85 | 2829.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 2839.70 | 2817.85 | 2829.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 2881.00 | 2830.48 | 2834.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 2881.00 | 2830.48 | 2834.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 2872.20 | 2838.82 | 2837.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 13:15:00 | 2902.65 | 2858.76 | 2847.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 2936.30 | 2944.30 | 2921.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:00:00 | 2936.30 | 2944.30 | 2921.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 2902.05 | 2935.85 | 2919.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 2902.05 | 2935.85 | 2919.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 2890.00 | 2926.68 | 2916.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 2890.00 | 2926.68 | 2916.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 2886.00 | 2909.15 | 2910.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 2855.95 | 2891.91 | 2900.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 2804.65 | 2792.01 | 2824.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 2804.65 | 2792.01 | 2824.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2809.45 | 2796.04 | 2812.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 2809.45 | 2796.04 | 2812.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 2806.70 | 2798.18 | 2811.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 2818.35 | 2798.18 | 2811.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 2799.80 | 2798.50 | 2810.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 2791.60 | 2797.12 | 2808.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 2822.00 | 2798.85 | 2805.48 | SL hit (close>static) qty=1.00 sl=2814.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 2830.80 | 2801.30 | 2798.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 14:15:00 | 2846.00 | 2821.42 | 2811.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 2924.35 | 2956.68 | 2921.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 2924.35 | 2956.68 | 2921.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 2924.35 | 2956.68 | 2921.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 2924.35 | 2956.68 | 2921.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 2886.45 | 2942.63 | 2918.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 2886.45 | 2942.63 | 2918.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 2863.70 | 2926.84 | 2913.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 2863.70 | 2926.84 | 2913.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 2857.55 | 2904.40 | 2905.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 2840.00 | 2884.17 | 2895.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 2711.00 | 2708.04 | 2757.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 2711.00 | 2708.04 | 2757.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 2732.25 | 2719.05 | 2753.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:30:00 | 2715.25 | 2736.28 | 2751.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 2727.10 | 2734.45 | 2748.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:00:00 | 2718.45 | 2731.25 | 2746.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 2728.10 | 2702.80 | 2700.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 2728.10 | 2702.80 | 2700.75 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 2666.30 | 2697.08 | 2700.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 2654.55 | 2688.58 | 2696.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 14:15:00 | 2645.25 | 2640.68 | 2655.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-04 15:00:00 | 2645.25 | 2640.68 | 2655.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 2640.60 | 2640.79 | 2653.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:30:00 | 2644.85 | 2640.79 | 2653.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 2653.25 | 2643.29 | 2653.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:45:00 | 2656.85 | 2643.29 | 2653.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 2631.30 | 2640.89 | 2651.10 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 2693.00 | 2659.22 | 2657.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 2752.10 | 2684.32 | 2669.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 12:15:00 | 2809.00 | 2810.95 | 2763.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 13:00:00 | 2809.00 | 2810.95 | 2763.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2786.55 | 2806.31 | 2776.71 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 2692.50 | 2755.44 | 2759.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 2680.90 | 2740.53 | 2752.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2578.25 | 2536.81 | 2575.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 2578.25 | 2536.81 | 2575.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 2578.25 | 2536.81 | 2575.66 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 14:15:00 | 2641.80 | 2599.29 | 2595.03 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 10:15:00 | 2547.45 | 2584.65 | 2589.49 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 2640.40 | 2590.91 | 2590.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 2662.55 | 2605.24 | 2596.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 2649.95 | 2651.84 | 2629.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 2649.95 | 2651.84 | 2629.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 2649.95 | 2651.84 | 2629.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 13:00:00 | 2683.00 | 2649.23 | 2638.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 15:15:00 | 2676.35 | 2658.71 | 2645.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 12:15:00 | 2627.85 | 2723.41 | 2728.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 2627.85 | 2723.41 | 2728.51 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 2710.45 | 2704.17 | 2703.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 2716.10 | 2707.92 | 2705.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 11:15:00 | 2707.55 | 2708.56 | 2706.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 11:45:00 | 2708.55 | 2708.56 | 2706.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 2715.10 | 2709.87 | 2707.03 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 2663.65 | 2698.06 | 2702.34 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 2713.95 | 2697.27 | 2696.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 2728.95 | 2710.48 | 2703.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 2691.00 | 2724.12 | 2718.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 2691.00 | 2724.12 | 2718.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 2691.00 | 2724.12 | 2718.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 2691.00 | 2724.12 | 2718.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 2668.65 | 2713.03 | 2714.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 2650.30 | 2684.48 | 2697.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 2676.90 | 2665.83 | 2681.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 2676.90 | 2665.83 | 2681.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 2676.90 | 2665.83 | 2681.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 2676.90 | 2665.83 | 2681.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 2675.20 | 2667.71 | 2680.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 2669.75 | 2667.71 | 2680.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 2683.00 | 2670.77 | 2681.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 2687.70 | 2670.77 | 2681.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 2678.95 | 2672.40 | 2680.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:45:00 | 2683.95 | 2672.40 | 2680.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 2688.70 | 2676.35 | 2681.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 2688.70 | 2676.35 | 2681.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 2702.70 | 2681.62 | 2683.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 2702.70 | 2681.62 | 2683.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 2685.00 | 2684.30 | 2684.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 2700.00 | 2687.44 | 2685.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 2685.75 | 2687.10 | 2685.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 2685.75 | 2687.10 | 2685.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 2685.75 | 2687.10 | 2685.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 2695.00 | 2687.10 | 2685.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 2670.75 | 2683.83 | 2684.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 2666.20 | 2680.31 | 2682.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 2638.70 | 2637.54 | 2651.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:45:00 | 2636.60 | 2637.54 | 2651.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2622.60 | 2634.88 | 2645.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:30:00 | 2621.25 | 2629.29 | 2639.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:45:00 | 2618.40 | 2612.85 | 2627.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:30:00 | 2615.80 | 2610.03 | 2617.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 15:15:00 | 2490.19 | 2513.81 | 2533.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 09:15:00 | 2487.48 | 2511.64 | 2530.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 09:15:00 | 2485.01 | 2511.64 | 2530.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-01 10:15:00 | 2512.00 | 2511.71 | 2529.01 | SL hit (close>ema200) qty=0.50 sl=2511.71 alert=retest2 |

### Cycle 109 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 2466.55 | 2452.83 | 2451.68 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 2424.45 | 2452.11 | 2452.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 2376.85 | 2421.32 | 2435.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 2364.45 | 2360.19 | 2388.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 2364.45 | 2360.19 | 2388.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 2364.65 | 2348.79 | 2367.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:00:00 | 2364.65 | 2348.79 | 2367.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 2366.00 | 2352.23 | 2367.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:45:00 | 2361.10 | 2352.23 | 2367.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 2360.40 | 2353.86 | 2366.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:30:00 | 2366.00 | 2353.86 | 2366.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 2358.00 | 2357.20 | 2364.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:45:00 | 2382.00 | 2357.20 | 2364.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 2345.85 | 2354.93 | 2362.69 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 2383.20 | 2364.20 | 2363.77 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 2358.30 | 2363.13 | 2363.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 2330.45 | 2356.57 | 2360.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 2351.75 | 2350.61 | 2356.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 13:00:00 | 2351.75 | 2350.61 | 2356.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 2355.90 | 2351.67 | 2356.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 2355.90 | 2351.67 | 2356.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 2360.75 | 2353.49 | 2356.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:45:00 | 2366.70 | 2353.49 | 2356.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 2360.00 | 2354.79 | 2357.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 2403.00 | 2354.79 | 2357.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 2367.55 | 2357.34 | 2358.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:30:00 | 2393.20 | 2357.34 | 2358.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 11:15:00 | 2367.50 | 2359.85 | 2359.06 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 2350.00 | 2358.57 | 2358.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 2346.50 | 2356.15 | 2357.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 2319.25 | 2313.35 | 2331.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 2319.25 | 2313.35 | 2331.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2363.90 | 2323.64 | 2332.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 2363.90 | 2323.64 | 2332.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2367.30 | 2332.38 | 2335.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 2367.30 | 2332.38 | 2335.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 2366.30 | 2343.47 | 2340.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 2370.05 | 2351.50 | 2344.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 2317.05 | 2346.77 | 2343.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 2317.05 | 2346.77 | 2343.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 2317.05 | 2346.77 | 2343.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 2317.05 | 2346.77 | 2343.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 2331.30 | 2343.68 | 2342.82 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 2329.75 | 2340.89 | 2341.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 2301.85 | 2330.63 | 2336.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 2238.95 | 2215.09 | 2241.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 2238.95 | 2215.09 | 2241.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 2238.95 | 2215.09 | 2241.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 2238.95 | 2215.09 | 2241.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 2259.45 | 2223.96 | 2243.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 2259.45 | 2223.96 | 2243.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 2262.90 | 2231.75 | 2244.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 2274.15 | 2231.75 | 2244.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 2280.40 | 2253.87 | 2252.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 2300.40 | 2266.54 | 2258.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 2241.60 | 2265.18 | 2261.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 2241.60 | 2265.18 | 2261.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 2241.60 | 2265.18 | 2261.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 2241.60 | 2265.18 | 2261.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 2271.20 | 2266.38 | 2262.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:30:00 | 2318.40 | 2274.81 | 2266.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:15:00 | 2313.75 | 2281.18 | 2270.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:45:00 | 2312.20 | 2319.06 | 2302.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:45:00 | 2311.10 | 2320.66 | 2304.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 2362.20 | 2330.01 | 2313.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 2378.70 | 2336.01 | 2317.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:45:00 | 2366.20 | 2354.65 | 2344.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 11:00:00 | 2378.95 | 2359.51 | 2347.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 12:15:00 | 2375.00 | 2360.15 | 2348.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 2335.55 | 2363.04 | 2355.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 2335.55 | 2363.04 | 2355.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2349.60 | 2360.35 | 2355.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 2334.00 | 2360.35 | 2355.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 2347.85 | 2357.85 | 2354.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:00:00 | 2347.85 | 2357.85 | 2354.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 2349.20 | 2356.12 | 2353.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 2337.10 | 2352.32 | 2352.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 2337.10 | 2352.32 | 2352.43 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 2355.60 | 2352.97 | 2352.72 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 2309.15 | 2344.05 | 2348.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 2303.45 | 2335.93 | 2344.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 2218.90 | 2213.93 | 2250.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 2219.30 | 2213.93 | 2250.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2249.30 | 2217.05 | 2237.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 2249.30 | 2217.05 | 2237.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 2245.45 | 2222.73 | 2238.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 2245.45 | 2222.73 | 2238.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 2236.70 | 2225.52 | 2238.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 2228.15 | 2226.05 | 2237.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 2024.00 | 2229.15 | 2235.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-14 09:15:00 | 2005.34 | 2170.93 | 2208.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 1973.20 | 1944.79 | 1941.14 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 1930.85 | 1943.56 | 1943.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 1920.85 | 1935.81 | 1939.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 1935.00 | 1934.10 | 1938.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 1935.00 | 1934.10 | 1938.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 1935.00 | 1934.10 | 1938.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 1937.45 | 1934.10 | 1938.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1958.55 | 1938.99 | 1940.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 1958.55 | 1938.99 | 1940.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 13:15:00 | 1951.50 | 1941.49 | 1941.12 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 14:15:00 | 1924.00 | 1942.51 | 1943.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 15:15:00 | 1918.45 | 1937.69 | 1941.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 1847.95 | 1834.18 | 1860.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 1847.95 | 1834.18 | 1860.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1859.90 | 1845.12 | 1857.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1859.90 | 1845.12 | 1857.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1872.45 | 1850.58 | 1859.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 1872.45 | 1850.58 | 1859.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 1849.45 | 1851.91 | 1858.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:45:00 | 1861.60 | 1851.91 | 1858.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 1861.45 | 1853.82 | 1858.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 1861.45 | 1853.82 | 1858.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 1858.85 | 1854.83 | 1858.70 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1903.00 | 1865.96 | 1863.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 1924.00 | 1877.57 | 1868.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 1971.40 | 1975.62 | 1958.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 1971.40 | 1975.62 | 1958.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1947.35 | 1967.52 | 1958.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 1947.35 | 1967.52 | 1958.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1940.00 | 1962.02 | 1956.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1958.55 | 1962.02 | 1956.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1964.95 | 1965.62 | 1960.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:45:00 | 1960.80 | 1965.62 | 1960.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1951.05 | 1962.71 | 1959.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 1951.05 | 1962.71 | 1959.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 1953.10 | 1960.79 | 1958.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:30:00 | 1955.00 | 1960.79 | 1958.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 1960.90 | 1960.81 | 1958.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 1964.00 | 1960.28 | 1958.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 09:15:00 | 1944.10 | 1957.04 | 1957.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 1944.10 | 1957.04 | 1957.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 1927.00 | 1951.03 | 1954.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 1959.00 | 1943.24 | 1948.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 1959.00 | 1943.24 | 1948.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1959.00 | 1943.24 | 1948.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1959.00 | 1943.24 | 1948.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1958.00 | 1946.19 | 1949.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 1952.90 | 1946.19 | 1949.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1960.20 | 1950.31 | 1950.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:45:00 | 1954.85 | 1950.31 | 1950.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 11:15:00 | 1964.30 | 1953.11 | 1952.11 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 1947.00 | 1951.25 | 1951.60 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1985.70 | 1958.14 | 1954.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 2029.40 | 1996.97 | 1983.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 2031.00 | 2037.89 | 2021.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 2031.00 | 2037.89 | 2021.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2032.55 | 2035.36 | 2023.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:00:00 | 2049.70 | 2037.04 | 2027.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 11:45:00 | 2049.95 | 2067.95 | 2059.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:00:00 | 2047.45 | 2058.87 | 2056.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 2035.00 | 2054.09 | 2054.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 2035.00 | 2054.09 | 2054.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 2025.00 | 2048.27 | 2051.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 2002.00 | 1998.44 | 2013.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 09:15:00 | 2021.95 | 1998.44 | 2013.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2005.10 | 1999.78 | 2013.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 1983.10 | 2000.00 | 2011.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 1980.05 | 1979.01 | 1987.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 2020.10 | 1992.15 | 1991.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 2020.10 | 1992.15 | 1991.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 12:15:00 | 2025.90 | 1998.90 | 1994.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1982.40 | 2008.74 | 2002.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1982.40 | 2008.74 | 2002.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1982.40 | 2008.74 | 2002.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:45:00 | 1973.40 | 2008.74 | 2002.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1992.75 | 2005.54 | 2001.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:30:00 | 1995.10 | 2005.23 | 2001.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 1949.00 | 1992.43 | 1996.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1949.00 | 1992.43 | 1996.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 1940.00 | 1970.96 | 1984.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 1881.85 | 1824.36 | 1839.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 1881.85 | 1824.36 | 1839.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1881.85 | 1824.36 | 1839.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:45:00 | 1884.95 | 1824.36 | 1839.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 1906.70 | 1849.49 | 1848.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 1933.60 | 1880.89 | 1864.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 1961.20 | 1963.71 | 1948.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 1961.20 | 1963.71 | 1948.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1965.00 | 1962.74 | 1950.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:30:00 | 1976.60 | 1965.41 | 1952.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1953.90 | 2002.13 | 2007.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1953.90 | 2002.13 | 2007.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1932.40 | 1988.19 | 2000.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1974.70 | 1967.95 | 1982.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 1974.70 | 1967.95 | 1982.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 1975.30 | 1970.04 | 1979.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 1977.40 | 1970.04 | 1979.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 1975.00 | 1972.56 | 1978.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 1999.80 | 1972.56 | 1978.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1985.60 | 1975.17 | 1979.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:15:00 | 1977.10 | 1981.21 | 1981.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 14:15:00 | 1989.50 | 1982.87 | 1982.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 1989.50 | 1982.87 | 1982.21 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 1959.60 | 1977.93 | 1980.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 1945.10 | 1968.57 | 1975.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1992.30 | 1963.98 | 1970.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1992.30 | 1963.98 | 1970.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1992.30 | 1963.98 | 1970.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1992.30 | 1963.98 | 1970.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 2001.20 | 1971.42 | 1973.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:00:00 | 2001.20 | 1971.42 | 1973.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 1978.90 | 1973.55 | 1974.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:00:00 | 1978.90 | 1973.55 | 1974.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 1986.30 | 1976.10 | 1975.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 14:15:00 | 1993.20 | 1979.52 | 1976.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 1990.20 | 2006.45 | 1997.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 1990.20 | 2006.45 | 1997.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1990.20 | 2006.45 | 1997.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 1990.20 | 2006.45 | 1997.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1988.50 | 2002.86 | 1996.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:15:00 | 1975.00 | 2002.86 | 1996.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 1970.10 | 1991.87 | 1992.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 1961.60 | 1985.81 | 1989.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 1984.00 | 1969.71 | 1978.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 1984.00 | 1969.71 | 1978.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1984.00 | 1969.71 | 1978.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 1984.00 | 1969.71 | 1978.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1978.70 | 1971.50 | 1978.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 1982.00 | 1971.50 | 1978.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 1986.10 | 1974.42 | 1978.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 1986.10 | 1974.42 | 1978.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1989.40 | 1977.42 | 1979.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:30:00 | 1981.90 | 1977.42 | 1979.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1985.50 | 1979.04 | 1980.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1979.30 | 1979.04 | 1980.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 1996.10 | 1982.45 | 1981.85 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 11:15:00 | 1968.30 | 1980.35 | 1981.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 12:15:00 | 1945.40 | 1973.36 | 1977.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1952.60 | 1920.23 | 1934.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1952.60 | 1920.23 | 1934.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1952.60 | 1920.23 | 1934.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 1982.70 | 1920.23 | 1934.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-05-12 12:15:00)

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

### Cycle 142 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 2060.50 | 2088.82 | 2090.53 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 2129.70 | 2096.85 | 2093.56 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-05-22 14:15:00)

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

### Cycle 145 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 2121.00 | 2063.14 | 2058.14 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-30 11:15:00)

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

### Cycle 147 — BUY (started 2025-06-24 12:15:00)

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

### Cycle 148 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 1960.80 | 1966.53 | 1967.17 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-07-02 09:15:00)

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

### Cycle 150 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 1974.90 | 1979.23 | 1979.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 1964.10 | 1976.20 | 1978.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 11:15:00 | 1965.90 | 1963.55 | 1968.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 1965.90 | 1963.55 | 1968.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1965.90 | 1963.55 | 1968.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 1967.70 | 1963.55 | 1968.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1965.00 | 1963.84 | 1968.54 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1979.10 | 1972.63 | 1971.77 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 1969.10 | 1981.16 | 1981.84 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-07-15 11:15:00)

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

### Cycle 154 — SELL (started 2025-07-16 09:15:00)

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

### Cycle 155 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 1825.10 | 1815.19 | 1814.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 1835.10 | 1821.32 | 1817.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 1852.00 | 1853.79 | 1841.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:45:00 | 1850.50 | 1853.79 | 1841.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1819.20 | 1857.13 | 1851.05 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-08-14 11:15:00)

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

### Cycle 157 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 1831.30 | 1816.24 | 1815.55 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 1812.50 | 1819.10 | 1819.29 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 1846.00 | 1824.48 | 1821.72 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-08-28 11:15:00)

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

### Cycle 161 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1768.40 | 1749.53 | 1749.35 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1743.10 | 1748.15 | 1748.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 13:15:00 | 1737.70 | 1746.06 | 1747.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 1748.20 | 1744.61 | 1746.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1748.20 | 1744.61 | 1746.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1748.20 | 1744.61 | 1746.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 1743.80 | 1744.88 | 1746.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1765.50 | 1750.52 | 1748.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1765.50 | 1750.52 | 1748.50 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 1746.70 | 1749.28 | 1749.45 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1759.90 | 1750.08 | 1749.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 1782.00 | 1758.79 | 1753.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 10:15:00 | 1824.10 | 1828.56 | 1806.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:45:00 | 1824.80 | 1828.56 | 1806.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1816.00 | 1821.57 | 1808.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1833.60 | 1819.51 | 1809.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1804.50 | 1823.57 | 1817.37 | SL hit (close<static) qty=1.00 sl=1806.50 alert=retest2 |

### Cycle 166 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 1812.80 | 1816.94 | 1816.97 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 1821.10 | 1817.77 | 1817.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 1828.80 | 1820.82 | 1818.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 15:15:00 | 1820.80 | 1820.95 | 1819.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 09:15:00 | 1822.40 | 1820.95 | 1819.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1826.70 | 1822.10 | 1820.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 1836.00 | 1822.10 | 1820.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 10:15:00 | 1832.50 | 1844.58 | 1845.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-09-29 10:15:00)

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

### Cycle 169 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1846.50 | 1838.00 | 1837.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 1856.90 | 1844.52 | 1840.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 1829.40 | 1844.79 | 1843.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 11:15:00 | 1829.40 | 1844.79 | 1843.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1829.40 | 1844.79 | 1843.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 1827.50 | 1844.79 | 1843.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-10-06 12:15:00)

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

### Cycle 171 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1779.00 | 1771.30 | 1770.90 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 1768.50 | 1770.53 | 1770.61 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 1772.00 | 1770.83 | 1770.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 1776.20 | 1772.50 | 1771.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 1766.00 | 1771.28 | 1771.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 1766.00 | 1771.28 | 1771.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1766.00 | 1771.28 | 1771.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 1766.00 | 1771.28 | 1771.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 1765.80 | 1770.19 | 1770.68 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1775.00 | 1771.02 | 1770.78 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-10-20 10:15:00)

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

### Cycle 177 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 1772.80 | 1768.22 | 1768.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 12:15:00 | 1775.10 | 1769.60 | 1768.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1767.10 | 1769.10 | 1768.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 13:15:00 | 1767.10 | 1769.10 | 1768.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1767.10 | 1769.10 | 1768.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 1767.10 | 1769.10 | 1768.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1764.50 | 1768.18 | 1768.29 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 1772.10 | 1768.50 | 1768.39 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-10-24 10:15:00)

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

### Cycle 181 — BUY (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 14:15:00 | 1768.80 | 1741.09 | 1739.47 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-11-06 11:15:00)

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

### Cycle 183 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1729.00 | 1726.86 | 1726.71 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 09:15:00 | 1713.80 | 1725.23 | 1726.28 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-11-14 10:15:00)

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

### Cycle 186 — SELL (started 2025-11-18 09:15:00)

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

### Cycle 187 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1727.50 | 1718.70 | 1718.44 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-11-20 14:15:00)

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

### Cycle 189 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 1561.50 | 1537.95 | 1536.47 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 1541.50 | 1544.55 | 1544.60 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1548.00 | 1545.24 | 1544.91 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1533.10 | 1542.81 | 1543.84 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-12-15 10:15:00)

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

### Cycle 194 — SELL (started 2025-12-30 14:15:00)

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

### Cycle 195 — BUY (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 10:15:00 | 1596.40 | 1557.80 | 1557.69 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-01-23 13:15:00)

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

### Cycle 197 — BUY (started 2026-01-29 13:15:00)

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

### Cycle 198 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 1573.90 | 1600.94 | 1603.03 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-02-02 14:15:00)

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

### Cycle 200 — SELL (started 2026-02-05 10:15:00)

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

### Cycle 201 — BUY (started 2026-02-09 09:15:00)

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

### Cycle 202 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 1655.90 | 1668.49 | 1669.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1625.20 | 1656.28 | 1662.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 1658.40 | 1655.25 | 1661.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 1658.40 | 1655.25 | 1661.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1645.00 | 1652.81 | 1658.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 1634.00 | 1652.81 | 1658.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 1654.90 | 1642.85 | 1642.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1654.90 | 1642.85 | 1642.84 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 1632.30 | 1641.03 | 1642.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 1630.00 | 1638.82 | 1641.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 1642.30 | 1638.12 | 1640.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 1642.30 | 1638.12 | 1640.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1642.30 | 1638.12 | 1640.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1642.30 | 1638.12 | 1640.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 1655.00 | 1641.50 | 1641.44 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-02-20 09:15:00)

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

### Cycle 207 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1372.40 | 1354.65 | 1353.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1378.10 | 1359.34 | 1355.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 1361.00 | 1361.25 | 1357.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 15:15:00 | 1361.00 | 1361.25 | 1357.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1361.00 | 1361.25 | 1357.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1370.00 | 1361.25 | 1357.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 1341.90 | 1357.38 | 1356.00 | SL hit (close<static) qty=1.00 sl=1355.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-03-27 10:15:00)

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

### Cycle 209 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1351.80 | 1332.99 | 1331.43 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1313.60 | 1330.25 | 1331.23 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-04-02 11:15:00)

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

### Cycle 212 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1678.90 | 1688.61 | 1688.81 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-04-29 10:15:00)

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
| BUY | retest2 | 2023-06-05 09:15:00 | 2106.30 | 2023-06-08 11:15:00 | 2080.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest1 | 2023-06-20 09:15:00 | 2199.70 | 2023-06-23 09:15:00 | 2231.95 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2023-06-28 10:30:00 | 2183.90 | 2023-07-07 12:15:00 | 2074.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-28 12:00:00 | 2192.80 | 2023-07-07 12:15:00 | 2083.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-28 14:15:00 | 2193.00 | 2023-07-07 12:15:00 | 2083.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-30 09:45:00 | 2186.50 | 2023-07-07 12:15:00 | 2077.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-05 13:45:00 | 2159.25 | 2023-07-10 09:15:00 | 2051.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-28 10:30:00 | 2183.90 | 2023-07-11 09:15:00 | 1965.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-06-28 12:00:00 | 2192.80 | 2023-07-11 09:15:00 | 1973.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-06-28 14:15:00 | 2193.00 | 2023-07-11 09:15:00 | 1973.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-06-30 09:45:00 | 2186.50 | 2023-07-11 09:15:00 | 1967.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-07-05 13:45:00 | 2159.25 | 2023-07-11 09:15:00 | 1943.33 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-07-18 13:30:00 | 1974.50 | 2023-07-21 11:15:00 | 1966.65 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-07-18 14:00:00 | 1976.00 | 2023-07-21 11:15:00 | 1966.65 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2023-07-21 10:45:00 | 1975.90 | 2023-07-21 11:15:00 | 1966.65 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-07-24 10:45:00 | 1962.50 | 2023-07-24 12:15:00 | 1988.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2023-08-08 09:15:00 | 2055.35 | 2023-08-10 09:15:00 | 2050.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2023-08-08 12:15:00 | 2065.35 | 2023-08-14 11:15:00 | 2057.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-08-08 14:30:00 | 2070.00 | 2023-08-14 11:15:00 | 2057.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-08-09 11:45:00 | 2065.00 | 2023-08-14 11:15:00 | 2057.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-08-09 13:00:00 | 2070.20 | 2023-08-14 12:15:00 | 2048.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-08-11 09:15:00 | 2075.20 | 2023-08-14 12:15:00 | 2048.70 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2023-08-14 10:15:00 | 2070.35 | 2023-08-14 12:15:00 | 2048.70 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-08-14 11:00:00 | 2071.00 | 2023-08-14 12:15:00 | 2048.70 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-08-22 09:45:00 | 1995.50 | 2023-08-24 09:15:00 | 2024.60 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-08-28 11:00:00 | 2031.20 | 2023-09-04 09:15:00 | 2234.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-14 10:30:00 | 2278.40 | 2023-09-22 09:15:00 | 2164.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-14 10:30:00 | 2278.40 | 2023-09-25 09:15:00 | 2152.05 | STOP_HIT | 0.50 | 5.55% |
| SELL | retest2 | 2023-10-27 12:45:00 | 1986.05 | 2023-10-31 09:15:00 | 2000.55 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-10-27 15:00:00 | 1986.00 | 2023-10-31 09:15:00 | 2000.55 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-10-30 09:15:00 | 1982.20 | 2023-10-31 09:15:00 | 2000.55 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-10-30 10:45:00 | 1989.00 | 2023-10-31 09:15:00 | 2000.55 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2023-11-13 12:30:00 | 2109.60 | 2023-11-20 13:15:00 | 2118.95 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2023-11-15 09:15:00 | 2117.70 | 2023-11-20 13:15:00 | 2118.95 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2023-12-01 09:15:00 | 2217.40 | 2023-12-08 11:15:00 | 2219.90 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2023-12-01 11:15:00 | 2209.80 | 2023-12-08 11:15:00 | 2219.90 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2023-12-12 14:30:00 | 2205.00 | 2023-12-13 13:15:00 | 2239.95 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2023-12-13 09:45:00 | 2209.30 | 2023-12-13 13:15:00 | 2239.95 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-12-13 10:30:00 | 2207.85 | 2023-12-13 13:15:00 | 2239.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-12-18 10:30:00 | 2337.00 | 2023-12-19 09:15:00 | 2269.95 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2023-12-18 11:30:00 | 2331.00 | 2023-12-19 09:15:00 | 2269.95 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2023-12-29 10:00:00 | 2454.25 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-01-02 10:00:00 | 2455.80 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-01-02 10:45:00 | 2448.95 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2024-01-02 11:30:00 | 2447.25 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2024-01-03 11:15:00 | 2475.95 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-01-03 12:45:00 | 2474.60 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-01-03 13:15:00 | 2474.70 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-01-03 14:45:00 | 2476.00 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-01-04 09:15:00 | 2483.20 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-01-04 10:00:00 | 2506.10 | 2024-01-05 12:15:00 | 2465.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest1 | 2024-01-09 11:45:00 | 2433.10 | 2024-01-11 09:15:00 | 2449.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest1 | 2024-01-09 13:45:00 | 2423.15 | 2024-01-11 09:15:00 | 2449.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-01-10 13:30:00 | 2400.00 | 2024-01-11 11:15:00 | 2471.35 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-01-16 10:45:00 | 2423.55 | 2024-01-18 09:15:00 | 2302.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-16 11:45:00 | 2420.35 | 2024-01-18 09:15:00 | 2299.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-16 10:45:00 | 2423.55 | 2024-01-19 09:15:00 | 2383.00 | STOP_HIT | 0.50 | 1.67% |
| SELL | retest2 | 2024-01-16 11:45:00 | 2420.35 | 2024-01-19 09:15:00 | 2383.00 | STOP_HIT | 0.50 | 1.54% |
| SELL | retest2 | 2024-02-07 11:15:00 | 2231.00 | 2024-02-13 13:15:00 | 2228.60 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2024-02-07 14:45:00 | 2232.20 | 2024-02-13 13:15:00 | 2228.60 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-02-08 10:30:00 | 2225.45 | 2024-02-13 13:15:00 | 2228.60 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-02-12 09:45:00 | 2228.00 | 2024-02-13 13:15:00 | 2228.60 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-02-20 09:15:00 | 2318.75 | 2024-02-20 09:15:00 | 2282.80 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-02-20 12:45:00 | 2300.65 | 2024-02-23 14:15:00 | 2308.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-03-04 11:15:00 | 2206.00 | 2024-03-07 09:15:00 | 2237.20 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-03-04 11:45:00 | 2205.40 | 2024-03-07 09:15:00 | 2237.20 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-03-05 11:00:00 | 2202.15 | 2024-03-07 09:15:00 | 2237.20 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-03-05 12:00:00 | 2201.40 | 2024-03-07 09:15:00 | 2237.20 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-03-27 09:15:00 | 2145.65 | 2024-03-27 09:15:00 | 2132.15 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-03-27 10:30:00 | 2150.80 | 2024-03-28 11:15:00 | 2131.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-04-04 12:30:00 | 2221.35 | 2024-04-09 13:15:00 | 2190.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-04-04 15:15:00 | 2217.00 | 2024-04-09 14:15:00 | 2193.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-04-05 15:00:00 | 2219.05 | 2024-04-09 14:15:00 | 2193.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-04-08 10:15:00 | 2222.05 | 2024-04-09 14:15:00 | 2193.50 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-04-09 09:15:00 | 2216.55 | 2024-04-09 14:15:00 | 2193.50 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-04-16 10:45:00 | 2315.00 | 2024-04-19 10:15:00 | 2248.75 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-04-30 11:15:00 | 2461.55 | 2024-05-03 12:15:00 | 2430.20 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-05-02 11:00:00 | 2458.65 | 2024-05-03 12:15:00 | 2430.20 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-05-02 12:00:00 | 2459.55 | 2024-05-03 12:15:00 | 2430.20 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-05-14 09:15:00 | 2471.55 | 2024-05-14 10:15:00 | 2485.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-06-04 09:15:00 | 2178.20 | 2024-06-04 12:15:00 | 2069.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:45:00 | 2191.15 | 2024-06-04 12:15:00 | 2081.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 2178.20 | 2024-06-05 09:15:00 | 2197.80 | STOP_HIT | 0.50 | -0.90% |
| SELL | retest2 | 2024-06-04 10:45:00 | 2191.15 | 2024-06-05 09:15:00 | 2197.80 | STOP_HIT | 0.50 | -0.30% |
| BUY | retest2 | 2024-06-18 12:00:00 | 2417.05 | 2024-06-24 15:15:00 | 2505.95 | STOP_HIT | 1.00 | 3.68% |
| BUY | retest2 | 2024-06-18 14:15:00 | 2415.90 | 2024-06-24 15:15:00 | 2505.95 | STOP_HIT | 1.00 | 3.73% |
| BUY | retest2 | 2024-07-04 11:45:00 | 2685.15 | 2024-07-10 09:15:00 | 2661.55 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-07-04 12:45:00 | 2687.75 | 2024-07-10 09:15:00 | 2661.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-04 13:45:00 | 2680.45 | 2024-07-10 10:15:00 | 2678.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-07-05 09:30:00 | 2718.90 | 2024-07-10 10:15:00 | 2678.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-07-09 09:15:00 | 2712.65 | 2024-07-10 10:15:00 | 2678.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-07-09 13:30:00 | 2729.35 | 2024-07-10 10:15:00 | 2678.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-07-19 10:15:00 | 2843.65 | 2024-07-19 15:15:00 | 2764.90 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-07-24 10:00:00 | 2873.00 | 2024-08-01 09:15:00 | 3160.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 13:30:00 | 3041.05 | 2024-08-07 12:15:00 | 3076.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-08-06 14:00:00 | 3039.75 | 2024-08-07 12:15:00 | 3076.30 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-08-06 14:30:00 | 3035.40 | 2024-08-07 12:15:00 | 3076.30 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-08-07 11:30:00 | 3048.20 | 2024-08-07 12:15:00 | 3076.30 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-08-19 10:15:00 | 2875.25 | 2024-08-21 10:15:00 | 2900.85 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-08-19 13:30:00 | 2876.55 | 2024-08-21 10:15:00 | 2900.85 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-08-20 09:30:00 | 2872.30 | 2024-08-21 10:15:00 | 2900.85 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-09-04 09:15:00 | 2956.90 | 2024-09-06 14:15:00 | 2928.85 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest1 | 2024-09-09 12:15:00 | 2935.00 | 2024-09-10 09:15:00 | 2967.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-09-10 12:00:00 | 2951.00 | 2024-09-10 14:15:00 | 2975.60 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-09-13 15:15:00 | 2925.15 | 2024-09-23 15:15:00 | 2869.00 | STOP_HIT | 1.00 | 1.92% |
| SELL | retest2 | 2024-09-16 14:30:00 | 2926.35 | 2024-09-23 15:15:00 | 2869.00 | STOP_HIT | 1.00 | 1.96% |
| SELL | retest2 | 2024-09-17 09:15:00 | 2927.20 | 2024-09-23 15:15:00 | 2869.00 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2024-10-09 13:00:00 | 2791.60 | 2024-10-10 09:15:00 | 2822.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-10 11:00:00 | 2792.00 | 2024-10-11 14:15:00 | 2830.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-10-10 11:30:00 | 2789.95 | 2024-10-11 14:15:00 | 2830.80 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-10-10 12:15:00 | 2788.00 | 2024-10-11 14:15:00 | 2830.80 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-10-24 09:30:00 | 2715.25 | 2024-10-30 10:15:00 | 2728.10 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-10-24 11:00:00 | 2727.10 | 2024-10-30 10:15:00 | 2728.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-10-24 12:00:00 | 2718.45 | 2024-10-30 10:15:00 | 2728.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-11-22 13:00:00 | 2683.00 | 2024-12-03 12:15:00 | 2627.85 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-11-22 15:15:00 | 2676.35 | 2024-12-03 12:15:00 | 2627.85 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-12-20 14:30:00 | 2621.25 | 2024-12-31 15:15:00 | 2490.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-23 11:45:00 | 2618.40 | 2025-01-01 09:15:00 | 2487.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-24 11:30:00 | 2615.80 | 2025-01-01 09:15:00 | 2485.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 14:30:00 | 2621.25 | 2025-01-01 10:15:00 | 2512.00 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2024-12-23 11:45:00 | 2618.40 | 2025-01-01 10:15:00 | 2512.00 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2024-12-24 11:30:00 | 2615.80 | 2025-01-01 10:15:00 | 2512.00 | STOP_HIT | 0.50 | 3.97% |
| BUY | retest2 | 2025-01-31 09:30:00 | 2318.40 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-01-31 11:15:00 | 2313.75 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2025-02-01 12:45:00 | 2312.20 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2025-02-01 13:45:00 | 2311.10 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2025-02-03 10:45:00 | 2378.70 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-02-06 09:45:00 | 2366.20 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-02-06 11:00:00 | 2378.95 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-02-06 12:15:00 | 2375.00 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-02-13 13:00:00 | 2228.15 | 2025-02-14 09:15:00 | 2005.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 2024.00 | 2025-02-14 10:15:00 | 1922.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 2024.00 | 2025-02-17 09:15:00 | 1821.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-12 09:15:00 | 1964.00 | 2025-03-12 09:15:00 | 1944.10 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-03-21 14:00:00 | 2049.70 | 2025-03-25 14:15:00 | 2035.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-03-25 11:45:00 | 2049.95 | 2025-03-25 14:15:00 | 2035.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-03-25 14:00:00 | 2047.45 | 2025-03-25 14:15:00 | 2035.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-03-28 12:15:00 | 1983.10 | 2025-04-02 11:15:00 | 2020.10 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-04-02 09:15:00 | 1980.05 | 2025-04-02 11:15:00 | 2020.10 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-04-03 11:30:00 | 1995.10 | 2025-04-04 09:15:00 | 1949.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-04-21 10:30:00 | 1976.60 | 2025-04-25 09:15:00 | 1953.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-04-29 14:15:00 | 1977.10 | 2025-04-29 14:15:00 | 1989.50 | STOP_HIT | 1.00 | -0.63% |
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
