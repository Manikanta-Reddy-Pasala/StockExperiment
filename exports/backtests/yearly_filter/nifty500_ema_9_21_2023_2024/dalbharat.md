# Dalmia Bharat Ltd. (DALBHARAT)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 1840.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 227 |
| ALERT1 | 147 |
| ALERT2 | 145 |
| ALERT2_SKIP | 65 |
| ALERT3 | 400 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 165 |
| PARTIAL | 21 |
| TARGET_HIT | 5 |
| STOP_HIT | 167 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 192 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 71 / 121
- **Target hits / Stop hits / Partials:** 5 / 166 / 21
- **Avg / median % per leg:** 0.63% / -0.63%
- **Sum % (uncompounded):** 121.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 80 | 19 | 23.8% | 3 | 77 | 0 | -0.26% | -21.0% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.73% | -4.4% |
| BUY @ 3rd Alert (retest2) | 74 | 19 | 25.7% | 3 | 71 | 0 | -0.22% | -16.6% |
| SELL (all) | 112 | 52 | 46.4% | 2 | 89 | 21 | 1.28% | 142.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 112 | 52 | 46.4% | 2 | 89 | 21 | 1.28% | 142.8% |
| retest1 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.73% | -4.4% |
| retest2 (combined) | 186 | 71 | 38.2% | 5 | 160 | 21 | 0.68% | 126.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 15:15:00 | 2070.00 | 2080.55 | 2081.74 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 13:15:00 | 2088.50 | 2082.52 | 2082.06 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 2077.05 | 2081.31 | 2081.59 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 09:15:00 | 2094.65 | 2083.98 | 2082.77 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 2074.85 | 2081.90 | 2082.02 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 09:15:00 | 2094.35 | 2083.82 | 2082.61 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-05-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 13:15:00 | 2069.00 | 2079.96 | 2081.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 14:15:00 | 2062.65 | 2076.50 | 2079.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 15:15:00 | 2056.95 | 2056.70 | 2065.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-22 09:15:00 | 2064.20 | 2056.70 | 2065.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 2075.50 | 2060.46 | 2066.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:00:00 | 2075.50 | 2060.46 | 2066.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 2063.30 | 2061.03 | 2065.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 11:45:00 | 2061.05 | 2060.97 | 2065.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 14:15:00 | 2078.00 | 2066.97 | 2067.30 | SL hit (close>static) qty=1.00 sl=2075.50 alert=retest2 |

### Cycle 8 — BUY (started 2023-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 15:15:00 | 2077.25 | 2069.02 | 2068.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 2095.00 | 2074.22 | 2070.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 12:15:00 | 2075.40 | 2077.07 | 2073.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 12:15:00 | 2075.40 | 2077.07 | 2073.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 2075.40 | 2077.07 | 2073.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 12:30:00 | 2075.85 | 2077.07 | 2073.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 2071.70 | 2076.00 | 2073.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 14:00:00 | 2071.70 | 2076.00 | 2073.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 2076.40 | 2076.08 | 2073.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 14:45:00 | 2071.20 | 2076.08 | 2073.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 2071.05 | 2075.07 | 2073.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:15:00 | 2053.45 | 2075.07 | 2073.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 09:15:00 | 2054.85 | 2071.03 | 2071.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 15:15:00 | 2042.10 | 2057.11 | 2063.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 15:15:00 | 2048.90 | 2042.62 | 2050.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 15:15:00 | 2048.90 | 2042.62 | 2050.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 2048.90 | 2042.62 | 2050.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 2047.85 | 2042.62 | 2050.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 2057.50 | 2045.59 | 2051.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:00:00 | 2057.50 | 2045.59 | 2051.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 2065.00 | 2049.48 | 2052.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:45:00 | 2063.25 | 2049.48 | 2052.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 12:15:00 | 2067.20 | 2055.82 | 2055.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 2081.90 | 2065.28 | 2060.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 11:15:00 | 2113.65 | 2124.20 | 2109.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 11:15:00 | 2113.65 | 2124.20 | 2109.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 2113.65 | 2124.20 | 2109.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 11:45:00 | 2115.95 | 2124.20 | 2109.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 2132.80 | 2125.92 | 2111.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 12:30:00 | 2115.95 | 2125.92 | 2111.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 2130.00 | 2140.10 | 2130.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 2149.05 | 2140.10 | 2130.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 10:00:00 | 2141.90 | 2140.46 | 2131.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 09:15:00 | 2147.60 | 2135.37 | 2132.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 10:30:00 | 2142.40 | 2138.16 | 2134.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 14:15:00 | 2134.25 | 2140.11 | 2136.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 15:00:00 | 2134.25 | 2140.11 | 2136.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 2143.95 | 2140.88 | 2137.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:15:00 | 2145.00 | 2140.88 | 2137.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 2160.45 | 2144.79 | 2139.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 12:30:00 | 2165.70 | 2150.69 | 2143.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 15:15:00 | 2165.90 | 2153.39 | 2146.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 13:45:00 | 2165.95 | 2156.28 | 2151.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 14:45:00 | 2168.15 | 2157.81 | 2152.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 2150.95 | 2157.82 | 2153.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:00:00 | 2150.95 | 2157.82 | 2153.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 2125.20 | 2151.30 | 2151.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-08 11:15:00 | 2125.20 | 2151.30 | 2151.16 | SL hit (close<static) qty=1.00 sl=2126.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 2111.30 | 2143.30 | 2147.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 14:15:00 | 2107.00 | 2124.10 | 2133.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 15:15:00 | 2112.80 | 2104.01 | 2115.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-13 09:15:00 | 2119.50 | 2104.01 | 2115.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 2140.65 | 2111.33 | 2117.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:00:00 | 2140.65 | 2111.33 | 2117.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 2135.10 | 2116.09 | 2118.97 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 2155.10 | 2123.89 | 2122.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 15:15:00 | 2163.90 | 2143.47 | 2133.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 14:15:00 | 2253.20 | 2253.25 | 2227.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 15:00:00 | 2253.20 | 2253.25 | 2227.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 2239.80 | 2250.74 | 2238.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 13:30:00 | 2234.65 | 2250.74 | 2238.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 14:15:00 | 2243.00 | 2249.19 | 2238.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 15:00:00 | 2243.00 | 2249.19 | 2238.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 15:15:00 | 2245.10 | 2248.38 | 2239.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:15:00 | 2230.65 | 2248.38 | 2239.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 2236.85 | 2246.07 | 2238.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 09:15:00 | 2253.60 | 2243.62 | 2240.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 10:00:00 | 2259.00 | 2256.41 | 2249.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 12:15:00 | 2186.05 | 2236.26 | 2241.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 2186.05 | 2236.26 | 2241.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 10:15:00 | 2179.80 | 2205.95 | 2223.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 11:15:00 | 2178.60 | 2176.46 | 2195.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 12:00:00 | 2178.60 | 2176.46 | 2195.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 2191.20 | 2180.17 | 2193.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:00:00 | 2191.20 | 2180.17 | 2193.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 2198.50 | 2183.84 | 2194.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:00:00 | 2198.50 | 2183.84 | 2194.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 2197.00 | 2186.47 | 2194.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 2218.00 | 2186.47 | 2194.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 2207.55 | 2190.69 | 2195.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 2218.05 | 2190.69 | 2195.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 2194.40 | 2192.12 | 2195.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 12:00:00 | 2194.40 | 2192.12 | 2195.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 12:15:00 | 2175.00 | 2188.70 | 2193.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 13:30:00 | 2170.00 | 2184.67 | 2191.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 11:15:00 | 2200.00 | 2183.67 | 2187.45 | SL hit (close>static) qty=1.00 sl=2194.40 alert=retest2 |

### Cycle 14 — BUY (started 2023-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 13:15:00 | 2207.95 | 2191.13 | 2190.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 14:15:00 | 2214.60 | 2195.83 | 2192.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 10:15:00 | 2189.65 | 2202.13 | 2197.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 10:15:00 | 2189.65 | 2202.13 | 2197.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 2189.65 | 2202.13 | 2197.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 11:00:00 | 2189.65 | 2202.13 | 2197.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 11:15:00 | 2193.70 | 2200.44 | 2196.77 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 13:15:00 | 2175.00 | 2193.97 | 2194.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 14:15:00 | 2166.05 | 2188.39 | 2191.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 09:15:00 | 2189.00 | 2184.61 | 2189.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 2189.00 | 2184.61 | 2189.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 2189.00 | 2184.61 | 2189.25 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 13:15:00 | 2199.00 | 2192.18 | 2191.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 14:15:00 | 2201.35 | 2194.02 | 2192.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 2187.30 | 2193.47 | 2192.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 2187.30 | 2193.47 | 2192.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 2187.30 | 2193.47 | 2192.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:00:00 | 2187.30 | 2193.47 | 2192.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 2200.00 | 2194.78 | 2193.36 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 14:15:00 | 2181.50 | 2191.09 | 2192.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 09:15:00 | 2167.20 | 2184.86 | 2189.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 15:15:00 | 2184.90 | 2178.34 | 2183.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 15:15:00 | 2184.90 | 2178.34 | 2183.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 2184.90 | 2178.34 | 2183.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:15:00 | 2175.45 | 2178.34 | 2183.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 2173.05 | 2177.28 | 2182.11 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-07-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 15:15:00 | 2187.05 | 2184.11 | 2183.75 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 09:15:00 | 2170.15 | 2181.32 | 2182.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 12:15:00 | 2158.25 | 2172.37 | 2177.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-13 10:15:00 | 2066.00 | 2064.16 | 2080.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-13 10:45:00 | 2066.00 | 2064.16 | 2080.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 2063.95 | 2061.96 | 2072.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 13:30:00 | 2047.50 | 2058.97 | 2063.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 09:45:00 | 2048.10 | 2055.82 | 2061.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 10:30:00 | 2047.40 | 2053.96 | 2059.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 09:15:00 | 1945.12 | 2011.57 | 2027.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 09:15:00 | 1945.69 | 2011.57 | 2027.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 09:15:00 | 1945.03 | 2011.57 | 2027.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-07-25 14:15:00 | 1908.25 | 1906.48 | 1928.23 | SL hit (close>ema200) qty=0.50 sl=1906.48 alert=retest2 |

### Cycle 20 — BUY (started 2023-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 14:15:00 | 1958.10 | 1933.53 | 1932.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 11:15:00 | 1980.05 | 1958.68 | 1952.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 1969.40 | 1975.15 | 1965.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 1969.40 | 1975.15 | 1965.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 1969.40 | 1975.15 | 1965.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 1969.40 | 1975.15 | 1965.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 1971.10 | 1974.34 | 1965.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 1967.50 | 1974.34 | 1965.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 1945.00 | 1970.21 | 1967.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 10:00:00 | 1945.00 | 1970.21 | 1967.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 1948.25 | 1965.82 | 1965.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 11:15:00 | 1953.60 | 1965.82 | 1965.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-03 11:15:00 | 1957.05 | 1964.06 | 1964.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 11:15:00 | 1957.05 | 1964.06 | 1964.92 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 15:15:00 | 1971.20 | 1965.84 | 1965.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 10:15:00 | 1980.35 | 1970.45 | 1967.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 10:15:00 | 1979.15 | 1985.24 | 1978.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 10:15:00 | 1979.15 | 1985.24 | 1978.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 1979.15 | 1985.24 | 1978.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:00:00 | 1979.15 | 1985.24 | 1978.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 1993.70 | 1986.93 | 1979.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 12:15:00 | 1995.50 | 1986.93 | 1979.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 12:45:00 | 1997.00 | 1988.93 | 1981.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 11:30:00 | 1994.90 | 1998.71 | 1990.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 12:15:00 | 1992.00 | 2000.59 | 2000.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 12:15:00 | 1992.00 | 2000.59 | 2000.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 09:15:00 | 1987.10 | 1995.19 | 1997.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 11:15:00 | 1995.40 | 1993.60 | 1996.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 11:15:00 | 1995.40 | 1993.60 | 1996.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 11:15:00 | 1995.40 | 1993.60 | 1996.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 12:00:00 | 1995.40 | 1993.60 | 1996.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 12:15:00 | 1991.10 | 1993.10 | 1996.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 15:15:00 | 1980.00 | 1994.99 | 1996.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 09:15:00 | 1881.00 | 1926.25 | 1953.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-08-17 14:15:00 | 1907.60 | 1900.37 | 1915.24 | SL hit (close>ema200) qty=0.50 sl=1900.37 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 15:15:00 | 1918.00 | 1909.07 | 1907.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 1922.85 | 1911.82 | 1909.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 15:15:00 | 1998.80 | 1998.99 | 1978.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-25 09:15:00 | 1992.45 | 1998.99 | 1978.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 1988.85 | 1996.96 | 1979.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 11:00:00 | 2015.95 | 2003.80 | 1991.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-04 11:15:00 | 2217.55 | 2154.75 | 2119.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 11:15:00 | 2312.45 | 2333.37 | 2335.42 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 2357.70 | 2336.07 | 2335.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 2395.55 | 2350.99 | 2342.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 13:15:00 | 2390.50 | 2395.34 | 2379.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 14:00:00 | 2390.50 | 2395.34 | 2379.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 2384.15 | 2390.97 | 2380.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:15:00 | 2379.40 | 2390.97 | 2380.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 2368.05 | 2386.39 | 2379.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 10:00:00 | 2368.05 | 2386.39 | 2379.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 10:15:00 | 2365.15 | 2382.14 | 2377.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 11:00:00 | 2365.15 | 2382.14 | 2377.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 12:15:00 | 2359.10 | 2374.89 | 2375.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 2327.05 | 2363.02 | 2369.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 09:15:00 | 2278.30 | 2261.03 | 2280.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 2278.30 | 2261.03 | 2280.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 2278.30 | 2261.03 | 2280.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:30:00 | 2295.05 | 2261.03 | 2280.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 2307.35 | 2270.29 | 2283.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 11:00:00 | 2307.35 | 2270.29 | 2283.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 11:15:00 | 2302.55 | 2276.74 | 2285.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 12:15:00 | 2316.85 | 2276.74 | 2285.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 13:15:00 | 2340.80 | 2296.56 | 2293.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 14:15:00 | 2370.15 | 2340.77 | 2321.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 2364.15 | 2364.86 | 2351.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 13:00:00 | 2364.15 | 2364.86 | 2351.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 2346.15 | 2361.13 | 2352.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 2346.15 | 2361.13 | 2352.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 2346.90 | 2358.29 | 2352.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:15:00 | 2366.30 | 2358.29 | 2352.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 2384.15 | 2363.46 | 2354.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 12:00:00 | 2392.20 | 2372.57 | 2360.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 09:15:00 | 2325.00 | 2364.88 | 2369.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 2325.00 | 2364.88 | 2369.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 2300.40 | 2346.41 | 2359.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-09 09:15:00 | 2235.65 | 2232.81 | 2258.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-09 10:00:00 | 2235.65 | 2232.81 | 2258.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 2215.05 | 2218.31 | 2237.15 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 15:15:00 | 2260.00 | 2243.09 | 2242.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 2317.80 | 2258.03 | 2249.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 13:15:00 | 2302.05 | 2306.83 | 2296.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 14:00:00 | 2302.05 | 2306.83 | 2296.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 2285.00 | 2302.47 | 2295.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 15:00:00 | 2285.00 | 2302.47 | 2295.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 2280.05 | 2297.98 | 2293.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 09:15:00 | 2300.45 | 2297.98 | 2293.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 11:15:00 | 2297.30 | 2298.19 | 2294.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-17 09:15:00 | 2244.95 | 2301.09 | 2300.05 | SL hit (close<static) qty=1.00 sl=2279.95 alert=retest2 |

### Cycle 31 — SELL (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 10:15:00 | 2242.90 | 2289.45 | 2294.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 2197.00 | 2231.82 | 2252.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 2036.90 | 2030.74 | 2062.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-27 09:30:00 | 2046.45 | 2030.74 | 2062.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 10:15:00 | 2040.00 | 2038.67 | 2050.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 10:30:00 | 2050.00 | 2038.67 | 2050.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 11:15:00 | 2057.80 | 2042.50 | 2051.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 12:00:00 | 2057.80 | 2042.50 | 2051.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 12:15:00 | 2048.15 | 2043.63 | 2051.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-30 15:00:00 | 2043.95 | 2043.93 | 2049.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 2072.15 | 2051.54 | 2052.47 | SL hit (close>static) qty=1.00 sl=2060.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 10:15:00 | 2089.90 | 2059.21 | 2055.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 13:15:00 | 2098.40 | 2072.20 | 2063.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 12:15:00 | 2080.40 | 2087.66 | 2076.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 13:00:00 | 2080.40 | 2087.66 | 2076.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 2077.90 | 2085.71 | 2076.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:45:00 | 2097.10 | 2082.37 | 2077.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 10:15:00 | 2092.00 | 2082.37 | 2077.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 11:15:00 | 2073.75 | 2079.79 | 2076.95 | SL hit (close<static) qty=1.00 sl=2076.35 alert=retest2 |

### Cycle 33 — SELL (started 2023-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 15:15:00 | 2073.00 | 2076.94 | 2077.43 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 09:15:00 | 2091.40 | 2079.83 | 2078.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 10:15:00 | 2103.10 | 2084.49 | 2080.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 09:15:00 | 2080.55 | 2094.25 | 2088.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 09:15:00 | 2080.55 | 2094.25 | 2088.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 2080.55 | 2094.25 | 2088.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:30:00 | 2077.15 | 2094.25 | 2088.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 2079.40 | 2091.28 | 2088.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 11:00:00 | 2079.40 | 2091.28 | 2088.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 2084.15 | 2087.21 | 2086.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 13:30:00 | 2089.55 | 2087.15 | 2086.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-09 15:15:00 | 2090.30 | 2101.18 | 2102.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-11-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 15:15:00 | 2090.30 | 2101.18 | 2102.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 09:15:00 | 2085.95 | 2098.13 | 2100.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 12:15:00 | 2102.75 | 2098.32 | 2100.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 12:15:00 | 2102.75 | 2098.32 | 2100.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 2102.75 | 2098.32 | 2100.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:45:00 | 2096.00 | 2098.32 | 2100.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 2089.90 | 2096.64 | 2099.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 14:15:00 | 2084.90 | 2096.64 | 2099.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-12 18:15:00 | 2106.00 | 2097.61 | 2098.79 | SL hit (close>static) qty=1.00 sl=2104.05 alert=retest2 |

### Cycle 36 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 2111.95 | 2100.48 | 2099.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 2132.05 | 2112.04 | 2106.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 14:15:00 | 2124.40 | 2128.95 | 2118.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-15 14:45:00 | 2121.00 | 2128.95 | 2118.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 15:15:00 | 2122.00 | 2127.56 | 2118.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 09:15:00 | 2119.40 | 2127.56 | 2118.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 2137.45 | 2129.54 | 2120.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 13:30:00 | 2141.70 | 2136.39 | 2126.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-22 11:15:00 | 2160.00 | 2191.99 | 2192.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 2160.00 | 2191.99 | 2192.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 12:15:00 | 2148.70 | 2183.33 | 2188.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 2176.10 | 2170.79 | 2179.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-23 09:45:00 | 2177.90 | 2170.79 | 2179.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 11:15:00 | 2180.55 | 2173.46 | 2179.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 11:45:00 | 2182.40 | 2173.46 | 2179.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 12:15:00 | 2199.95 | 2178.75 | 2181.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 12:45:00 | 2202.40 | 2178.75 | 2181.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 13:15:00 | 2187.75 | 2180.55 | 2181.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 13:30:00 | 2191.00 | 2180.55 | 2181.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-11-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 15:15:00 | 2189.20 | 2183.00 | 2182.76 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 09:15:00 | 2176.70 | 2181.74 | 2182.21 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 10:15:00 | 2211.00 | 2187.59 | 2184.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 11:15:00 | 2253.35 | 2200.74 | 2191.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 11:15:00 | 2213.45 | 2221.33 | 2209.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 12:00:00 | 2213.45 | 2221.33 | 2209.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 12:15:00 | 2214.55 | 2219.97 | 2209.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 12:45:00 | 2211.15 | 2219.97 | 2209.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 2209.75 | 2217.93 | 2209.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 14:00:00 | 2209.75 | 2217.93 | 2209.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 2204.00 | 2215.14 | 2209.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 15:00:00 | 2204.00 | 2215.14 | 2209.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 2218.20 | 2215.75 | 2209.96 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 11:15:00 | 2183.00 | 2207.26 | 2207.39 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 10:15:00 | 2226.40 | 2209.55 | 2207.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 15:15:00 | 2242.50 | 2217.09 | 2211.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 11:15:00 | 2332.45 | 2337.07 | 2313.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-06 12:00:00 | 2332.45 | 2337.07 | 2313.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 2310.55 | 2331.76 | 2313.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:00:00 | 2310.55 | 2331.76 | 2313.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 2326.05 | 2330.62 | 2314.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 14:15:00 | 2351.40 | 2330.62 | 2314.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 13:15:00 | 2310.50 | 2331.77 | 2333.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 2310.50 | 2331.77 | 2333.55 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 14:15:00 | 2341.75 | 2334.61 | 2334.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 2360.15 | 2342.18 | 2337.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 11:15:00 | 2356.20 | 2366.18 | 2356.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 11:15:00 | 2356.20 | 2366.18 | 2356.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 11:15:00 | 2356.20 | 2366.18 | 2356.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 12:00:00 | 2356.20 | 2366.18 | 2356.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 2359.65 | 2364.87 | 2357.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 14:30:00 | 2363.00 | 2366.55 | 2359.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 15:15:00 | 2366.00 | 2379.37 | 2378.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-15 15:15:00 | 2366.00 | 2376.69 | 2377.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 15:15:00 | 2366.00 | 2376.69 | 2377.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 14:15:00 | 2340.95 | 2362.66 | 2369.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 09:15:00 | 2325.10 | 2323.56 | 2339.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-20 09:30:00 | 2331.60 | 2323.56 | 2339.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 11:15:00 | 2253.15 | 2220.20 | 2232.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 12:00:00 | 2253.15 | 2220.20 | 2232.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 12:15:00 | 2232.25 | 2222.61 | 2232.43 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 2297.30 | 2244.11 | 2239.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 10:15:00 | 2324.70 | 2260.23 | 2247.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 13:15:00 | 2319.85 | 2319.96 | 2296.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 13:30:00 | 2318.00 | 2319.96 | 2296.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 2290.95 | 2314.16 | 2295.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 15:00:00 | 2290.95 | 2314.16 | 2295.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 2305.00 | 2312.32 | 2296.75 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 13:15:00 | 2266.10 | 2291.59 | 2291.65 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 09:15:00 | 2330.00 | 2295.04 | 2292.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 12:15:00 | 2354.90 | 2323.79 | 2307.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 09:15:00 | 2335.95 | 2339.52 | 2321.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 09:45:00 | 2342.15 | 2339.52 | 2321.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 2301.25 | 2331.87 | 2320.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:00:00 | 2301.25 | 2331.87 | 2320.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 2302.55 | 2326.01 | 2318.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:45:00 | 2299.00 | 2326.01 | 2318.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 14:15:00 | 2319.45 | 2320.71 | 2317.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 15:15:00 | 2325.00 | 2320.71 | 2317.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 09:45:00 | 2325.75 | 2321.14 | 2318.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 10:15:00 | 2311.05 | 2319.12 | 2317.66 | SL hit (close<static) qty=1.00 sl=2313.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 15:15:00 | 2305.00 | 2315.28 | 2316.46 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 10:15:00 | 2339.80 | 2321.12 | 2318.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 11:15:00 | 2356.05 | 2328.11 | 2322.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 2360.55 | 2380.19 | 2364.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 2360.55 | 2380.19 | 2364.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 2360.55 | 2380.19 | 2364.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 10:00:00 | 2360.55 | 2380.19 | 2364.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 2328.80 | 2369.91 | 2360.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 2328.80 | 2369.91 | 2360.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 2320.55 | 2360.04 | 2357.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 12:00:00 | 2320.55 | 2360.04 | 2357.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 2333.65 | 2354.76 | 2355.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 14:15:00 | 2303.80 | 2334.77 | 2343.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 11:15:00 | 2290.95 | 2287.81 | 2305.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-11 12:15:00 | 2300.55 | 2287.81 | 2305.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 12:15:00 | 2290.00 | 2288.24 | 2303.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 12:30:00 | 2302.15 | 2288.24 | 2303.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 2288.95 | 2288.39 | 2302.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:00:00 | 2288.95 | 2288.39 | 2302.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 2306.65 | 2292.04 | 2302.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:45:00 | 2312.15 | 2292.04 | 2302.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 2302.90 | 2294.21 | 2302.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:15:00 | 2293.40 | 2294.21 | 2302.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 2301.95 | 2295.76 | 2302.82 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 12:15:00 | 2316.60 | 2304.36 | 2303.56 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 13:15:00 | 2286.75 | 2300.84 | 2302.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 11:15:00 | 2274.15 | 2289.88 | 2295.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 2201.95 | 2189.10 | 2212.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 2201.95 | 2189.10 | 2212.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 2201.95 | 2189.10 | 2212.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 10:30:00 | 2188.80 | 2190.90 | 2211.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 11:45:00 | 2185.70 | 2187.95 | 2208.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 13:30:00 | 2192.10 | 2187.40 | 2204.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 15:15:00 | 2219.85 | 2198.41 | 2206.78 | SL hit (close>static) qty=1.00 sl=2218.05 alert=retest2 |

### Cycle 54 — BUY (started 2024-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 12:15:00 | 2223.40 | 2213.58 | 2212.27 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 14:15:00 | 2209.85 | 2211.30 | 2211.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 2163.55 | 2200.73 | 2206.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 2142.10 | 2133.20 | 2154.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 14:00:00 | 2142.10 | 2133.20 | 2154.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 2158.00 | 2138.16 | 2155.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 2158.00 | 2138.16 | 2155.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 2168.55 | 2144.24 | 2156.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 2201.30 | 2144.24 | 2156.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 2193.00 | 2153.99 | 2159.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:30:00 | 2192.40 | 2153.99 | 2159.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 2181.50 | 2159.49 | 2161.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:30:00 | 2197.00 | 2159.49 | 2161.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 11:15:00 | 2207.25 | 2169.04 | 2165.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 2255.00 | 2202.50 | 2184.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 15:15:00 | 2267.45 | 2274.24 | 2259.12 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 09:15:00 | 2284.90 | 2274.24 | 2259.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 10:00:00 | 2286.65 | 2276.72 | 2261.63 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 11:45:00 | 2285.05 | 2278.19 | 2264.93 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 12:45:00 | 2289.10 | 2279.24 | 2266.61 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 2270.80 | 2280.32 | 2273.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-02 12:15:00 | 2270.80 | 2280.32 | 2273.81 | SL hit (close<ema400) qty=1.00 sl=2273.81 alert=retest1 |

### Cycle 57 — SELL (started 2024-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 15:15:00 | 2256.00 | 2270.01 | 2270.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 2245.20 | 2265.05 | 2267.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 11:15:00 | 2222.50 | 2216.10 | 2234.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-06 11:30:00 | 2216.45 | 2216.10 | 2234.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 2200.00 | 2204.60 | 2221.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 10:30:00 | 2180.00 | 2198.68 | 2217.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 13:00:00 | 2186.85 | 2192.27 | 2210.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 14:30:00 | 2180.95 | 2187.82 | 2205.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 10:15:00 | 2077.51 | 2117.09 | 2151.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 11:15:00 | 2071.00 | 2114.79 | 2147.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 11:15:00 | 2071.90 | 2114.79 | 2147.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-09 13:15:00 | 2121.75 | 2115.08 | 2141.86 | SL hit (close>ema200) qty=0.50 sl=2115.08 alert=retest2 |

### Cycle 58 — BUY (started 2024-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 10:15:00 | 2108.00 | 2066.61 | 2062.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-21 10:15:00 | 2121.45 | 2097.69 | 2091.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 2097.50 | 2101.72 | 2095.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 13:15:00 | 2097.50 | 2101.72 | 2095.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 2097.50 | 2101.72 | 2095.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 2097.50 | 2101.72 | 2095.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 2092.00 | 2099.78 | 2095.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:45:00 | 2081.45 | 2099.78 | 2095.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 2085.00 | 2096.82 | 2094.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 2086.45 | 2096.82 | 2094.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 10:15:00 | 2086.65 | 2091.67 | 2092.15 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 13:15:00 | 2104.30 | 2093.94 | 2093.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 14:15:00 | 2105.05 | 2096.16 | 2094.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 13:15:00 | 2086.30 | 2099.31 | 2097.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 13:15:00 | 2086.30 | 2099.31 | 2097.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 13:15:00 | 2086.30 | 2099.31 | 2097.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 13:30:00 | 2082.30 | 2099.31 | 2097.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 2090.10 | 2097.47 | 2096.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 15:00:00 | 2090.10 | 2097.47 | 2096.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2024-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 15:15:00 | 2090.00 | 2095.98 | 2096.03 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 10:15:00 | 2098.10 | 2096.29 | 2096.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 11:15:00 | 2110.50 | 2099.13 | 2097.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 09:15:00 | 2099.10 | 2104.81 | 2101.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 2099.10 | 2104.81 | 2101.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 2099.10 | 2104.81 | 2101.61 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 12:15:00 | 2085.85 | 2098.31 | 2099.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 14:15:00 | 2076.15 | 2092.01 | 2096.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 2023.55 | 2022.00 | 2043.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 2023.55 | 2022.00 | 2043.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 2046.60 | 2025.94 | 2041.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:00:00 | 2046.60 | 2025.94 | 2041.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 2042.90 | 2029.33 | 2041.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:45:00 | 2044.20 | 2029.33 | 2041.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 2048.85 | 2033.23 | 2042.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:00:00 | 2048.85 | 2033.23 | 2042.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 12:15:00 | 2051.15 | 2036.82 | 2042.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 13:00:00 | 2051.15 | 2036.82 | 2042.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 09:15:00 | 2062.50 | 2048.66 | 2047.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 09:15:00 | 2069.10 | 2057.48 | 2051.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 10:15:00 | 2052.40 | 2056.46 | 2051.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 10:15:00 | 2052.40 | 2056.46 | 2051.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 2052.40 | 2056.46 | 2051.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:00:00 | 2052.40 | 2056.46 | 2051.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 2043.00 | 2053.77 | 2051.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 12:00:00 | 2043.00 | 2053.77 | 2051.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 2043.50 | 2051.72 | 2050.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 12:30:00 | 2044.70 | 2051.72 | 2050.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2024-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 13:15:00 | 2028.35 | 2047.04 | 2048.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 2012.00 | 2034.31 | 2041.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 1959.20 | 1956.72 | 1980.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 09:15:00 | 1960.00 | 1960.05 | 1971.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 1960.00 | 1960.05 | 1971.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 10:15:00 | 1957.90 | 1960.05 | 1971.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 11:45:00 | 1948.45 | 1955.24 | 1966.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 13:15:00 | 1957.65 | 1956.16 | 1966.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 13:45:00 | 1956.15 | 1954.93 | 1964.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 1860.01 | 1904.29 | 1927.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 1859.77 | 1904.29 | 1927.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 1858.34 | 1904.29 | 1927.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:15:00 | 1851.03 | 1882.11 | 1912.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-14 09:15:00 | 1865.30 | 1854.08 | 1884.85 | SL hit (close>ema200) qty=0.50 sl=1854.08 alert=retest2 |

### Cycle 66 — BUY (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 13:15:00 | 1886.90 | 1877.60 | 1876.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 15:15:00 | 1889.50 | 1881.40 | 1878.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 1871.25 | 1879.37 | 1877.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 1871.25 | 1879.37 | 1877.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 1871.25 | 1879.37 | 1877.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 1871.25 | 1879.37 | 1877.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1882.35 | 1879.97 | 1878.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 11:45:00 | 1885.00 | 1880.97 | 1878.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 1852.60 | 1875.92 | 1877.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 09:15:00 | 1852.60 | 1875.92 | 1877.64 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 14:15:00 | 1896.95 | 1880.37 | 1878.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 1914.25 | 1889.19 | 1883.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 1937.50 | 1941.27 | 1923.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 14:15:00 | 1967.70 | 1946.68 | 1931.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 09:15:00 | 1975.00 | 1951.24 | 1936.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 1957.00 | 1964.00 | 1952.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 1948.30 | 1964.00 | 1952.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1955.00 | 1962.20 | 1952.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:30:00 | 1970.80 | 1963.29 | 1955.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 13:15:00 | 1955.25 | 1961.48 | 1955.63 | SL hit (close<ema400) qty=1.00 sl=1955.63 alert=retest1 |

### Cycle 69 — SELL (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 11:15:00 | 1990.00 | 2012.98 | 2013.58 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 09:15:00 | 2041.80 | 2010.49 | 2007.45 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 1985.15 | 2005.60 | 2006.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 12:15:00 | 1984.45 | 1994.61 | 1999.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 10:15:00 | 1952.10 | 1946.46 | 1958.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 10:45:00 | 1953.70 | 1946.46 | 1958.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 1955.20 | 1947.96 | 1956.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 13:00:00 | 1955.20 | 1947.96 | 1956.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 1970.50 | 1952.47 | 1958.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:00:00 | 1970.50 | 1952.47 | 1958.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 1970.50 | 1956.08 | 1959.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 15:00:00 | 1970.50 | 1956.08 | 1959.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 1978.90 | 1960.64 | 1960.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 1966.50 | 1960.64 | 1960.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 1955.30 | 1959.79 | 1960.51 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 13:15:00 | 1963.95 | 1961.02 | 1960.93 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 1945.00 | 1957.82 | 1959.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 10:15:00 | 1937.05 | 1950.35 | 1955.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 12:15:00 | 1954.70 | 1950.39 | 1954.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 12:15:00 | 1954.70 | 1950.39 | 1954.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 1954.70 | 1950.39 | 1954.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 12:45:00 | 1960.00 | 1950.39 | 1954.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 1953.45 | 1951.00 | 1954.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 14:30:00 | 1942.85 | 1949.66 | 1953.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 10:15:00 | 1960.95 | 1951.25 | 1953.13 | SL hit (close>static) qty=1.00 sl=1957.70 alert=retest2 |

### Cycle 74 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 1960.15 | 1954.82 | 1954.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 1966.00 | 1958.63 | 1956.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 09:15:00 | 1943.90 | 1955.68 | 1955.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 09:15:00 | 1943.90 | 1955.68 | 1955.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 1943.90 | 1955.68 | 1955.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 09:45:00 | 1934.50 | 1955.68 | 1955.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 10:15:00 | 1948.95 | 1954.33 | 1954.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 15:15:00 | 1942.80 | 1950.48 | 1952.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 1966.35 | 1953.66 | 1953.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 09:15:00 | 1966.35 | 1953.66 | 1953.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 1966.35 | 1953.66 | 1953.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:30:00 | 1978.00 | 1953.66 | 1953.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 10:15:00 | 1971.75 | 1957.27 | 1955.46 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 09:15:00 | 1880.15 | 1947.36 | 1952.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 10:15:00 | 1852.70 | 1928.43 | 1943.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 11:15:00 | 1824.00 | 1813.56 | 1841.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 11:45:00 | 1821.55 | 1813.56 | 1841.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 1834.20 | 1825.20 | 1837.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:30:00 | 1838.10 | 1825.20 | 1837.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 1840.40 | 1828.24 | 1837.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:30:00 | 1838.00 | 1828.24 | 1837.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 1841.00 | 1830.79 | 1837.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 11:30:00 | 1836.95 | 1830.79 | 1837.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 1845.90 | 1833.81 | 1838.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:45:00 | 1841.35 | 1833.81 | 1838.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 1837.70 | 1836.96 | 1839.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 1824.50 | 1836.96 | 1839.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 1802.30 | 1830.03 | 1835.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:15:00 | 1799.15 | 1830.03 | 1835.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 12:45:00 | 1798.15 | 1818.09 | 1828.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 1797.80 | 1805.25 | 1818.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 09:30:00 | 1796.65 | 1793.31 | 1804.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 1794.30 | 1794.20 | 1800.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 09:15:00 | 1790.55 | 1795.76 | 1800.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 12:15:00 | 1709.19 | 1738.50 | 1752.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 13:15:00 | 1708.24 | 1733.37 | 1749.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 13:15:00 | 1707.91 | 1733.37 | 1749.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 13:15:00 | 1706.82 | 1733.37 | 1749.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 10:15:00 | 1701.02 | 1724.01 | 1739.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 11:15:00 | 1727.20 | 1724.65 | 1738.15 | SL hit (close>ema200) qty=0.50 sl=1724.65 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 1756.70 | 1742.20 | 1741.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 15:15:00 | 1763.00 | 1756.30 | 1750.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 1830.00 | 1833.92 | 1818.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 09:45:00 | 1836.65 | 1833.92 | 1818.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1838.80 | 1841.55 | 1830.63 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 13:15:00 | 1814.00 | 1824.32 | 1824.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 1793.00 | 1809.13 | 1815.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 1804.95 | 1804.72 | 1811.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 11:00:00 | 1804.95 | 1804.72 | 1811.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 1810.00 | 1805.78 | 1809.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 15:00:00 | 1810.00 | 1805.78 | 1809.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 1805.00 | 1805.63 | 1809.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:30:00 | 1797.50 | 1803.81 | 1807.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:15:00 | 1793.80 | 1803.81 | 1807.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 13:00:00 | 1797.50 | 1802.55 | 1806.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 14:15:00 | 1796.95 | 1802.04 | 1805.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 1806.00 | 1802.83 | 1805.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 1806.00 | 1802.83 | 1805.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 1793.45 | 1800.95 | 1804.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 1803.25 | 1800.95 | 1804.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1787.95 | 1798.35 | 1803.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 12:15:00 | 1776.30 | 1793.85 | 1800.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 14:00:00 | 1780.00 | 1789.82 | 1797.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1829.45 | 1783.27 | 1779.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1829.45 | 1783.27 | 1779.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 10:15:00 | 1835.60 | 1793.74 | 1784.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1772.00 | 1826.20 | 1810.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1772.00 | 1826.20 | 1810.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1772.00 | 1826.20 | 1810.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1748.20 | 1826.20 | 1810.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1706.65 | 1802.29 | 1800.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1706.65 | 1802.29 | 1800.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1692.05 | 1780.24 | 1790.91 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 14:15:00 | 1773.30 | 1755.59 | 1754.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 1779.80 | 1760.43 | 1756.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 10:15:00 | 1882.25 | 1883.21 | 1860.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:45:00 | 1871.80 | 1883.21 | 1860.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1880.10 | 1891.81 | 1883.05 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 1868.00 | 1882.50 | 1882.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 1852.75 | 1870.84 | 1876.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 13:15:00 | 1860.50 | 1857.02 | 1865.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 14:00:00 | 1860.50 | 1857.02 | 1865.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1838.50 | 1854.94 | 1862.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:30:00 | 1826.45 | 1850.38 | 1859.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 10:30:00 | 1831.65 | 1825.32 | 1827.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:15:00 | 1832.65 | 1811.26 | 1811.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 10:15:00 | 1832.95 | 1815.60 | 1813.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 1832.95 | 1815.60 | 1813.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 1837.95 | 1822.49 | 1817.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 1835.35 | 1839.20 | 1829.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 1835.35 | 1839.20 | 1829.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 1822.30 | 1835.82 | 1829.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 1822.30 | 1835.82 | 1829.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1830.95 | 1834.85 | 1829.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:30:00 | 1830.00 | 1834.85 | 1829.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 1838.85 | 1835.65 | 1830.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 12:30:00 | 1841.90 | 1837.50 | 1833.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:30:00 | 1843.75 | 1845.38 | 1839.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:00:00 | 1840.50 | 1845.38 | 1839.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:30:00 | 1847.00 | 1845.98 | 1843.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 1838.50 | 1844.49 | 1842.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:30:00 | 1839.95 | 1844.49 | 1842.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1844.95 | 1844.58 | 1842.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:15:00 | 1860.40 | 1844.58 | 1842.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 1831.70 | 1853.66 | 1854.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 1831.70 | 1853.66 | 1854.58 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 1864.75 | 1855.27 | 1854.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 1870.25 | 1862.06 | 1858.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 1906.00 | 1919.81 | 1907.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 11:15:00 | 1906.00 | 1919.81 | 1907.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 1906.00 | 1919.81 | 1907.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 1906.00 | 1919.81 | 1907.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 1913.00 | 1918.45 | 1908.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:30:00 | 1905.00 | 1918.45 | 1908.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1900.00 | 1914.76 | 1907.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 1900.00 | 1914.76 | 1907.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1915.00 | 1914.81 | 1908.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 15:15:00 | 1918.50 | 1914.81 | 1908.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 10:15:00 | 1895.70 | 1910.34 | 1907.89 | SL hit (close<static) qty=1.00 sl=1899.05 alert=retest2 |

### Cycle 87 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1847.75 | 1901.47 | 1905.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 1815.20 | 1884.22 | 1897.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 1803.25 | 1793.54 | 1822.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 11:00:00 | 1803.25 | 1793.54 | 1822.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1801.80 | 1777.14 | 1782.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 1801.80 | 1777.14 | 1782.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 1822.20 | 1786.15 | 1786.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 1834.35 | 1795.79 | 1790.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 1847.15 | 1847.63 | 1838.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 1847.15 | 1847.63 | 1838.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1865.40 | 1852.36 | 1842.15 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 1812.25 | 1837.20 | 1837.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 1801.95 | 1824.73 | 1831.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1791.30 | 1775.45 | 1792.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1791.30 | 1775.45 | 1792.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1791.30 | 1775.45 | 1792.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 1791.30 | 1775.45 | 1792.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1775.00 | 1775.36 | 1790.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:45:00 | 1766.00 | 1773.49 | 1788.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 1770.00 | 1758.67 | 1761.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:45:00 | 1770.00 | 1760.23 | 1762.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 15:15:00 | 1770.05 | 1763.71 | 1763.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 15:15:00 | 1770.05 | 1763.71 | 1763.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1775.00 | 1765.97 | 1764.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 11:15:00 | 1759.00 | 1765.32 | 1764.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 11:15:00 | 1759.00 | 1765.32 | 1764.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 1759.00 | 1765.32 | 1764.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:00:00 | 1759.00 | 1765.32 | 1764.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 12:15:00 | 1750.80 | 1762.41 | 1763.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 13:15:00 | 1739.25 | 1757.78 | 1761.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 14:15:00 | 1749.35 | 1746.01 | 1751.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-12 15:00:00 | 1749.35 | 1746.01 | 1751.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1732.05 | 1726.51 | 1732.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 12:15:00 | 1727.50 | 1728.08 | 1732.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:00:00 | 1729.65 | 1728.40 | 1731.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 14:15:00 | 1753.00 | 1734.29 | 1734.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 1753.00 | 1734.29 | 1734.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1759.40 | 1741.84 | 1737.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 1761.20 | 1764.35 | 1754.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 1761.20 | 1764.35 | 1754.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 1755.10 | 1762.50 | 1754.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 1754.00 | 1762.50 | 1754.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1770.00 | 1764.00 | 1755.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 1754.25 | 1764.00 | 1755.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1770.45 | 1767.73 | 1761.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 1778.95 | 1761.68 | 1760.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:45:00 | 1782.40 | 1767.53 | 1763.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1875.50 | 1907.44 | 1909.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1875.50 | 1907.44 | 1909.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 1842.00 | 1878.22 | 1891.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 1876.45 | 1870.37 | 1881.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 1876.45 | 1870.37 | 1881.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1899.00 | 1877.41 | 1883.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 1899.00 | 1877.41 | 1883.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 1899.60 | 1881.85 | 1884.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:30:00 | 1908.15 | 1881.85 | 1884.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 1903.70 | 1889.10 | 1887.66 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 11:15:00 | 1879.35 | 1888.60 | 1889.82 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 1909.80 | 1892.10 | 1890.85 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 1870.10 | 1891.15 | 1892.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 13:15:00 | 1865.75 | 1882.70 | 1888.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 1854.80 | 1843.29 | 1858.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 10:00:00 | 1854.80 | 1843.29 | 1858.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1847.10 | 1844.05 | 1857.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:45:00 | 1837.95 | 1841.64 | 1855.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:15:00 | 1842.25 | 1830.76 | 1833.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 1865.55 | 1840.56 | 1837.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 1865.55 | 1840.56 | 1837.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 1868.45 | 1846.13 | 1840.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 1911.05 | 1915.49 | 1896.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:00:00 | 1911.05 | 1915.49 | 1896.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 1901.40 | 1932.99 | 1920.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:00:00 | 1901.40 | 1932.99 | 1920.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 1917.10 | 1929.81 | 1920.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 10:00:00 | 1926.70 | 1920.88 | 1918.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 1900.00 | 1939.95 | 1944.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 1900.00 | 1939.95 | 1944.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 1867.45 | 1912.19 | 1929.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 1897.25 | 1888.77 | 1910.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 1897.25 | 1888.77 | 1910.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 1889.50 | 1891.99 | 1908.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:30:00 | 1879.15 | 1890.77 | 1906.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 13:45:00 | 1882.20 | 1888.90 | 1904.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 1883.00 | 1891.28 | 1902.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:30:00 | 1873.60 | 1883.60 | 1896.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 1852.40 | 1852.12 | 1860.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:30:00 | 1855.60 | 1852.12 | 1860.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 1858.50 | 1853.39 | 1860.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:45:00 | 1860.75 | 1853.39 | 1860.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 1872.00 | 1857.12 | 1861.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 1872.00 | 1857.12 | 1861.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 1870.15 | 1859.72 | 1862.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 1880.95 | 1859.72 | 1862.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-14 10:15:00 | 1878.50 | 1866.63 | 1865.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 1878.50 | 1866.63 | 1865.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 11:15:00 | 1897.75 | 1872.86 | 1868.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 1881.10 | 1882.46 | 1875.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1881.10 | 1882.46 | 1875.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1881.10 | 1882.46 | 1875.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 1881.10 | 1882.46 | 1875.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1875.70 | 1881.11 | 1875.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 1875.70 | 1881.11 | 1875.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1865.45 | 1877.97 | 1874.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:45:00 | 1867.55 | 1877.97 | 1874.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 1860.40 | 1874.46 | 1873.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:45:00 | 1857.55 | 1874.46 | 1873.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1867.60 | 1875.90 | 1874.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 1867.60 | 1875.90 | 1874.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 1867.00 | 1874.12 | 1874.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 1842.20 | 1864.04 | 1868.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 1842.80 | 1839.15 | 1850.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:30:00 | 1837.75 | 1839.15 | 1850.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1855.40 | 1842.40 | 1850.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 1855.40 | 1842.40 | 1850.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1848.45 | 1843.61 | 1850.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 14:30:00 | 1845.00 | 1848.27 | 1851.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:15:00 | 1840.00 | 1848.27 | 1851.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 09:15:00 | 1752.75 | 1792.54 | 1805.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 1817.60 | 1792.54 | 1805.30 | SL hit (close>static) qty=0.50 sl=1792.54 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1822.10 | 1795.80 | 1795.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 1828.40 | 1805.77 | 1800.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 1829.60 | 1831.33 | 1820.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 13:00:00 | 1829.60 | 1831.33 | 1820.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1834.05 | 1832.99 | 1822.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:30:00 | 1826.90 | 1832.99 | 1822.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1809.00 | 1830.80 | 1825.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1809.00 | 1830.80 | 1825.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1788.00 | 1822.24 | 1822.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1788.00 | 1822.24 | 1822.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 1788.55 | 1815.50 | 1818.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 12:15:00 | 1780.55 | 1808.51 | 1815.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 1792.25 | 1785.96 | 1796.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 14:45:00 | 1790.85 | 1785.96 | 1796.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 1792.30 | 1787.22 | 1795.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 1812.10 | 1793.40 | 1797.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1818.45 | 1798.41 | 1799.66 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 1823.25 | 1803.38 | 1801.81 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 12:15:00 | 1798.15 | 1803.35 | 1804.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 1782.90 | 1796.48 | 1800.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1765.20 | 1762.48 | 1772.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:15:00 | 1774.10 | 1762.48 | 1772.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1771.10 | 1764.20 | 1772.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 1768.40 | 1764.20 | 1772.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1771.05 | 1765.57 | 1772.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:15:00 | 1775.40 | 1765.57 | 1772.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 1773.95 | 1767.25 | 1772.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:30:00 | 1776.80 | 1767.25 | 1772.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 1779.65 | 1769.73 | 1773.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:00:00 | 1779.65 | 1769.73 | 1773.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 1787.70 | 1773.32 | 1774.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 1787.70 | 1773.32 | 1774.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 15:15:00 | 1787.95 | 1776.25 | 1775.90 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 12:15:00 | 1764.75 | 1774.22 | 1775.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 13:15:00 | 1750.20 | 1769.41 | 1772.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 1721.30 | 1719.77 | 1734.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 12:00:00 | 1721.30 | 1719.77 | 1734.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1744.85 | 1715.70 | 1724.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 1744.85 | 1715.70 | 1724.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 1752.65 | 1723.09 | 1726.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:45:00 | 1754.20 | 1723.09 | 1726.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1760.10 | 1730.49 | 1729.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 12:15:00 | 1765.00 | 1745.87 | 1739.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 15:15:00 | 1818.15 | 1825.79 | 1805.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:15:00 | 1816.00 | 1825.79 | 1805.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 1811.00 | 1820.85 | 1806.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:30:00 | 1809.60 | 1820.85 | 1806.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 1823.10 | 1818.99 | 1810.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:30:00 | 1825.75 | 1817.67 | 1812.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 1825.85 | 1817.67 | 1812.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:00:00 | 1834.85 | 1821.11 | 1814.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 11:30:00 | 1829.35 | 1828.89 | 1822.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1815.90 | 1826.29 | 1821.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 1815.90 | 1826.29 | 1821.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 1816.90 | 1824.41 | 1821.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 1816.90 | 1824.41 | 1821.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1818.65 | 1823.26 | 1820.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 1828.00 | 1823.01 | 1821.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 1829.60 | 1821.24 | 1820.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 12:15:00 | 1813.50 | 1819.37 | 1819.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 12:15:00 | 1813.50 | 1819.37 | 1819.72 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 1843.85 | 1823.61 | 1821.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 1872.75 | 1839.67 | 1829.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 12:15:00 | 1908.70 | 1912.75 | 1892.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 13:00:00 | 1908.70 | 1912.75 | 1892.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 1903.65 | 1920.62 | 1908.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:00:00 | 1903.65 | 1920.62 | 1908.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 1899.50 | 1916.39 | 1907.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:30:00 | 1890.00 | 1916.39 | 1907.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 1911.35 | 1915.32 | 1908.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:45:00 | 1905.70 | 1915.31 | 1909.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1916.50 | 1915.55 | 1909.83 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 10:15:00 | 1890.95 | 1910.58 | 1911.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 11:15:00 | 1888.60 | 1896.63 | 1902.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 13:15:00 | 1895.00 | 1893.93 | 1900.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 14:00:00 | 1895.00 | 1893.93 | 1900.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 1899.55 | 1895.05 | 1899.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 1899.55 | 1895.05 | 1899.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1898.45 | 1895.73 | 1899.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 1934.55 | 1895.73 | 1899.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 1949.40 | 1906.47 | 1904.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 11:15:00 | 1953.45 | 1922.83 | 1912.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 10:15:00 | 1940.75 | 1946.18 | 1931.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 10:45:00 | 1941.55 | 1946.18 | 1931.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1938.20 | 1944.58 | 1931.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 1938.20 | 1944.58 | 1931.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1933.25 | 1942.31 | 1932.02 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 1870.55 | 1925.72 | 1927.48 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 1947.00 | 1923.95 | 1923.71 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 1916.70 | 1933.17 | 1934.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 1902.30 | 1927.00 | 1931.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 10:15:00 | 1747.45 | 1739.22 | 1768.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 11:00:00 | 1747.45 | 1739.22 | 1768.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1742.90 | 1741.07 | 1754.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 1745.55 | 1741.07 | 1754.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1750.25 | 1741.14 | 1751.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 1750.25 | 1741.14 | 1751.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 1740.00 | 1740.91 | 1750.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 15:15:00 | 1735.05 | 1740.91 | 1750.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 1772.80 | 1746.35 | 1751.04 | SL hit (close>static) qty=1.00 sl=1750.85 alert=retest2 |

### Cycle 116 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 1771.35 | 1755.42 | 1754.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 1784.90 | 1771.93 | 1766.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 15:15:00 | 1765.50 | 1773.86 | 1768.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 15:15:00 | 1765.50 | 1773.86 | 1768.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 1765.50 | 1773.86 | 1768.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:45:00 | 1794.45 | 1778.06 | 1771.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 1784.30 | 1809.75 | 1802.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 12:15:00 | 1771.15 | 1796.74 | 1797.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1771.15 | 1796.74 | 1797.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1756.05 | 1788.61 | 1793.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1788.95 | 1779.50 | 1787.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1788.95 | 1779.50 | 1787.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1788.95 | 1779.50 | 1787.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 1797.70 | 1779.50 | 1787.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1785.35 | 1780.67 | 1787.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:15:00 | 1790.05 | 1780.67 | 1787.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1784.95 | 1781.52 | 1787.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 1788.90 | 1781.52 | 1787.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1795.00 | 1784.22 | 1787.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 1794.25 | 1784.22 | 1787.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1807.20 | 1788.82 | 1789.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 1805.90 | 1788.82 | 1789.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 1798.95 | 1790.84 | 1790.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 09:15:00 | 1827.55 | 1799.33 | 1794.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 14:15:00 | 1802.85 | 1806.05 | 1800.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 14:15:00 | 1802.85 | 1806.05 | 1800.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1802.85 | 1806.05 | 1800.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1802.85 | 1806.05 | 1800.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1798.95 | 1804.63 | 1800.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 1782.00 | 1804.63 | 1800.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1813.05 | 1806.32 | 1801.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:30:00 | 1779.00 | 1806.32 | 1801.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1801.55 | 1805.36 | 1801.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:45:00 | 1801.95 | 1805.36 | 1801.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 1800.75 | 1804.44 | 1801.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:00:00 | 1800.75 | 1804.44 | 1801.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 1803.50 | 1804.25 | 1801.56 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 1763.50 | 1795.90 | 1798.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 12:15:00 | 1745.35 | 1777.82 | 1789.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 1723.20 | 1714.88 | 1737.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 1723.20 | 1714.88 | 1737.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1728.25 | 1718.74 | 1731.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 1728.25 | 1718.74 | 1731.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 1714.75 | 1717.94 | 1730.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:45:00 | 1735.90 | 1718.26 | 1729.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 1731.00 | 1720.80 | 1729.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 1726.90 | 1720.80 | 1729.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 1737.60 | 1724.16 | 1730.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:45:00 | 1738.85 | 1724.16 | 1730.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 1738.45 | 1729.37 | 1731.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:45:00 | 1738.20 | 1729.37 | 1731.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 1731.00 | 1729.70 | 1731.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:15:00 | 1758.30 | 1729.70 | 1731.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 1748.45 | 1733.45 | 1732.84 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 1704.25 | 1729.91 | 1732.38 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 1748.30 | 1730.05 | 1729.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 14:15:00 | 1751.30 | 1734.30 | 1731.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 11:15:00 | 1810.15 | 1812.62 | 1799.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 11:45:00 | 1805.75 | 1812.62 | 1799.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 1801.45 | 1809.23 | 1800.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 15:00:00 | 1801.45 | 1809.23 | 1800.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 1810.00 | 1809.38 | 1801.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 1798.45 | 1809.38 | 1801.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1800.00 | 1807.51 | 1801.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 13:00:00 | 1808.65 | 1805.15 | 1801.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 09:15:00 | 1787.65 | 1798.84 | 1799.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 1787.65 | 1798.84 | 1799.59 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 1826.20 | 1803.29 | 1801.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 1828.00 | 1808.23 | 1803.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-28 15:15:00 | 1805.50 | 1811.09 | 1806.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 15:15:00 | 1805.50 | 1811.09 | 1806.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 1805.50 | 1811.09 | 1806.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 1812.75 | 1811.09 | 1806.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1846.00 | 1818.08 | 1810.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 10:15:00 | 1857.10 | 1818.08 | 1810.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 14:45:00 | 1848.95 | 1847.07 | 1837.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 1853.10 | 1843.72 | 1837.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 13:15:00 | 1817.65 | 1850.48 | 1850.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 1817.65 | 1850.48 | 1850.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 1799.15 | 1831.99 | 1840.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 10:15:00 | 1851.55 | 1821.92 | 1828.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 10:15:00 | 1851.55 | 1821.92 | 1828.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1851.55 | 1821.92 | 1828.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 1851.55 | 1821.92 | 1828.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 1838.20 | 1825.18 | 1829.44 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 1851.05 | 1834.65 | 1833.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1867.60 | 1846.66 | 1839.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 1864.35 | 1876.73 | 1862.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 1864.35 | 1876.73 | 1862.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1864.35 | 1876.73 | 1862.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1869.05 | 1876.73 | 1862.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1867.30 | 1874.84 | 1862.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 1867.35 | 1874.84 | 1862.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1868.05 | 1873.48 | 1863.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:45:00 | 1866.15 | 1873.48 | 1863.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1878.70 | 1874.53 | 1864.66 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 1845.10 | 1861.74 | 1862.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1828.10 | 1852.70 | 1858.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 1793.40 | 1790.04 | 1808.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 1792.10 | 1790.04 | 1808.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1794.65 | 1793.15 | 1807.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 1803.65 | 1793.15 | 1807.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1821.70 | 1800.11 | 1806.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 1821.70 | 1800.11 | 1806.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1819.50 | 1803.99 | 1808.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 1819.50 | 1803.99 | 1808.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 1809.65 | 1805.96 | 1808.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:30:00 | 1812.65 | 1805.96 | 1808.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1776.60 | 1796.72 | 1802.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 1769.85 | 1791.78 | 1800.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:00:00 | 1769.75 | 1787.37 | 1797.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 1766.15 | 1780.69 | 1790.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:45:00 | 1774.25 | 1778.83 | 1788.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1778.35 | 1778.74 | 1787.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 1791.45 | 1778.74 | 1787.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 1783.10 | 1779.61 | 1787.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:00:00 | 1783.10 | 1779.61 | 1787.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 1795.90 | 1782.87 | 1787.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 1795.90 | 1782.87 | 1787.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1803.60 | 1787.01 | 1789.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 1811.80 | 1787.01 | 1789.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 1808.00 | 1791.21 | 1791.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 1808.00 | 1791.21 | 1791.10 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 1786.00 | 1790.44 | 1790.91 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 1799.45 | 1791.95 | 1791.37 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 1783.95 | 1791.15 | 1791.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 14:15:00 | 1772.50 | 1786.11 | 1789.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 10:15:00 | 1792.85 | 1783.69 | 1786.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 10:15:00 | 1792.85 | 1783.69 | 1786.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 1792.85 | 1783.69 | 1786.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:00:00 | 1792.85 | 1783.69 | 1786.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 1784.50 | 1783.85 | 1786.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 12:15:00 | 1781.75 | 1783.85 | 1786.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 09:30:00 | 1769.00 | 1775.32 | 1781.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 09:15:00 | 1692.66 | 1722.95 | 1735.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 12:15:00 | 1745.95 | 1725.04 | 1733.16 | SL hit (close>ema200) qty=0.50 sl=1725.04 alert=retest2 |

### Cycle 132 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 1711.00 | 1700.92 | 1700.86 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 11:15:00 | 1687.50 | 1703.94 | 1704.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 12:15:00 | 1681.20 | 1699.39 | 1702.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 09:15:00 | 1697.50 | 1697.09 | 1699.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1697.50 | 1697.09 | 1699.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1697.50 | 1697.09 | 1699.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:45:00 | 1700.00 | 1697.09 | 1699.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1685.80 | 1694.83 | 1698.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 11:45:00 | 1683.85 | 1691.17 | 1696.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 15:00:00 | 1685.00 | 1677.68 | 1683.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 1668.30 | 1680.82 | 1684.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 14:15:00 | 1665.25 | 1646.81 | 1644.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 1665.25 | 1646.81 | 1644.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1689.05 | 1656.51 | 1649.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1703.20 | 1717.84 | 1700.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 1703.20 | 1717.84 | 1700.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1703.20 | 1717.84 | 1700.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 1703.20 | 1717.84 | 1700.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 1702.75 | 1714.82 | 1700.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:45:00 | 1702.10 | 1714.82 | 1700.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 1700.00 | 1711.86 | 1700.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:00:00 | 1700.00 | 1711.86 | 1700.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 1702.00 | 1709.88 | 1700.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:00:00 | 1702.00 | 1709.88 | 1700.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 1703.85 | 1708.68 | 1701.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:45:00 | 1710.30 | 1703.28 | 1700.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:30:00 | 1710.50 | 1706.34 | 1702.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 13:00:00 | 1716.75 | 1706.34 | 1702.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-03 10:15:00 | 1881.33 | 1834.26 | 1816.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1784.35 | 1834.30 | 1834.43 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 1831.10 | 1815.36 | 1814.17 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 11:15:00 | 1807.15 | 1812.95 | 1813.55 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 1853.00 | 1819.32 | 1815.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 1882.60 | 1837.47 | 1825.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 1859.90 | 1861.56 | 1846.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 12:00:00 | 1859.90 | 1861.56 | 1846.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 1849.00 | 1856.99 | 1846.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 1839.20 | 1856.99 | 1846.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 1845.80 | 1854.75 | 1846.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 14:45:00 | 1846.50 | 1854.75 | 1846.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 1846.20 | 1853.04 | 1846.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:15:00 | 1808.70 | 1853.04 | 1846.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1832.10 | 1848.85 | 1845.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:15:00 | 1860.60 | 1849.54 | 1846.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 1861.90 | 1853.88 | 1849.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 13:15:00 | 1876.30 | 1890.15 | 1891.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 13:15:00 | 1876.30 | 1890.15 | 1891.23 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 09:15:00 | 1958.20 | 1904.64 | 1897.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 1997.80 | 1971.41 | 1951.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 15:15:00 | 1979.00 | 1979.36 | 1965.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 09:15:00 | 1977.50 | 1979.36 | 1965.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1969.30 | 1975.63 | 1965.98 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 1938.00 | 1958.53 | 1960.83 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 10:15:00 | 1962.80 | 1952.64 | 1951.92 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 1938.20 | 1949.40 | 1950.76 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 1966.70 | 1952.86 | 1952.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 11:15:00 | 1979.10 | 1958.11 | 1954.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 1962.20 | 1968.58 | 1964.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 1962.20 | 1968.58 | 1964.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1962.20 | 1968.58 | 1964.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 1963.30 | 1968.58 | 1964.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1968.40 | 1968.55 | 1964.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:15:00 | 1950.00 | 1968.55 | 1964.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1950.00 | 1964.84 | 1963.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1932.40 | 1964.84 | 1963.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1922.90 | 1956.45 | 1959.44 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1986.10 | 1953.57 | 1950.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 2011.40 | 1977.33 | 1964.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 2070.00 | 2071.32 | 2049.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 09:45:00 | 2068.40 | 2071.32 | 2049.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 2074.00 | 2073.58 | 2064.79 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 2058.80 | 2063.41 | 2063.51 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 2084.90 | 2066.20 | 2064.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 2088.10 | 2070.58 | 2066.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 09:15:00 | 2107.40 | 2128.23 | 2113.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 2107.40 | 2128.23 | 2113.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2107.40 | 2128.23 | 2113.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 2107.40 | 2128.23 | 2113.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 2086.40 | 2119.86 | 2110.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 2086.40 | 2119.86 | 2110.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 2063.10 | 2108.51 | 2106.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:45:00 | 2067.10 | 2108.51 | 2106.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 12:15:00 | 2067.80 | 2100.37 | 2102.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 2053.90 | 2091.07 | 2098.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 14:15:00 | 2065.30 | 2062.93 | 2075.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-27 15:00:00 | 2065.30 | 2062.93 | 2075.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 2055.30 | 2061.10 | 2072.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:00:00 | 2040.90 | 2053.86 | 2063.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:30:00 | 2041.80 | 2049.69 | 2059.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 2032.20 | 2040.25 | 2051.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 2070.00 | 2042.42 | 2041.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 2070.00 | 2042.42 | 2041.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 2083.00 | 2050.54 | 2045.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 2056.70 | 2062.96 | 2055.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 2056.70 | 2062.96 | 2055.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 2056.70 | 2062.96 | 2055.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 2056.70 | 2062.96 | 2055.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 2059.90 | 2062.35 | 2055.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:30:00 | 2055.30 | 2062.35 | 2055.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 2064.10 | 2062.70 | 2056.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 2064.10 | 2062.70 | 2056.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 2051.00 | 2060.36 | 2056.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 2060.00 | 2060.36 | 2056.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 2052.30 | 2058.75 | 2055.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 2050.60 | 2058.75 | 2055.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 2057.60 | 2058.52 | 2055.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 2074.90 | 2058.01 | 2056.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 2100.80 | 2129.11 | 2129.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 2100.80 | 2129.11 | 2129.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 2093.50 | 2114.12 | 2122.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 2104.70 | 2096.22 | 2106.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 2104.70 | 2096.22 | 2106.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 2104.70 | 2096.22 | 2106.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 2104.70 | 2096.22 | 2106.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 2116.00 | 2100.17 | 2107.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 2116.00 | 2100.17 | 2107.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2112.30 | 2102.60 | 2107.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:30:00 | 2114.00 | 2102.60 | 2107.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 2100.20 | 2101.46 | 2106.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 2100.20 | 2101.46 | 2106.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2086.00 | 2096.65 | 2102.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 2081.00 | 2092.30 | 2099.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 2075.30 | 2058.49 | 2056.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 2075.30 | 2058.49 | 2056.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 2099.00 | 2071.51 | 2063.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 2076.20 | 2077.58 | 2069.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 2075.90 | 2077.58 | 2069.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 2067.20 | 2074.73 | 2069.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:45:00 | 2096.60 | 2079.26 | 2071.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 2179.40 | 2202.78 | 2205.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 14:15:00 | 2179.40 | 2202.78 | 2205.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 15:15:00 | 2174.00 | 2197.02 | 2202.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 2156.10 | 2151.70 | 2166.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:45:00 | 2156.00 | 2151.70 | 2166.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 2160.60 | 2153.67 | 2164.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 2158.60 | 2153.67 | 2164.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2172.90 | 2156.69 | 2161.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 2174.00 | 2156.69 | 2161.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 2186.60 | 2162.67 | 2163.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 2181.90 | 2162.67 | 2163.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 2174.00 | 2164.94 | 2164.85 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 2150.40 | 2168.50 | 2170.69 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 2180.30 | 2169.15 | 2169.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 2191.30 | 2176.25 | 2172.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 2161.10 | 2175.85 | 2173.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 2161.10 | 2175.85 | 2173.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 2161.10 | 2175.85 | 2173.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 2155.00 | 2175.85 | 2173.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 11:15:00 | 2150.00 | 2170.68 | 2171.08 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 14:15:00 | 2188.70 | 2170.45 | 2169.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 2208.00 | 2181.09 | 2174.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 2286.00 | 2305.04 | 2279.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2286.00 | 2305.04 | 2279.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2286.00 | 2305.04 | 2279.63 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 2252.80 | 2263.49 | 2264.63 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 2273.50 | 2265.49 | 2265.44 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 2256.80 | 2265.61 | 2265.64 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 11:15:00 | 2268.80 | 2266.25 | 2265.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 12:15:00 | 2283.00 | 2269.60 | 2267.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 15:15:00 | 2269.00 | 2269.89 | 2268.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 15:15:00 | 2269.00 | 2269.89 | 2268.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 2269.00 | 2269.89 | 2268.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 2263.70 | 2269.89 | 2268.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 2261.60 | 2268.24 | 2267.59 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 2243.80 | 2263.35 | 2265.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 2231.30 | 2253.63 | 2260.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2250.50 | 2243.86 | 2252.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2250.50 | 2243.86 | 2252.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2250.50 | 2243.86 | 2252.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 2261.30 | 2243.86 | 2252.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 2225.00 | 2240.09 | 2250.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 2220.60 | 2233.54 | 2245.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 2242.00 | 2223.23 | 2222.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 2242.00 | 2223.23 | 2222.20 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 2217.30 | 2226.17 | 2227.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 2202.00 | 2219.04 | 2223.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 2213.60 | 2210.63 | 2217.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 2213.60 | 2210.63 | 2217.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 2217.10 | 2211.93 | 2217.30 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 2239.30 | 2221.81 | 2220.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 2245.90 | 2226.63 | 2222.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 2232.00 | 2237.90 | 2231.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 10:45:00 | 2236.10 | 2237.90 | 2231.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 2238.10 | 2237.94 | 2232.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:45:00 | 2234.90 | 2237.94 | 2232.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 2251.30 | 2257.61 | 2250.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 2251.30 | 2257.61 | 2250.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 2240.10 | 2254.11 | 2249.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 2240.10 | 2254.11 | 2249.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 2241.00 | 2251.49 | 2248.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:30:00 | 2242.10 | 2251.49 | 2248.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2242.10 | 2249.28 | 2248.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 2239.90 | 2249.28 | 2248.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 2231.70 | 2245.76 | 2247.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 14:15:00 | 2219.90 | 2238.24 | 2243.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 2242.60 | 2238.13 | 2242.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 2242.60 | 2238.13 | 2242.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 2242.60 | 2238.13 | 2242.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 2245.20 | 2238.13 | 2242.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 2260.10 | 2242.53 | 2243.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 2260.10 | 2242.53 | 2243.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 2259.00 | 2245.82 | 2245.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 2265.00 | 2251.88 | 2248.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 10:15:00 | 2265.30 | 2266.07 | 2260.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 10:15:00 | 2265.30 | 2266.07 | 2260.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2265.30 | 2266.07 | 2260.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 2265.30 | 2266.07 | 2260.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 2259.10 | 2264.67 | 2259.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 2259.10 | 2264.67 | 2259.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 2258.50 | 2263.44 | 2259.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:30:00 | 2259.70 | 2263.44 | 2259.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 2258.00 | 2262.35 | 2259.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:45:00 | 2266.30 | 2264.76 | 2260.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 15:15:00 | 2319.20 | 2336.84 | 2337.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 2319.20 | 2336.84 | 2337.12 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 2350.40 | 2339.55 | 2338.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 2392.70 | 2364.63 | 2358.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 14:15:00 | 2404.20 | 2404.67 | 2390.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-01 15:00:00 | 2404.20 | 2404.67 | 2390.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 2401.30 | 2403.25 | 2392.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 2401.10 | 2403.25 | 2392.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 2402.80 | 2402.49 | 2394.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:45:00 | 2401.90 | 2402.49 | 2394.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 2401.70 | 2408.52 | 2404.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 2431.80 | 2417.16 | 2411.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 12:15:00 | 2412.30 | 2413.32 | 2413.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 12:15:00 | 2412.30 | 2413.32 | 2413.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 2379.00 | 2405.38 | 2409.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 2403.70 | 2400.65 | 2406.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 2403.70 | 2400.65 | 2406.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 2403.70 | 2400.65 | 2406.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 2403.70 | 2400.65 | 2406.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 2404.90 | 2401.50 | 2406.34 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 14:15:00 | 2412.00 | 2407.55 | 2407.04 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 2403.40 | 2406.44 | 2406.67 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 2419.60 | 2409.07 | 2407.85 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 11:15:00 | 2405.00 | 2407.36 | 2407.62 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 2410.20 | 2407.92 | 2407.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 13:15:00 | 2414.00 | 2409.14 | 2408.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 2449.30 | 2463.24 | 2448.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 2449.30 | 2463.24 | 2448.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2449.30 | 2463.24 | 2448.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 2446.50 | 2463.24 | 2448.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 2454.70 | 2461.53 | 2448.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 2453.90 | 2461.53 | 2448.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 2451.90 | 2459.60 | 2449.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 2450.70 | 2459.60 | 2449.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 2450.00 | 2457.68 | 2449.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 2456.40 | 2457.68 | 2449.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 2453.10 | 2454.77 | 2449.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 2439.70 | 2450.19 | 2448.58 | SL hit (close<static) qty=1.00 sl=2443.50 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 2427.50 | 2445.65 | 2446.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 2418.40 | 2440.20 | 2444.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 2232.20 | 2230.52 | 2257.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 2232.20 | 2230.52 | 2257.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 2225.00 | 2221.58 | 2235.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 2221.20 | 2221.68 | 2234.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 2242.40 | 2226.80 | 2229.98 | SL hit (close>static) qty=1.00 sl=2237.80 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 2240.70 | 2233.56 | 2232.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 2247.60 | 2236.36 | 2233.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 2246.50 | 2246.55 | 2241.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:30:00 | 2250.00 | 2246.55 | 2241.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 2241.00 | 2247.82 | 2244.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 2241.00 | 2247.82 | 2244.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 2242.50 | 2246.76 | 2244.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 2236.80 | 2246.76 | 2244.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 2242.60 | 2245.93 | 2244.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:45:00 | 2241.10 | 2245.93 | 2244.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 2242.30 | 2245.20 | 2244.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 2241.20 | 2245.20 | 2244.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 2240.20 | 2244.20 | 2243.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 2242.70 | 2244.20 | 2243.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 2253.00 | 2245.96 | 2244.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 2243.60 | 2245.96 | 2244.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 2228.20 | 2242.41 | 2243.05 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 2258.70 | 2236.92 | 2236.67 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 2230.30 | 2235.44 | 2236.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2199.30 | 2225.32 | 2230.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 2216.10 | 2213.72 | 2219.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 2216.10 | 2213.72 | 2219.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 2214.60 | 2213.89 | 2219.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:15:00 | 2224.80 | 2213.89 | 2219.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 2223.10 | 2215.73 | 2219.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 2225.50 | 2215.73 | 2219.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 2223.90 | 2217.37 | 2220.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 2221.50 | 2217.37 | 2220.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2223.10 | 2218.51 | 2220.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 2225.90 | 2218.51 | 2220.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2226.60 | 2220.26 | 2220.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 2223.00 | 2220.26 | 2220.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 2231.90 | 2222.59 | 2221.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 2238.90 | 2227.83 | 2225.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 2247.10 | 2251.34 | 2238.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 14:15:00 | 2247.10 | 2251.34 | 2238.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 2247.10 | 2251.34 | 2238.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 2243.00 | 2251.34 | 2238.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 2189.90 | 2238.84 | 2235.13 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 2176.20 | 2226.31 | 2229.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 09:15:00 | 2150.00 | 2186.46 | 2203.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 13:15:00 | 2102.00 | 2100.97 | 2122.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 14:00:00 | 2102.00 | 2100.97 | 2122.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 2131.30 | 2102.66 | 2112.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 2131.30 | 2102.66 | 2112.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 2127.40 | 2107.60 | 2114.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 2123.20 | 2107.60 | 2114.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 2120.50 | 2117.04 | 2117.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 2135.10 | 2120.65 | 2119.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 2135.10 | 2120.65 | 2119.05 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 2106.70 | 2119.29 | 2120.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 12:15:00 | 2102.20 | 2115.87 | 2118.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 2061.30 | 2056.55 | 2069.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:00:00 | 2061.30 | 2056.55 | 2069.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 2057.00 | 2046.49 | 2053.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 2057.90 | 2046.49 | 2053.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2061.70 | 2049.53 | 2054.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 2061.40 | 2049.53 | 2054.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 2055.00 | 2055.53 | 2055.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 2047.70 | 2055.53 | 2055.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2040.80 | 2052.59 | 2054.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 2039.10 | 2052.59 | 2054.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:45:00 | 2040.50 | 2050.15 | 2053.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:00:00 | 2040.40 | 2048.20 | 2052.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:00:00 | 2040.00 | 2046.56 | 2050.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 2045.50 | 2045.38 | 2049.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 2045.50 | 2045.38 | 2049.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 2030.60 | 2042.30 | 2047.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 2023.60 | 2038.80 | 2045.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:45:00 | 2024.20 | 2034.53 | 2042.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:15:00 | 2024.20 | 2034.53 | 2042.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:00:00 | 2024.20 | 2032.46 | 2040.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 2040.00 | 2035.08 | 2040.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 2045.40 | 2035.08 | 2040.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 2053.50 | 2038.77 | 2041.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 2053.50 | 2038.77 | 2041.61 | SL hit (close>static) qty=1.00 sl=2049.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 2064.30 | 2043.87 | 2043.67 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 13:15:00 | 2029.80 | 2043.95 | 2045.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 14:15:00 | 2014.50 | 2027.53 | 2035.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 2028.70 | 2005.80 | 2014.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 2028.70 | 2005.80 | 2014.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 2028.70 | 2005.80 | 2014.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:45:00 | 2024.90 | 2005.80 | 2014.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 2008.50 | 2006.34 | 2013.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:45:00 | 2004.80 | 2012.29 | 2014.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 2002.40 | 2009.48 | 2012.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 2004.00 | 2009.48 | 2012.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:45:00 | 2004.60 | 2008.94 | 2011.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 2009.30 | 2006.80 | 2009.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:45:00 | 2017.10 | 2006.80 | 2009.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 2011.00 | 2007.64 | 2009.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 15:15:00 | 2001.30 | 2007.64 | 2009.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 1997.20 | 2006.52 | 2008.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 2017.70 | 2008.75 | 2009.39 | SL hit (close>static) qty=1.00 sl=2014.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 2021.00 | 2011.20 | 2010.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 2029.40 | 2018.81 | 2016.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 2008.10 | 2021.16 | 2019.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 2008.10 | 2021.16 | 2019.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 2008.10 | 2021.16 | 2019.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:00:00 | 2008.10 | 2021.16 | 2019.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 2010.30 | 2018.99 | 2018.67 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 2002.70 | 2015.73 | 2017.22 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 2019.50 | 2016.80 | 2016.65 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 2005.60 | 2014.56 | 2015.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 2000.90 | 2009.40 | 2012.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 2013.20 | 2009.86 | 2011.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 2013.20 | 2009.86 | 2011.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 2013.20 | 2009.86 | 2011.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 2013.20 | 2009.86 | 2011.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 2010.00 | 2009.89 | 2011.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 2004.50 | 2009.89 | 2011.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1996.70 | 2007.25 | 2010.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:00:00 | 1992.60 | 2004.18 | 2008.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:00:00 | 1990.00 | 1988.80 | 1996.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:00:00 | 1991.40 | 1988.59 | 1993.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 13:45:00 | 1990.40 | 1990.19 | 1991.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1988.60 | 1989.87 | 1990.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:15:00 | 1995.00 | 1989.87 | 1990.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1995.00 | 1990.90 | 1991.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 2009.30 | 1990.90 | 1991.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 2002.20 | 1993.16 | 1992.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 09:15:00 | 2002.20 | 1993.16 | 1992.19 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1970.90 | 1988.45 | 1990.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 1950.00 | 1964.45 | 1974.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 10:15:00 | 1953.70 | 1951.98 | 1962.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 11:00:00 | 1953.70 | 1951.98 | 1962.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1971.90 | 1955.97 | 1963.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 1971.90 | 1955.97 | 1963.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1977.20 | 1960.21 | 1964.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 1984.90 | 1960.21 | 1964.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1987.60 | 1969.51 | 1968.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 2049.30 | 1987.64 | 1976.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2071.50 | 2083.94 | 2056.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 2071.50 | 2083.94 | 2056.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 2065.10 | 2073.48 | 2059.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:30:00 | 2066.70 | 2071.85 | 2060.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 2070.00 | 2070.74 | 2060.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 2048.00 | 2066.19 | 2059.72 | SL hit (close<static) qty=1.00 sl=2054.10 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 2029.00 | 2059.64 | 2059.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 2018.10 | 2041.54 | 2050.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 2014.90 | 2014.45 | 2023.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 2014.90 | 2014.45 | 2023.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2056.10 | 2021.27 | 2025.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:45:00 | 2047.20 | 2021.27 | 2025.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2049.20 | 2026.85 | 2027.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 2058.80 | 2026.85 | 2027.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 11:15:00 | 2048.30 | 2031.14 | 2029.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 12:15:00 | 2055.60 | 2036.03 | 2031.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 15:15:00 | 2155.20 | 2158.84 | 2138.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:15:00 | 2130.10 | 2158.84 | 2138.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2136.40 | 2154.35 | 2138.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 2140.10 | 2154.35 | 2138.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 2143.80 | 2152.24 | 2139.08 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 2118.80 | 2132.80 | 2134.40 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 2141.30 | 2136.00 | 2135.67 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 2121.50 | 2132.99 | 2134.35 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 2138.90 | 2135.37 | 2135.19 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 2132.20 | 2134.74 | 2134.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 12:15:00 | 2121.20 | 2132.03 | 2133.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 2141.60 | 2133.94 | 2134.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 13:15:00 | 2141.60 | 2133.94 | 2134.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 2141.60 | 2133.94 | 2134.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 2141.60 | 2133.94 | 2134.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 2143.60 | 2135.87 | 2135.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 2150.00 | 2138.70 | 2136.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 2145.90 | 2146.69 | 2141.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:00:00 | 2145.90 | 2146.69 | 2141.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 2134.40 | 2144.23 | 2141.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 2134.40 | 2144.23 | 2141.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2137.80 | 2142.95 | 2140.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 2130.10 | 2142.95 | 2140.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2133.40 | 2140.10 | 2139.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2133.40 | 2140.10 | 2139.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 2127.50 | 2137.58 | 2138.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 2120.00 | 2132.42 | 2136.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 2137.20 | 2127.62 | 2132.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 2137.20 | 2127.62 | 2132.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 2137.20 | 2127.62 | 2132.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 2095.70 | 2126.25 | 2129.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 2096.00 | 2076.49 | 2074.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 2096.00 | 2076.49 | 2074.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 12:15:00 | 2103.20 | 2081.83 | 2077.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 2155.00 | 2161.45 | 2137.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:00:00 | 2155.00 | 2161.45 | 2137.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2158.70 | 2158.50 | 2143.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 2154.70 | 2158.50 | 2143.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 2143.90 | 2160.15 | 2151.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 2200.00 | 2160.15 | 2151.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 2162.80 | 2186.94 | 2187.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 13:15:00 | 2162.80 | 2186.94 | 2187.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 14:15:00 | 2147.20 | 2178.99 | 2184.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 11:15:00 | 2118.00 | 2110.71 | 2133.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 12:00:00 | 2118.00 | 2110.71 | 2133.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 2108.10 | 2109.07 | 2125.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:30:00 | 2098.30 | 2105.68 | 2122.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 11:30:00 | 2099.00 | 2101.23 | 2117.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-01 12:15:00 | 1888.47 | 2059.75 | 2069.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2096.20 | 2052.90 | 2052.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2176.80 | 2146.49 | 2134.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 2192.70 | 2193.38 | 2172.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 2192.70 | 2193.38 | 2172.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2167.10 | 2186.80 | 2172.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 2167.10 | 2186.80 | 2172.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 2162.20 | 2181.88 | 2171.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 2162.20 | 2181.88 | 2171.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 2173.00 | 2180.10 | 2171.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 2178.30 | 2179.00 | 2172.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:30:00 | 2184.80 | 2179.96 | 2173.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 2157.00 | 2172.79 | 2171.39 | SL hit (close<static) qty=1.00 sl=2162.20 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 2159.00 | 2170.04 | 2170.26 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 2171.60 | 2170.41 | 2170.29 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 2126.20 | 2161.82 | 2166.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 13:15:00 | 2116.10 | 2130.00 | 2139.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 10:15:00 | 2124.80 | 2121.87 | 2131.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:45:00 | 2125.00 | 2121.87 | 2131.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 2131.90 | 2124.31 | 2131.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 2131.90 | 2124.31 | 2131.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 2133.20 | 2126.09 | 2131.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 2133.20 | 2126.09 | 2131.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2125.10 | 2125.89 | 2130.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 2116.00 | 2123.63 | 2128.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 2010.20 | 2052.58 | 2061.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 1968.90 | 1964.20 | 1993.24 | SL hit (close>ema200) qty=0.50 sl=1964.20 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 1890.00 | 1867.78 | 1866.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 1893.30 | 1872.88 | 1868.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 1846.00 | 1876.35 | 1872.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 1846.00 | 1876.35 | 1872.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1846.00 | 1876.35 | 1872.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 1846.00 | 1876.35 | 1872.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 1860.50 | 1873.18 | 1871.20 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 1840.80 | 1865.14 | 1867.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 1836.80 | 1859.47 | 1864.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1852.00 | 1837.60 | 1847.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1852.00 | 1837.60 | 1847.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1852.00 | 1837.60 | 1847.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 1855.60 | 1837.60 | 1847.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1833.10 | 1836.70 | 1846.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 1853.20 | 1836.70 | 1846.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1856.50 | 1840.66 | 1847.49 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 1857.10 | 1850.89 | 1850.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 1870.00 | 1854.71 | 1852.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1865.40 | 1890.68 | 1876.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1865.40 | 1890.68 | 1876.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1865.40 | 1890.68 | 1876.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1865.40 | 1890.68 | 1876.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1846.80 | 1881.91 | 1873.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1846.80 | 1881.91 | 1873.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1847.60 | 1866.40 | 1868.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1830.90 | 1859.30 | 1864.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 1857.60 | 1851.99 | 1857.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 12:15:00 | 1857.60 | 1851.99 | 1857.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 1857.60 | 1851.99 | 1857.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 1855.60 | 1851.99 | 1857.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 1856.40 | 1852.88 | 1857.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 1860.90 | 1852.88 | 1857.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1841.90 | 1850.68 | 1856.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 1830.00 | 1850.68 | 1856.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1738.50 | 1786.92 | 1818.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 1779.40 | 1763.12 | 1789.29 | SL hit (close>ema200) qty=0.50 sl=1763.12 alert=retest2 |

### Cycle 214 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1869.80 | 1811.89 | 1804.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 14:15:00 | 1890.90 | 1848.54 | 1826.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1837.00 | 1852.87 | 1832.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 1837.00 | 1852.87 | 1832.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1835.80 | 1849.45 | 1832.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 1831.60 | 1849.45 | 1832.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1842.10 | 1847.98 | 1833.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:30:00 | 1828.10 | 1847.98 | 1833.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1812.20 | 1842.05 | 1836.57 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1794.80 | 1832.60 | 1832.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 11:15:00 | 1786.00 | 1823.28 | 1828.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1811.20 | 1799.22 | 1812.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1811.20 | 1799.22 | 1812.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1811.20 | 1799.22 | 1812.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1802.80 | 1799.22 | 1812.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 1801.80 | 1805.63 | 1811.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 1799.60 | 1805.49 | 1811.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1767.10 | 1806.35 | 1810.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1797.30 | 1786.27 | 1796.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 1797.30 | 1786.27 | 1796.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 1799.50 | 1788.92 | 1797.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 1799.50 | 1788.92 | 1797.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1780.00 | 1787.14 | 1795.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 1806.20 | 1787.14 | 1795.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1827.90 | 1795.29 | 1798.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 1833.40 | 1795.29 | 1798.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 1825.40 | 1801.31 | 1800.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 1825.40 | 1801.31 | 1800.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1850.60 | 1817.82 | 1808.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1841.80 | 1847.00 | 1828.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 10:00:00 | 1841.80 | 1847.00 | 1828.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 1828.70 | 1843.34 | 1828.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 1828.70 | 1843.34 | 1828.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 1828.80 | 1840.43 | 1828.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:15:00 | 1827.10 | 1840.43 | 1828.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1832.40 | 1838.83 | 1828.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1915.80 | 1835.92 | 1829.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 1962.30 | 1983.05 | 1984.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 1962.30 | 1983.05 | 1984.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 1955.60 | 1977.56 | 1981.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 1953.70 | 1947.23 | 1960.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 1953.70 | 1947.23 | 1960.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1961.30 | 1950.05 | 1960.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:45:00 | 1961.80 | 1950.05 | 1960.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1951.00 | 1950.24 | 1959.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 1980.30 | 1950.24 | 1959.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1976.70 | 1955.53 | 1961.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 1984.00 | 1955.53 | 1961.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1978.20 | 1965.20 | 1964.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 1980.40 | 1968.24 | 1966.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 1942.70 | 1967.24 | 1966.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 1942.70 | 1967.24 | 1966.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1942.70 | 1967.24 | 1966.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:30:00 | 1934.20 | 1967.24 | 1966.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 1934.80 | 1960.76 | 1963.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 11:15:00 | 1927.10 | 1954.02 | 1960.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 1940.90 | 1940.86 | 1950.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 1940.90 | 1940.86 | 1950.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1940.90 | 1940.86 | 1950.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:15:00 | 1964.00 | 1940.86 | 1950.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1964.30 | 1945.55 | 1951.55 | EMA400 retest candle locked (from downside) |

### Cycle 220 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 1975.70 | 1955.78 | 1955.42 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 1950.00 | 1954.62 | 1954.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 1944.80 | 1952.66 | 1954.00 | Break + close below crossover candle low |

### Cycle 222 — BUY (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 15:15:00 | 1964.80 | 1955.09 | 1954.98 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1917.10 | 1947.49 | 1951.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 11:15:00 | 1908.70 | 1934.93 | 1944.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1962.80 | 1925.52 | 1934.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1962.80 | 1925.52 | 1934.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1962.80 | 1925.52 | 1934.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 1962.80 | 1925.52 | 1934.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1989.00 | 1938.22 | 1939.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 1989.00 | 1938.22 | 1939.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1987.10 | 1948.00 | 1943.68 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 13:15:00 | 1949.40 | 1960.43 | 1960.96 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1977.10 | 1963.76 | 1962.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 1980.00 | 1967.01 | 1964.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 1972.90 | 1972.97 | 1968.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 1972.90 | 1972.97 | 1968.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 227 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1918.00 | 1961.64 | 1964.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 1882.30 | 1945.78 | 1956.74 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 11:45:00 | 2061.05 | 2023-05-22 14:15:00 | 2078.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-06-02 09:15:00 | 2149.05 | 2023-06-08 11:15:00 | 2125.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2023-06-02 10:00:00 | 2141.90 | 2023-06-08 11:15:00 | 2125.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-06-05 09:15:00 | 2147.60 | 2023-06-08 11:15:00 | 2125.20 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-06-05 10:30:00 | 2142.40 | 2023-06-08 11:15:00 | 2125.20 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-06-06 12:30:00 | 2165.70 | 2023-06-08 11:15:00 | 2125.20 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2023-06-06 15:15:00 | 2165.90 | 2023-06-08 11:15:00 | 2125.20 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2023-06-07 13:45:00 | 2165.95 | 2023-06-08 11:15:00 | 2125.20 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2023-06-07 14:45:00 | 2168.15 | 2023-06-08 11:15:00 | 2125.20 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2023-06-21 09:15:00 | 2253.60 | 2023-06-22 12:15:00 | 2186.05 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2023-06-22 10:00:00 | 2259.00 | 2023-06-22 12:15:00 | 2186.05 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2023-06-27 13:30:00 | 2170.00 | 2023-06-28 11:15:00 | 2200.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2023-07-17 13:30:00 | 2047.50 | 2023-07-21 09:15:00 | 1945.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-18 09:45:00 | 2048.10 | 2023-07-21 09:15:00 | 1945.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-18 10:30:00 | 2047.40 | 2023-07-21 09:15:00 | 1945.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-17 13:30:00 | 2047.50 | 2023-07-25 14:15:00 | 1908.25 | STOP_HIT | 0.50 | 6.80% |
| SELL | retest2 | 2023-07-18 09:45:00 | 2048.10 | 2023-07-25 14:15:00 | 1908.25 | STOP_HIT | 0.50 | 6.83% |
| SELL | retest2 | 2023-07-18 10:30:00 | 2047.40 | 2023-07-25 14:15:00 | 1908.25 | STOP_HIT | 0.50 | 6.80% |
| BUY | retest2 | 2023-08-03 11:15:00 | 1953.60 | 2023-08-03 11:15:00 | 1957.05 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2023-08-07 12:15:00 | 1995.50 | 2023-08-10 12:15:00 | 1992.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2023-08-07 12:45:00 | 1997.00 | 2023-08-10 12:15:00 | 1992.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2023-08-08 11:30:00 | 1994.90 | 2023-08-10 12:15:00 | 1992.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2023-08-11 15:15:00 | 1980.00 | 2023-08-16 09:15:00 | 1881.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-11 15:15:00 | 1980.00 | 2023-08-17 14:15:00 | 1907.60 | STOP_HIT | 0.50 | 3.66% |
| BUY | retest2 | 2023-08-28 11:00:00 | 2015.95 | 2023-09-04 11:15:00 | 2217.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-29 12:00:00 | 2392.20 | 2023-10-04 09:15:00 | 2325.00 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2023-10-16 09:15:00 | 2300.45 | 2023-10-17 09:15:00 | 2244.95 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2023-10-16 11:15:00 | 2297.30 | 2023-10-17 09:15:00 | 2244.95 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2023-10-30 15:00:00 | 2043.95 | 2023-10-31 09:15:00 | 2072.15 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-11-02 09:45:00 | 2097.10 | 2023-11-02 11:15:00 | 2073.75 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2023-11-02 10:15:00 | 2092.00 | 2023-11-02 11:15:00 | 2073.75 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-11-07 13:30:00 | 2089.55 | 2023-11-09 15:15:00 | 2090.30 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2023-11-10 14:15:00 | 2084.90 | 2023-11-12 18:15:00 | 2106.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-11-16 13:30:00 | 2141.70 | 2023-11-22 11:15:00 | 2160.00 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2023-12-06 14:15:00 | 2351.40 | 2023-12-08 13:15:00 | 2310.50 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2023-12-13 14:30:00 | 2363.00 | 2023-12-15 15:15:00 | 2366.00 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2023-12-15 15:15:00 | 2366.00 | 2023-12-15 15:15:00 | 2366.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-01-02 15:15:00 | 2325.00 | 2024-01-03 10:15:00 | 2311.05 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-01-03 09:45:00 | 2325.75 | 2024-01-03 10:15:00 | 2311.05 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-01-19 10:30:00 | 2188.80 | 2024-01-19 15:15:00 | 2219.85 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-01-19 11:45:00 | 2185.70 | 2024-01-19 15:15:00 | 2219.85 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-01-19 13:30:00 | 2192.10 | 2024-01-19 15:15:00 | 2219.85 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest1 | 2024-02-01 09:15:00 | 2284.90 | 2024-02-02 12:15:00 | 2270.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-02-01 10:00:00 | 2286.65 | 2024-02-02 12:15:00 | 2270.80 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2024-02-01 11:45:00 | 2285.05 | 2024-02-02 12:15:00 | 2270.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-02-01 12:45:00 | 2289.10 | 2024-02-02 12:15:00 | 2270.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-02-07 10:30:00 | 2180.00 | 2024-02-09 10:15:00 | 2077.51 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2024-02-07 13:00:00 | 2186.85 | 2024-02-09 11:15:00 | 2071.00 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2024-02-07 14:30:00 | 2180.95 | 2024-02-09 11:15:00 | 2071.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 10:30:00 | 2180.00 | 2024-02-09 13:15:00 | 2121.75 | STOP_HIT | 0.50 | 2.67% |
| SELL | retest2 | 2024-02-07 13:00:00 | 2186.85 | 2024-02-09 13:15:00 | 2121.75 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2024-02-07 14:30:00 | 2180.95 | 2024-02-09 13:15:00 | 2121.75 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2024-03-11 10:15:00 | 1957.90 | 2024-03-13 09:15:00 | 1860.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 11:45:00 | 1948.45 | 2024-03-13 09:15:00 | 1859.77 | PARTIAL | 0.50 | 4.55% |
| SELL | retest2 | 2024-03-11 13:15:00 | 1957.65 | 2024-03-13 09:15:00 | 1858.34 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2024-03-11 13:45:00 | 1956.15 | 2024-03-13 11:15:00 | 1851.03 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2024-03-11 10:15:00 | 1957.90 | 2024-03-14 09:15:00 | 1865.30 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2024-03-11 11:45:00 | 1948.45 | 2024-03-14 09:15:00 | 1865.30 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2024-03-11 13:15:00 | 1957.65 | 2024-03-14 09:15:00 | 1865.30 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2024-03-11 13:45:00 | 1956.15 | 2024-03-14 09:15:00 | 1865.30 | STOP_HIT | 0.50 | 4.64% |
| BUY | retest2 | 2024-03-19 11:45:00 | 1885.00 | 2024-03-20 09:15:00 | 1852.60 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2024-03-26 14:15:00 | 1967.70 | 2024-03-28 13:15:00 | 1955.25 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2024-03-27 09:15:00 | 1975.00 | 2024-03-28 13:15:00 | 1955.25 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-03-28 11:30:00 | 1970.80 | 2024-04-05 11:15:00 | 1990.00 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2024-04-01 09:15:00 | 1987.75 | 2024-04-05 11:15:00 | 1990.00 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-04-02 14:00:00 | 1971.25 | 2024-04-05 11:15:00 | 1990.00 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2024-04-19 14:30:00 | 1942.85 | 2024-04-22 10:15:00 | 1960.95 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-05-02 10:15:00 | 1799.15 | 2024-05-10 12:15:00 | 1709.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 12:45:00 | 1798.15 | 2024-05-10 13:15:00 | 1708.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 10:15:00 | 1797.80 | 2024-05-10 13:15:00 | 1707.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 09:30:00 | 1796.65 | 2024-05-10 13:15:00 | 1706.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 09:15:00 | 1790.55 | 2024-05-13 10:15:00 | 1701.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 10:15:00 | 1799.15 | 2024-05-13 11:15:00 | 1727.20 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2024-05-02 12:45:00 | 1798.15 | 2024-05-13 11:15:00 | 1727.20 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2024-05-03 10:15:00 | 1797.80 | 2024-05-13 11:15:00 | 1727.20 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2024-05-06 09:30:00 | 1796.65 | 2024-05-13 11:15:00 | 1727.20 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2024-05-07 09:15:00 | 1790.55 | 2024-05-13 11:15:00 | 1727.20 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2024-05-28 11:30:00 | 1797.50 | 2024-06-03 09:15:00 | 1829.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-05-28 12:15:00 | 1793.80 | 2024-06-03 09:15:00 | 1829.45 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-05-28 13:00:00 | 1797.50 | 2024-06-03 09:15:00 | 1829.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-05-28 14:15:00 | 1796.95 | 2024-06-03 09:15:00 | 1829.45 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-05-29 12:15:00 | 1776.30 | 2024-06-03 09:15:00 | 1829.45 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-05-29 14:00:00 | 1780.00 | 2024-06-03 09:15:00 | 1829.45 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-06-21 10:30:00 | 1826.45 | 2024-07-01 10:15:00 | 1832.95 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2024-06-26 10:30:00 | 1831.65 | 2024-07-01 10:15:00 | 1832.95 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-07-01 10:15:00 | 1832.65 | 2024-07-01 10:15:00 | 1832.95 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-07-03 12:30:00 | 1841.90 | 2024-07-10 10:15:00 | 1831.70 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-07-04 10:30:00 | 1843.75 | 2024-07-10 10:15:00 | 1831.70 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-07-04 11:00:00 | 1840.50 | 2024-07-10 10:15:00 | 1831.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-07-05 09:30:00 | 1847.00 | 2024-07-10 10:15:00 | 1831.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-05 12:15:00 | 1860.40 | 2024-07-10 10:15:00 | 1831.70 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-07-16 15:15:00 | 1918.50 | 2024-07-18 10:15:00 | 1895.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-07-18 12:00:00 | 1923.00 | 2024-07-19 09:15:00 | 1847.75 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2024-07-18 13:30:00 | 1917.70 | 2024-07-19 09:15:00 | 1847.75 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2024-07-18 15:00:00 | 1918.05 | 2024-07-19 09:15:00 | 1847.75 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2024-08-06 11:45:00 | 1766.00 | 2024-08-08 15:15:00 | 1770.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-08-08 13:00:00 | 1770.00 | 2024-08-08 15:15:00 | 1770.05 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2024-08-08 13:45:00 | 1770.00 | 2024-08-08 15:15:00 | 1770.05 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2024-08-16 12:15:00 | 1727.50 | 2024-08-16 14:15:00 | 1753.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-08-16 13:00:00 | 1729.65 | 2024-08-16 14:15:00 | 1753.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-08-22 09:15:00 | 1778.95 | 2024-09-06 09:15:00 | 1875.50 | STOP_HIT | 1.00 | 5.43% |
| BUY | retest2 | 2024-08-22 10:45:00 | 1782.40 | 2024-09-06 09:15:00 | 1875.50 | STOP_HIT | 1.00 | 5.22% |
| SELL | retest2 | 2024-09-18 11:45:00 | 1837.95 | 2024-09-23 09:15:00 | 1865.55 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-09-20 13:15:00 | 1842.25 | 2024-09-23 09:15:00 | 1865.55 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-09-30 10:00:00 | 1926.70 | 2024-10-07 09:15:00 | 1900.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-10-08 12:30:00 | 1879.15 | 2024-10-14 10:15:00 | 1878.50 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-10-08 13:45:00 | 1882.20 | 2024-10-14 10:15:00 | 1878.50 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-10-09 09:15:00 | 1883.00 | 2024-10-14 10:15:00 | 1878.50 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2024-10-09 11:30:00 | 1873.60 | 2024-10-14 10:15:00 | 1878.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-10-18 14:30:00 | 1845.00 | 2024-10-24 09:15:00 | 1752.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 14:30:00 | 1845.00 | 2024-10-24 09:15:00 | 1817.60 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2024-10-18 15:15:00 | 1840.00 | 2024-10-30 09:15:00 | 1822.10 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2024-11-27 11:30:00 | 1825.75 | 2024-11-29 12:15:00 | 1813.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-11-27 12:15:00 | 1825.85 | 2024-11-29 12:15:00 | 1813.50 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-11-27 13:00:00 | 1834.85 | 2024-11-29 12:15:00 | 1813.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-11-28 11:30:00 | 1829.35 | 2024-11-29 12:15:00 | 1813.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-11-29 09:15:00 | 1828.00 | 2024-11-29 12:15:00 | 1813.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-11-29 09:45:00 | 1829.60 | 2024-11-29 12:15:00 | 1813.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-12-27 15:15:00 | 1735.05 | 2024-12-30 09:15:00 | 1772.80 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-01-02 10:45:00 | 1794.45 | 2025-01-06 12:15:00 | 1771.15 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-01-06 11:15:00 | 1784.30 | 2025-01-06 12:15:00 | 1771.15 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-01-27 13:00:00 | 1808.65 | 2025-01-28 09:15:00 | 1787.65 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-01-29 10:15:00 | 1857.10 | 2025-02-01 13:15:00 | 1817.65 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-01-30 14:45:00 | 1848.95 | 2025-02-01 13:15:00 | 1817.65 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-01-31 09:15:00 | 1853.10 | 2025-02-01 13:15:00 | 1817.65 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-02-14 10:30:00 | 1769.85 | 2025-02-17 14:15:00 | 1808.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-02-14 12:00:00 | 1769.75 | 2025-02-17 14:15:00 | 1808.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-02-17 09:15:00 | 1766.15 | 2025-02-17 14:15:00 | 1808.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-02-17 09:45:00 | 1774.25 | 2025-02-17 14:15:00 | 1808.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-02-20 12:15:00 | 1781.75 | 2025-02-27 09:15:00 | 1692.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-20 12:15:00 | 1781.75 | 2025-02-27 12:15:00 | 1745.95 | STOP_HIT | 0.50 | 2.01% |
| SELL | retest2 | 2025-02-21 09:30:00 | 1769.00 | 2025-02-28 10:15:00 | 1680.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 09:30:00 | 1769.00 | 2025-02-28 15:15:00 | 1702.50 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-03-10 11:45:00 | 1683.85 | 2025-03-17 14:15:00 | 1665.25 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2025-03-11 15:00:00 | 1685.00 | 2025-03-17 14:15:00 | 1665.25 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-03-12 09:15:00 | 1668.30 | 2025-03-17 14:15:00 | 1665.25 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-03-21 10:45:00 | 1710.30 | 2025-04-03 10:15:00 | 1881.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 12:30:00 | 1710.50 | 2025-04-03 10:15:00 | 1881.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 13:00:00 | 1716.75 | 2025-04-07 09:15:00 | 1784.35 | STOP_HIT | 1.00 | 3.94% |
| BUY | retest2 | 2025-04-17 12:15:00 | 1860.60 | 2025-04-23 13:15:00 | 1876.30 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-04-21 09:15:00 | 1861.90 | 2025-04-23 13:15:00 | 1876.30 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-05-29 10:00:00 | 2040.90 | 2025-06-02 12:15:00 | 2070.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-05-29 11:30:00 | 2041.80 | 2025-06-02 12:15:00 | 2070.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-05-30 09:30:00 | 2032.20 | 2025-06-02 12:15:00 | 2070.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-06-05 09:15:00 | 2074.90 | 2025-06-12 13:15:00 | 2100.80 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2025-06-17 11:45:00 | 2081.00 | 2025-06-23 13:15:00 | 2075.30 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-06-25 09:45:00 | 2096.60 | 2025-07-04 14:15:00 | 2179.40 | STOP_HIT | 1.00 | 3.95% |
| SELL | retest2 | 2025-07-28 12:30:00 | 2220.60 | 2025-07-30 13:15:00 | 2242.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-08-14 14:45:00 | 2266.30 | 2025-08-22 15:15:00 | 2319.20 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2025-09-08 09:30:00 | 2431.80 | 2025-09-09 12:15:00 | 2412.30 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-18 13:15:00 | 2456.40 | 2025-09-19 10:15:00 | 2439.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-09-19 09:15:00 | 2453.10 | 2025-09-19 10:15:00 | 2439.70 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-10-01 09:45:00 | 2221.20 | 2025-10-03 09:15:00 | 2242.40 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-10-28 14:15:00 | 2123.20 | 2025-10-29 10:15:00 | 2135.10 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-10-29 09:45:00 | 2120.50 | 2025-10-29 10:15:00 | 2135.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-11-11 10:15:00 | 2039.10 | 2025-11-13 09:15:00 | 2053.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-11-11 10:45:00 | 2040.50 | 2025-11-13 09:15:00 | 2053.50 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-11 12:00:00 | 2040.40 | 2025-11-13 09:15:00 | 2053.50 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-11 13:00:00 | 2040.00 | 2025-11-13 09:15:00 | 2053.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-12 10:45:00 | 2023.60 | 2025-11-13 10:15:00 | 2064.30 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-11-12 12:45:00 | 2024.20 | 2025-11-13 10:15:00 | 2064.30 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-11-12 13:15:00 | 2024.20 | 2025-11-13 10:15:00 | 2064.30 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-11-12 14:00:00 | 2024.20 | 2025-11-13 10:15:00 | 2064.30 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-11-20 09:45:00 | 2004.80 | 2025-11-24 10:15:00 | 2017.70 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-20 12:30:00 | 2002.40 | 2025-11-24 10:15:00 | 2017.70 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-20 13:15:00 | 2004.00 | 2025-11-24 11:15:00 | 2021.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-21 09:45:00 | 2004.60 | 2025-11-24 11:15:00 | 2021.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-21 15:15:00 | 2001.30 | 2025-11-24 11:15:00 | 2021.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-11-24 10:15:00 | 1997.20 | 2025-11-24 11:15:00 | 2021.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-02 13:00:00 | 1992.60 | 2025-12-08 09:15:00 | 2002.20 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-03 13:00:00 | 1990.00 | 2025-12-08 09:15:00 | 2002.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-12-04 11:00:00 | 1991.40 | 2025-12-08 09:15:00 | 2002.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-05 13:45:00 | 1990.40 | 2025-12-08 09:15:00 | 2002.20 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-16 14:30:00 | 2066.70 | 2025-12-17 09:15:00 | 2048.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-12-17 09:15:00 | 2070.00 | 2025-12-17 09:15:00 | 2048.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-17 12:15:00 | 2068.50 | 2025-12-18 09:15:00 | 2029.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-12-17 12:45:00 | 2070.80 | 2025-12-18 09:15:00 | 2029.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-01-08 09:15:00 | 2095.70 | 2026-01-13 11:15:00 | 2096.00 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2026-01-20 09:15:00 | 2200.00 | 2026-01-22 13:15:00 | 2162.80 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-01-28 09:30:00 | 2098.30 | 2026-02-01 12:15:00 | 1888.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-28 11:30:00 | 2099.00 | 2026-02-01 12:15:00 | 1889.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-11 12:45:00 | 2178.30 | 2026-02-12 09:15:00 | 2157.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-02-11 13:30:00 | 2184.80 | 2026-02-12 09:15:00 | 2157.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-02-19 11:30:00 | 2116.00 | 2026-02-27 09:15:00 | 2010.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 2116.00 | 2026-03-02 15:15:00 | 1968.90 | STOP_HIT | 0.50 | 6.95% |
| SELL | retest2 | 2026-03-20 15:15:00 | 1830.00 | 2026-03-23 13:15:00 | 1738.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 15:15:00 | 1830.00 | 2026-03-24 12:15:00 | 1779.40 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1802.80 | 2026-04-06 10:15:00 | 1825.40 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-04-01 13:30:00 | 1801.80 | 2026-04-06 10:15:00 | 1825.40 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-04-01 14:30:00 | 1799.60 | 2026-04-06 10:15:00 | 1825.40 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1767.10 | 2026-04-06 10:15:00 | 1825.40 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1915.80 | 2026-04-23 10:15:00 | 1962.30 | STOP_HIT | 1.00 | 2.43% |
