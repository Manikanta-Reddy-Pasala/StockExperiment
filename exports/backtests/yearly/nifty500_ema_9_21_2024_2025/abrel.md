# Aditya Birla Real Estate Ltd. (ABREL)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1479.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 133 |
| ALERT1 | 92 |
| ALERT2 | 91 |
| ALERT2_SKIP | 47 |
| ALERT3 | 239 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 102 |
| PARTIAL | 20 |
| TARGET_HIT | 4 |
| STOP_HIT | 98 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 122 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 77
- **Target hits / Stop hits / Partials:** 4 / 98 / 20
- **Avg / median % per leg:** 0.40% / -0.82%
- **Sum % (uncompounded):** 48.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 5 | 11.6% | 4 | 39 | 0 | -0.45% | -19.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 43 | 5 | 11.6% | 4 | 39 | 0 | -0.45% | -19.4% |
| SELL (all) | 79 | 40 | 50.6% | 0 | 59 | 20 | 0.86% | 67.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 79 | 40 | 50.6% | 0 | 59 | 20 | 0.86% | 67.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 122 | 45 | 36.9% | 4 | 98 | 20 | 0.40% | 48.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 1920.80 | 1968.19 | 1974.31 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1998.45 | 1970.71 | 1969.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 2056.70 | 2005.39 | 1989.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 2045.10 | 2049.57 | 2021.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 14:45:00 | 2055.00 | 2049.57 | 2021.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 2160.50 | 2165.18 | 2144.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 2148.65 | 2165.18 | 2144.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 2139.95 | 2160.13 | 2143.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 2139.95 | 2160.13 | 2143.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 2152.50 | 2158.61 | 2144.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:00:00 | 2156.95 | 2158.28 | 2145.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 2157.60 | 2155.61 | 2147.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 10:15:00 | 2154.50 | 2154.97 | 2148.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 10:15:00 | 2120.40 | 2148.06 | 2145.66 | SL hit (close<static) qty=1.00 sl=2135.05 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 2125.00 | 2143.44 | 2143.78 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 2172.75 | 2146.75 | 2144.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 11:15:00 | 2203.45 | 2166.03 | 2154.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 2178.90 | 2195.54 | 2176.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 2178.90 | 2195.54 | 2176.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 2178.90 | 2195.54 | 2176.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 2170.30 | 2195.54 | 2176.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 2173.40 | 2191.12 | 2175.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:30:00 | 2173.95 | 2191.12 | 2175.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 2165.25 | 2185.94 | 2174.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:30:00 | 2158.00 | 2185.94 | 2174.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 2168.35 | 2182.42 | 2174.34 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 10:15:00 | 2146.50 | 2166.31 | 2168.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 2125.45 | 2158.14 | 2164.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 2158.35 | 2152.51 | 2160.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 14:15:00 | 2158.35 | 2152.51 | 2160.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 2158.35 | 2152.51 | 2160.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:45:00 | 2164.00 | 2152.51 | 2160.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 2176.65 | 2156.94 | 2160.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:45:00 | 2188.40 | 2156.94 | 2160.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 2167.20 | 2158.99 | 2161.37 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 2170.00 | 2162.94 | 2162.86 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 2098.00 | 2150.20 | 2157.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 10:15:00 | 2092.00 | 2138.56 | 2151.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 2070.45 | 2061.66 | 2088.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 2070.45 | 2061.66 | 2088.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2070.45 | 2061.66 | 2088.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:15:00 | 2109.95 | 2061.66 | 2088.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 2090.85 | 2067.50 | 2088.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:30:00 | 2116.00 | 2067.50 | 2088.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 2087.15 | 2071.43 | 2088.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 11:45:00 | 2093.90 | 2071.43 | 2088.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 2106.35 | 2078.41 | 2090.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:30:00 | 2107.10 | 2078.41 | 2090.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 13:15:00 | 2108.80 | 2084.49 | 2092.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 13:30:00 | 2108.85 | 2084.49 | 2092.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 2076.35 | 1896.17 | 1929.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 2076.35 | 1896.17 | 1929.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 2089.00 | 1934.73 | 1943.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 2089.00 | 1934.73 | 1943.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 2059.15 | 1959.62 | 1954.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 2105.75 | 2003.29 | 1975.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 2093.60 | 2101.34 | 2065.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 2093.60 | 2101.34 | 2065.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 2085.35 | 2094.87 | 2075.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 2067.70 | 2094.87 | 2075.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 2083.10 | 2092.52 | 2076.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:30:00 | 2060.90 | 2092.52 | 2076.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 2137.90 | 2101.59 | 2082.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:30:00 | 2146.30 | 2114.94 | 2102.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 12:15:00 | 2360.93 | 2268.12 | 2218.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 2254.90 | 2273.39 | 2273.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 2228.10 | 2264.33 | 2269.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 2244.40 | 2236.38 | 2250.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 11:15:00 | 2244.40 | 2236.38 | 2250.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 2244.40 | 2236.38 | 2250.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 2247.35 | 2236.38 | 2250.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 2317.00 | 2252.50 | 2256.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:00:00 | 2317.00 | 2252.50 | 2256.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 2317.00 | 2265.40 | 2262.29 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 2255.50 | 2279.01 | 2280.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 2242.10 | 2264.81 | 2272.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 2275.85 | 2256.91 | 2265.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 2275.85 | 2256.91 | 2265.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 2275.85 | 2256.91 | 2265.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 2275.85 | 2256.91 | 2265.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 2280.00 | 2261.53 | 2267.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 2280.15 | 2261.53 | 2267.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 2294.65 | 2270.89 | 2270.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 13:15:00 | 2324.90 | 2287.86 | 2278.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 15:15:00 | 2290.10 | 2293.86 | 2283.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 09:15:00 | 2287.00 | 2293.86 | 2283.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 2306.95 | 2296.48 | 2285.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 10:15:00 | 2323.00 | 2296.48 | 2285.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 09:30:00 | 2320.00 | 2313.02 | 2299.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 14:15:00 | 2320.00 | 2344.61 | 2345.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 2320.00 | 2344.61 | 2345.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 12:15:00 | 2291.95 | 2330.78 | 2338.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 11:15:00 | 2251.55 | 2246.58 | 2271.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 11:30:00 | 2259.65 | 2246.58 | 2271.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 2254.90 | 2248.24 | 2269.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 15:00:00 | 2236.35 | 2247.66 | 2265.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 2203.00 | 2235.72 | 2257.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 11:15:00 | 2282.10 | 2244.08 | 2257.17 | SL hit (close>static) qty=1.00 sl=2269.85 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 12:15:00 | 2184.60 | 2167.68 | 2166.79 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 2157.05 | 2165.55 | 2165.90 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 14:15:00 | 2211.90 | 2174.82 | 2170.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 12:15:00 | 2273.35 | 2214.01 | 2192.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 2198.15 | 2227.89 | 2207.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 2198.15 | 2227.89 | 2207.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 2198.15 | 2227.89 | 2207.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 2198.15 | 2227.89 | 2207.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 2198.40 | 2221.99 | 2207.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:15:00 | 2212.95 | 2221.99 | 2207.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 2217.15 | 2221.02 | 2207.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 13:15:00 | 2231.80 | 2220.36 | 2208.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 2192.35 | 2213.74 | 2207.77 | SL hit (close<static) qty=1.00 sl=2195.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 12:15:00 | 2167.25 | 2200.04 | 2202.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 13:15:00 | 2159.70 | 2191.97 | 2198.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 2193.55 | 2192.29 | 2198.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 14:15:00 | 2193.55 | 2192.29 | 2198.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 2193.55 | 2192.29 | 2198.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 2193.55 | 2192.29 | 2198.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 2198.00 | 2193.43 | 2198.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 2179.35 | 2193.43 | 2198.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 2187.45 | 2192.24 | 2197.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 2124.45 | 2192.34 | 2196.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:15:00 | 2160.00 | 2172.24 | 2184.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 2225.55 | 2180.95 | 2186.34 | SL hit (close>static) qty=1.00 sl=2206.05 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 2306.00 | 2205.96 | 2197.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 2356.00 | 2301.62 | 2269.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 2333.00 | 2344.50 | 2319.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 13:00:00 | 2333.00 | 2344.50 | 2319.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 2350.00 | 2343.93 | 2323.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:45:00 | 2310.85 | 2343.93 | 2323.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 2335.80 | 2341.72 | 2326.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:00:00 | 2370.85 | 2347.83 | 2337.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 13:15:00 | 2366.50 | 2350.56 | 2339.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 10:00:00 | 2368.90 | 2351.17 | 2342.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 13:15:00 | 2310.50 | 2339.59 | 2339.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 2310.50 | 2339.59 | 2339.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 2307.30 | 2333.13 | 2337.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2211.50 | 2190.01 | 2229.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 2211.50 | 2190.01 | 2229.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 2211.50 | 2190.01 | 2229.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 2248.50 | 2190.01 | 2229.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 2176.95 | 2171.62 | 2200.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:15:00 | 2151.00 | 2166.70 | 2188.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 11:15:00 | 2193.15 | 2186.51 | 2185.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 2193.15 | 2186.51 | 2185.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 12:15:00 | 2201.40 | 2189.49 | 2187.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 2187.95 | 2193.37 | 2189.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 14:15:00 | 2187.95 | 2193.37 | 2189.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 2187.95 | 2193.37 | 2189.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 2187.95 | 2193.37 | 2189.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 2195.00 | 2193.69 | 2190.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 2219.35 | 2193.69 | 2190.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:15:00 | 2207.90 | 2193.54 | 2190.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 15:00:00 | 2201.00 | 2203.96 | 2201.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 09:15:00 | 2126.30 | 2187.16 | 2194.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 2126.30 | 2187.16 | 2194.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 2119.30 | 2173.58 | 2187.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 2131.05 | 2123.70 | 2141.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 15:00:00 | 2131.05 | 2123.70 | 2141.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 2137.95 | 2126.60 | 2139.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 14:30:00 | 2123.00 | 2131.90 | 2138.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 15:15:00 | 2149.95 | 2135.51 | 2139.15 | SL hit (close>static) qty=1.00 sl=2148.30 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 12:15:00 | 2151.95 | 2142.30 | 2141.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 15:15:00 | 2159.00 | 2147.93 | 2144.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 10:15:00 | 2339.35 | 2341.02 | 2316.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 10:45:00 | 2334.75 | 2341.02 | 2316.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 2450.65 | 2479.94 | 2451.24 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 09:15:00 | 2426.00 | 2441.09 | 2441.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 10:15:00 | 2412.05 | 2435.28 | 2439.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 2424.55 | 2407.91 | 2420.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 2424.55 | 2407.91 | 2420.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 2424.55 | 2407.91 | 2420.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 12:00:00 | 2371.70 | 2400.85 | 2415.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 2396.20 | 2353.26 | 2349.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 2396.20 | 2353.26 | 2349.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 11:15:00 | 2421.20 | 2366.85 | 2356.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 14:15:00 | 2382.40 | 2384.76 | 2368.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 15:00:00 | 2382.40 | 2384.76 | 2368.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 2360.00 | 2379.81 | 2367.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 2381.95 | 2379.81 | 2367.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 2368.45 | 2377.54 | 2367.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 10:45:00 | 2417.70 | 2377.89 | 2368.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 2341.00 | 2376.49 | 2373.61 | SL hit (close<static) qty=1.00 sl=2344.75 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 2333.05 | 2367.80 | 2369.92 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 2408.00 | 2375.56 | 2371.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 2480.00 | 2404.72 | 2387.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 2577.30 | 2579.52 | 2514.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 10:00:00 | 2577.30 | 2579.52 | 2514.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 2784.35 | 2828.27 | 2808.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 2784.35 | 2828.27 | 2808.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 2777.00 | 2818.02 | 2805.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 2765.00 | 2818.02 | 2805.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 2754.75 | 2796.88 | 2797.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 11:15:00 | 2749.50 | 2787.41 | 2793.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 2785.00 | 2729.40 | 2747.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 2785.00 | 2729.40 | 2747.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 2785.00 | 2729.40 | 2747.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 2785.00 | 2729.40 | 2747.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 2834.00 | 2750.32 | 2754.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 2810.80 | 2750.32 | 2754.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 2844.35 | 2774.19 | 2765.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 2880.80 | 2795.51 | 2775.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 13:15:00 | 2779.70 | 2802.96 | 2783.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 13:15:00 | 2779.70 | 2802.96 | 2783.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 2779.70 | 2802.96 | 2783.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:00:00 | 2779.70 | 2802.96 | 2783.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 2772.40 | 2796.85 | 2782.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 2772.40 | 2796.85 | 2782.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 2794.90 | 2796.46 | 2783.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 10:45:00 | 2816.05 | 2794.72 | 2784.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 12:30:00 | 2812.35 | 2801.50 | 2789.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 13:30:00 | 2818.00 | 2808.46 | 2801.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 15:15:00 | 2773.50 | 2796.02 | 2796.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 2773.50 | 2796.02 | 2796.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 2742.30 | 2779.48 | 2788.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 2737.25 | 2723.17 | 2742.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 2737.25 | 2723.17 | 2742.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 2737.25 | 2723.17 | 2742.38 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 10:15:00 | 2761.00 | 2750.31 | 2749.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 11:15:00 | 2801.00 | 2760.45 | 2753.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 2868.90 | 2869.85 | 2838.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:15:00 | 2845.65 | 2869.85 | 2838.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 2838.70 | 2863.62 | 2838.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 2838.70 | 2863.62 | 2838.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 2805.40 | 2851.98 | 2835.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 2805.40 | 2851.98 | 2835.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 2829.90 | 2847.56 | 2834.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 14:15:00 | 2844.10 | 2847.56 | 2834.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 12:15:00 | 2792.45 | 2824.59 | 2828.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 2792.45 | 2824.59 | 2828.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 2783.60 | 2816.39 | 2824.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 09:15:00 | 2817.05 | 2798.50 | 2812.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 2817.05 | 2798.50 | 2812.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 2817.05 | 2798.50 | 2812.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:30:00 | 2854.25 | 2798.50 | 2812.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 2737.00 | 2786.20 | 2805.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 12:30:00 | 2701.80 | 2756.34 | 2788.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:15:00 | 2700.00 | 2715.44 | 2717.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 13:30:00 | 2705.00 | 2707.01 | 2712.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 14:00:00 | 2705.00 | 2707.01 | 2712.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 2700.95 | 2700.52 | 2707.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 2700.95 | 2700.52 | 2707.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 2709.00 | 2702.21 | 2707.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 2701.10 | 2702.21 | 2707.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 2691.70 | 2700.11 | 2706.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 14:15:00 | 2662.00 | 2696.93 | 2703.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 2718.10 | 2703.98 | 2705.60 | SL hit (close>static) qty=1.00 sl=2710.40 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 2760.00 | 2715.18 | 2710.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 2765.00 | 2742.18 | 2728.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 2879.90 | 2892.68 | 2845.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 09:45:00 | 2864.20 | 2892.68 | 2845.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 2867.10 | 2883.34 | 2852.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:45:00 | 2848.15 | 2883.34 | 2852.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 2840.00 | 2874.68 | 2851.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:30:00 | 2845.25 | 2874.68 | 2851.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 2866.00 | 2872.94 | 2852.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:45:00 | 2849.30 | 2872.94 | 2852.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 2851.00 | 2868.55 | 2852.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 2795.10 | 2868.55 | 2852.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 2843.00 | 2863.44 | 2851.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:15:00 | 2857.00 | 2863.44 | 2851.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 12:15:00 | 2823.60 | 2931.69 | 2937.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 2823.60 | 2931.69 | 2937.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 13:15:00 | 2767.25 | 2898.80 | 2922.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 13:15:00 | 2832.00 | 2826.08 | 2862.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 14:00:00 | 2832.00 | 2826.08 | 2862.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 2791.95 | 2822.07 | 2852.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:15:00 | 2763.55 | 2803.17 | 2835.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:45:00 | 2762.20 | 2792.05 | 2824.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 15:15:00 | 2754.00 | 2792.05 | 2824.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 2625.37 | 2725.77 | 2784.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 2624.09 | 2725.77 | 2784.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 2616.30 | 2725.77 | 2784.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 2861.45 | 2705.20 | 2735.25 | SL hit (close>ema200) qty=0.50 sl=2705.20 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 2861.95 | 2758.24 | 2755.49 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 2722.60 | 2749.41 | 2752.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 2687.40 | 2717.36 | 2731.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 09:15:00 | 2738.75 | 2717.02 | 2727.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 2738.75 | 2717.02 | 2727.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 2738.75 | 2717.02 | 2727.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:45:00 | 2730.95 | 2717.02 | 2727.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 2790.30 | 2731.68 | 2733.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 2790.30 | 2731.68 | 2733.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 2759.60 | 2737.26 | 2735.72 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 2697.85 | 2730.70 | 2735.04 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 14:15:00 | 2753.50 | 2739.19 | 2737.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 09:15:00 | 2767.15 | 2745.39 | 2741.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 15:15:00 | 2766.85 | 2768.53 | 2757.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 15:15:00 | 2766.85 | 2768.53 | 2757.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 2766.85 | 2768.53 | 2757.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 2868.35 | 2768.53 | 2757.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 12:15:00 | 2806.95 | 2828.39 | 2830.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 2806.95 | 2828.39 | 2830.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 2779.45 | 2818.60 | 2825.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 2762.05 | 2743.47 | 2772.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 2762.05 | 2743.47 | 2772.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 2762.05 | 2743.47 | 2772.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 2759.35 | 2743.47 | 2772.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 2656.05 | 2613.22 | 2655.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:00:00 | 2656.05 | 2613.22 | 2655.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 2650.05 | 2620.59 | 2655.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:45:00 | 2626.90 | 2623.47 | 2653.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:15:00 | 2629.90 | 2623.47 | 2653.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 2634.00 | 2573.38 | 2573.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 2634.00 | 2573.38 | 2573.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 13:15:00 | 2664.95 | 2615.90 | 2595.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 10:15:00 | 2609.70 | 2627.38 | 2608.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 10:15:00 | 2609.70 | 2627.38 | 2608.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 2609.70 | 2627.38 | 2608.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:45:00 | 2610.00 | 2627.38 | 2608.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 2620.10 | 2625.93 | 2609.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 2634.60 | 2612.37 | 2607.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-03 09:15:00 | 2898.06 | 2818.27 | 2781.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 2830.50 | 2851.60 | 2853.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 11:15:00 | 2819.25 | 2837.25 | 2844.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 2834.25 | 2820.55 | 2831.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 2834.25 | 2820.55 | 2831.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 2834.25 | 2820.55 | 2831.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 2834.25 | 2820.55 | 2831.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 2800.00 | 2816.44 | 2829.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:00:00 | 2789.35 | 2811.02 | 2825.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 10:15:00 | 2807.50 | 2773.72 | 2772.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 10:15:00 | 2807.50 | 2773.72 | 2772.70 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 15:15:00 | 2760.00 | 2770.42 | 2771.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 2744.65 | 2765.27 | 2769.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 2708.15 | 2696.04 | 2715.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:00:00 | 2708.15 | 2696.04 | 2715.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 2736.90 | 2704.21 | 2717.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:00:00 | 2736.90 | 2704.21 | 2717.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 2703.40 | 2704.05 | 2716.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 14:00:00 | 2660.60 | 2695.36 | 2711.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:00:00 | 2692.65 | 2681.35 | 2696.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 2558.02 | 2647.03 | 2671.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 13:15:00 | 2527.57 | 2570.27 | 2606.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 13:15:00 | 2493.50 | 2461.75 | 2482.56 | SL hit (close>ema200) qty=0.50 sl=2461.75 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 2544.00 | 2501.87 | 2496.92 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 13:15:00 | 2489.75 | 2503.96 | 2505.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 14:15:00 | 2461.80 | 2495.53 | 2501.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 11:15:00 | 2289.75 | 2283.50 | 2338.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:00:00 | 2289.75 | 2283.50 | 2338.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 2009.80 | 1982.37 | 2018.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 1974.20 | 1982.37 | 2018.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1954.60 | 1976.82 | 2012.44 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 2053.85 | 2016.57 | 2015.92 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 2000.15 | 2033.07 | 2033.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 1962.20 | 2003.85 | 2016.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1913.45 | 1877.94 | 1923.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1913.45 | 1877.94 | 1923.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1913.45 | 1877.94 | 1923.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1913.45 | 1877.94 | 1923.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1935.00 | 1889.35 | 1924.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 1935.00 | 1889.35 | 1924.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1920.40 | 1895.56 | 1924.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 1909.00 | 1919.17 | 1925.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:15:00 | 1905.90 | 1918.04 | 1924.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 1813.55 | 1882.41 | 1903.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 1810.61 | 1882.41 | 1903.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 13:15:00 | 1891.40 | 1879.94 | 1896.70 | SL hit (close>ema200) qty=0.50 sl=1879.94 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 1918.95 | 1891.09 | 1887.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 1943.70 | 1911.45 | 1899.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 2065.40 | 2108.18 | 2055.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 2065.40 | 2108.18 | 2055.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 2058.75 | 2116.14 | 2077.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 2055.80 | 2116.14 | 2077.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 2063.10 | 2105.54 | 2076.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 14:15:00 | 2091.50 | 2089.11 | 2075.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 12:15:00 | 2218.15 | 2240.32 | 2241.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 2218.15 | 2240.32 | 2241.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 14:15:00 | 2207.80 | 2230.22 | 2236.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 2140.60 | 2112.93 | 2150.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 2133.25 | 2112.93 | 2150.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 2130.00 | 2119.52 | 2144.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 2114.25 | 2119.52 | 2144.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2108.65 | 2117.35 | 2141.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:15:00 | 2104.30 | 2117.35 | 2141.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:45:00 | 2093.20 | 2114.40 | 2137.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 2097.00 | 2114.20 | 2131.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:15:00 | 2100.55 | 2114.20 | 2131.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1999.09 | 2047.18 | 2088.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1988.54 | 2047.18 | 2088.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1992.15 | 2047.18 | 2088.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1995.52 | 2047.18 | 2088.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 2074.25 | 2040.21 | 2077.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-14 14:15:00 | 2074.25 | 2040.21 | 2077.65 | SL hit (close>ema200) qty=0.50 sl=2040.21 alert=retest2 |

### Cycle 50 — BUY (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 12:15:00 | 2084.00 | 2049.70 | 2048.70 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 2017.40 | 2043.69 | 2046.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 2008.45 | 2031.53 | 2039.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 1875.00 | 1766.13 | 1822.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 1875.00 | 1766.13 | 1822.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1875.00 | 1766.13 | 1822.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:45:00 | 1856.20 | 1766.13 | 1822.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 1900.00 | 1792.91 | 1829.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 1752.30 | 1792.91 | 1829.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 13:15:00 | 1932.40 | 1827.38 | 1830.72 | SL hit (close>static) qty=1.00 sl=1923.80 alert=retest2 |

### Cycle 52 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 1988.05 | 1859.51 | 1845.02 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 1903.55 | 1957.40 | 1961.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 1893.20 | 1944.56 | 1955.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 15:15:00 | 1775.35 | 1773.97 | 1802.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:15:00 | 1839.15 | 1773.97 | 1802.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1821.20 | 1783.42 | 1803.97 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 1840.05 | 1820.05 | 1817.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1929.90 | 1861.63 | 1839.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 2005.25 | 2037.57 | 2004.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 2005.25 | 2037.57 | 2004.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 2005.25 | 2037.57 | 2004.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 2005.25 | 2037.57 | 2004.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1998.90 | 2029.83 | 2004.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 1999.55 | 2029.83 | 2004.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1979.65 | 2019.80 | 2002.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 1980.00 | 2019.80 | 2002.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 1962.00 | 1993.18 | 1994.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 10:15:00 | 1950.65 | 1979.66 | 1987.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 12:15:00 | 1978.15 | 1976.31 | 1984.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 12:15:00 | 1978.15 | 1976.31 | 1984.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 1978.15 | 1976.31 | 1984.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:30:00 | 2019.85 | 1976.31 | 1984.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 2009.10 | 1982.87 | 1986.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 2011.50 | 1982.87 | 1986.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 1946.40 | 1975.57 | 1982.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:15:00 | 1940.20 | 1970.69 | 1977.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 1943.15 | 1942.51 | 1943.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:45:00 | 1939.00 | 1942.64 | 1943.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 1962.40 | 1946.59 | 1944.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 1962.40 | 1946.59 | 1944.79 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1893.45 | 1943.20 | 1948.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 1862.40 | 1914.44 | 1933.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1825.00 | 1824.41 | 1863.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1840.20 | 1824.41 | 1863.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1823.90 | 1824.30 | 1859.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:15:00 | 1785.10 | 1827.80 | 1848.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 15:15:00 | 1695.84 | 1745.59 | 1787.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1758.60 | 1748.19 | 1784.70 | SL hit (close>ema200) qty=0.50 sl=1748.19 alert=retest2 |

### Cycle 58 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 1835.50 | 1797.47 | 1793.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 1850.60 | 1808.10 | 1798.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 1882.30 | 1887.14 | 1859.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:15:00 | 1879.90 | 1887.14 | 1859.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1973.70 | 1979.90 | 1959.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:45:00 | 1970.80 | 1979.90 | 1959.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1897.50 | 1963.38 | 1957.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1897.50 | 1963.38 | 1957.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1883.40 | 1947.38 | 1950.70 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 1942.00 | 1926.37 | 1925.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 1953.90 | 1934.38 | 1929.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 11:15:00 | 1931.20 | 1934.67 | 1930.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 11:15:00 | 1931.20 | 1934.67 | 1930.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 1931.20 | 1934.67 | 1930.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 1933.70 | 1934.67 | 1930.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 1936.50 | 1935.04 | 1930.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:30:00 | 1931.60 | 1935.04 | 1930.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 1920.00 | 1931.38 | 1929.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:30:00 | 1923.30 | 1931.38 | 1929.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 1917.00 | 1928.51 | 1928.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 10:15:00 | 1899.60 | 1919.35 | 1924.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1884.00 | 1878.60 | 1888.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 10:45:00 | 1880.00 | 1878.60 | 1888.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1883.90 | 1879.66 | 1888.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 1889.90 | 1879.66 | 1888.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1883.80 | 1880.49 | 1888.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 1881.10 | 1880.49 | 1888.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1896.20 | 1884.15 | 1888.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1896.20 | 1884.15 | 1888.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1876.80 | 1882.68 | 1887.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1898.90 | 1882.68 | 1887.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1897.70 | 1885.68 | 1888.28 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 1899.40 | 1890.38 | 1890.09 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 1886.80 | 1889.76 | 1889.86 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 15:15:00 | 1896.10 | 1890.49 | 1890.15 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1821.60 | 1876.71 | 1883.91 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1923.50 | 1872.58 | 1872.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 1941.40 | 1927.32 | 1911.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 2135.40 | 2160.33 | 2130.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:00:00 | 2135.40 | 2160.33 | 2130.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 2149.00 | 2158.06 | 2131.84 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 2097.30 | 2122.20 | 2123.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 2083.00 | 2109.69 | 2117.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 2108.00 | 2089.53 | 2101.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 2108.00 | 2089.53 | 2101.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 2108.00 | 2089.53 | 2101.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 2108.00 | 2089.53 | 2101.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 2120.50 | 2095.73 | 2102.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 2125.20 | 2095.73 | 2102.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2115.40 | 2102.28 | 2104.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 2115.40 | 2102.28 | 2104.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 2102.00 | 2105.22 | 2105.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:15:00 | 2095.30 | 2105.22 | 2105.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 15:15:00 | 2192.20 | 2122.05 | 2113.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 2192.20 | 2122.05 | 2113.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 2232.30 | 2182.88 | 2154.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 2220.00 | 2231.79 | 2209.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 14:45:00 | 2220.00 | 2231.79 | 2209.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 2167.50 | 2237.78 | 2236.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 2167.50 | 2237.78 | 2236.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 2142.00 | 2218.62 | 2227.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 13:15:00 | 2128.60 | 2160.59 | 2185.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 14:15:00 | 2142.70 | 2138.55 | 2157.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 14:30:00 | 2150.10 | 2138.55 | 2157.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 2149.40 | 2139.64 | 2154.57 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 2203.40 | 2165.58 | 2162.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 2233.50 | 2179.16 | 2169.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 2435.00 | 2440.06 | 2388.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 2445.50 | 2440.06 | 2388.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 2446.10 | 2470.31 | 2445.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 2446.10 | 2470.31 | 2445.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 2448.90 | 2466.03 | 2446.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:45:00 | 2470.60 | 2466.03 | 2446.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 2461.00 | 2465.02 | 2447.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 2405.60 | 2465.02 | 2447.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2398.00 | 2451.62 | 2442.95 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 2400.00 | 2432.68 | 2435.31 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 2472.00 | 2438.54 | 2435.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 2494.70 | 2449.77 | 2441.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 13:15:00 | 2500.00 | 2503.15 | 2482.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 14:00:00 | 2500.00 | 2503.15 | 2482.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 2476.10 | 2497.74 | 2482.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 2476.10 | 2497.74 | 2482.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2456.10 | 2489.41 | 2479.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:30:00 | 2481.50 | 2484.95 | 2478.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 2455.10 | 2474.75 | 2474.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 2455.10 | 2474.75 | 2474.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 2439.60 | 2465.29 | 2470.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 14:15:00 | 2456.60 | 2436.26 | 2449.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 14:15:00 | 2456.60 | 2436.26 | 2449.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 2456.60 | 2436.26 | 2449.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:45:00 | 2459.70 | 2436.26 | 2449.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 2444.00 | 2437.81 | 2449.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 2388.70 | 2437.81 | 2449.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 2417.80 | 2431.09 | 2439.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 15:15:00 | 2414.00 | 2383.07 | 2381.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 15:15:00 | 2414.00 | 2383.07 | 2381.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 2439.90 | 2404.86 | 2393.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 13:15:00 | 2409.40 | 2415.31 | 2405.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 13:15:00 | 2409.40 | 2415.31 | 2405.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 2409.40 | 2415.31 | 2405.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 2409.40 | 2415.31 | 2405.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 2404.00 | 2412.92 | 2406.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 2407.10 | 2410.74 | 2405.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2412.20 | 2411.03 | 2406.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:30:00 | 2416.30 | 2413.90 | 2407.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 2397.60 | 2407.08 | 2406.98 | SL hit (close<static) qty=1.00 sl=2400.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 2392.00 | 2404.06 | 2405.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 12:15:00 | 2375.30 | 2398.31 | 2402.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 2340.00 | 2332.09 | 2356.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 2340.00 | 2332.09 | 2356.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2340.00 | 2332.09 | 2356.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:45:00 | 2284.60 | 2325.53 | 2347.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 2234.40 | 2218.47 | 2217.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 2234.40 | 2218.47 | 2217.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 2240.00 | 2225.26 | 2221.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 13:15:00 | 2229.30 | 2233.52 | 2227.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 13:15:00 | 2229.30 | 2233.52 | 2227.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 2229.30 | 2233.52 | 2227.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 2225.70 | 2233.52 | 2227.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 2237.60 | 2234.33 | 2228.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 2227.40 | 2234.33 | 2228.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2221.30 | 2230.95 | 2227.67 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 2197.50 | 2224.26 | 2224.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 11:15:00 | 2189.00 | 2217.21 | 2221.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 10:15:00 | 2192.70 | 2189.46 | 2203.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-17 11:00:00 | 2192.70 | 2189.46 | 2203.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 2186.80 | 2188.93 | 2201.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:30:00 | 2192.70 | 2188.93 | 2201.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 2158.00 | 2137.74 | 2157.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 2158.00 | 2137.74 | 2157.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 2147.30 | 2139.65 | 2156.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 2113.90 | 2149.12 | 2152.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 13:15:00 | 2008.20 | 2104.51 | 2128.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 2081.80 | 2068.63 | 2097.36 | SL hit (close>ema200) qty=0.50 sl=2068.63 alert=retest2 |

### Cycle 78 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1935.70 | 1913.06 | 1911.81 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1908.50 | 1913.41 | 1913.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1887.60 | 1908.25 | 1911.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 1885.00 | 1879.28 | 1887.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 1885.00 | 1879.28 | 1887.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1885.00 | 1879.28 | 1887.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:45:00 | 1885.60 | 1879.28 | 1887.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1879.60 | 1879.35 | 1887.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:30:00 | 1880.40 | 1879.35 | 1887.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1885.00 | 1880.48 | 1886.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 1885.00 | 1880.48 | 1886.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1872.00 | 1878.78 | 1885.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:30:00 | 1883.00 | 1878.78 | 1885.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1862.50 | 1876.22 | 1883.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 1848.70 | 1876.22 | 1883.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:00:00 | 1842.80 | 1854.99 | 1864.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 09:15:00 | 1756.26 | 1787.92 | 1812.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 1789.30 | 1787.92 | 1812.57 | SL hit (close>static) qty=0.50 sl=1787.92 alert=retest2 |

### Cycle 80 — BUY (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 14:15:00 | 1831.70 | 1807.47 | 1805.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 1851.20 | 1816.21 | 1809.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 1850.10 | 1850.63 | 1834.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:15:00 | 1833.00 | 1850.63 | 1834.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1806.60 | 1841.83 | 1832.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 1806.60 | 1841.83 | 1832.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 1813.60 | 1836.18 | 1830.47 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 1810.20 | 1826.36 | 1826.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 13:15:00 | 1803.10 | 1821.71 | 1824.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 1822.00 | 1817.69 | 1821.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1822.00 | 1817.69 | 1821.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1822.00 | 1817.69 | 1821.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1834.70 | 1817.69 | 1821.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1825.30 | 1819.21 | 1821.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1825.30 | 1819.21 | 1821.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1827.20 | 1820.81 | 1822.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 1827.50 | 1820.81 | 1822.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1806.00 | 1817.91 | 1820.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1792.40 | 1814.18 | 1818.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 1802.20 | 1810.80 | 1816.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1844.50 | 1811.65 | 1812.06 | SL hit (close>static) qty=1.00 sl=1822.30 alert=retest2 |

### Cycle 82 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 1837.50 | 1816.82 | 1814.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 14:15:00 | 1845.40 | 1829.93 | 1821.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1813.40 | 1828.39 | 1822.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1813.40 | 1828.39 | 1822.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1813.40 | 1828.39 | 1822.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 1807.90 | 1828.39 | 1822.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1810.70 | 1824.85 | 1821.41 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 1815.30 | 1818.92 | 1819.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1790.90 | 1813.32 | 1816.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1778.00 | 1768.90 | 1785.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1778.00 | 1768.90 | 1785.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1745.00 | 1756.18 | 1772.70 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1821.10 | 1779.46 | 1776.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 1851.50 | 1793.87 | 1783.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1820.80 | 1821.81 | 1806.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1820.80 | 1821.81 | 1806.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1812.90 | 1818.79 | 1809.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 1812.90 | 1818.79 | 1809.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1820.00 | 1819.03 | 1810.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 1834.70 | 1819.14 | 1815.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 1804.50 | 1816.21 | 1814.40 | SL hit (close<static) qty=1.00 sl=1807.30 alert=retest2 |

### Cycle 85 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1793.80 | 1811.73 | 1812.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 13:15:00 | 1781.40 | 1799.38 | 1805.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1802.90 | 1792.79 | 1799.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1802.90 | 1792.79 | 1799.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1802.90 | 1792.79 | 1799.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 1805.90 | 1792.79 | 1799.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1801.40 | 1794.51 | 1800.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 1803.20 | 1794.51 | 1800.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1786.40 | 1792.89 | 1798.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:15:00 | 1779.90 | 1792.89 | 1798.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:00:00 | 1780.90 | 1789.38 | 1795.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 1808.00 | 1795.76 | 1795.81 | SL hit (close>static) qty=1.00 sl=1805.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 1810.00 | 1798.61 | 1797.10 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 1785.60 | 1795.42 | 1796.13 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 1807.20 | 1798.41 | 1797.33 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 1789.00 | 1796.39 | 1796.65 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1806.60 | 1797.99 | 1797.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 11:15:00 | 1811.70 | 1801.52 | 1799.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 1924.50 | 1928.60 | 1905.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:00:00 | 1924.50 | 1928.60 | 1905.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1909.30 | 1923.27 | 1907.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 1909.30 | 1923.27 | 1907.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1915.00 | 1921.61 | 1907.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 1911.00 | 1921.61 | 1907.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1900.00 | 1917.29 | 1907.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1917.60 | 1914.09 | 1906.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 1921.00 | 1915.91 | 1908.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1891.70 | 1903.89 | 1905.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1891.70 | 1903.89 | 1905.52 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 1910.70 | 1906.32 | 1906.11 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 1893.00 | 1903.66 | 1904.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 1878.00 | 1897.11 | 1901.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1758.90 | 1758.44 | 1787.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:45:00 | 1760.00 | 1758.44 | 1787.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1591.20 | 1592.52 | 1619.45 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1627.80 | 1617.81 | 1616.68 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 1614.30 | 1619.75 | 1619.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1601.80 | 1614.38 | 1617.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 12:15:00 | 1611.00 | 1610.57 | 1614.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 13:00:00 | 1611.00 | 1610.57 | 1614.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1613.40 | 1611.14 | 1614.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:30:00 | 1616.60 | 1611.14 | 1614.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1615.60 | 1612.03 | 1614.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 1618.00 | 1612.03 | 1614.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1613.10 | 1612.24 | 1614.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 1628.10 | 1612.24 | 1614.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 1663.20 | 1622.43 | 1618.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 1718.70 | 1649.76 | 1632.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 1644.80 | 1649.99 | 1635.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 14:00:00 | 1644.80 | 1649.99 | 1635.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1647.90 | 1647.98 | 1638.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:15:00 | 1641.70 | 1647.98 | 1638.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1638.70 | 1646.12 | 1638.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 1638.70 | 1646.12 | 1638.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1642.90 | 1645.48 | 1638.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:15:00 | 1639.70 | 1645.48 | 1638.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1632.10 | 1642.80 | 1637.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 1629.40 | 1642.80 | 1637.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1624.50 | 1639.14 | 1636.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 1624.50 | 1639.14 | 1636.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1630.20 | 1637.35 | 1636.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:30:00 | 1625.30 | 1637.35 | 1636.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1630.60 | 1635.63 | 1635.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 1630.60 | 1635.63 | 1635.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1633.90 | 1635.28 | 1635.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 15:15:00 | 1628.50 | 1632.16 | 1633.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 1631.60 | 1631.27 | 1632.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 1631.60 | 1631.27 | 1632.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1631.60 | 1631.27 | 1632.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:45:00 | 1635.70 | 1631.27 | 1632.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 1635.40 | 1632.10 | 1633.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:45:00 | 1637.00 | 1632.10 | 1633.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1643.80 | 1634.44 | 1634.16 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1631.00 | 1633.54 | 1633.78 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 1636.60 | 1634.15 | 1634.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1648.00 | 1636.92 | 1635.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 1722.50 | 1728.80 | 1706.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:45:00 | 1721.10 | 1728.80 | 1706.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1686.00 | 1718.50 | 1705.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 1692.30 | 1718.50 | 1705.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1685.30 | 1711.86 | 1703.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 14:30:00 | 1700.10 | 1706.79 | 1702.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:30:00 | 1694.40 | 1703.30 | 1701.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-30 09:15:00 | 1863.84 | 1786.89 | 1749.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1812.80 | 1851.36 | 1852.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 1802.40 | 1837.12 | 1845.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 13:15:00 | 1753.80 | 1751.16 | 1767.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 13:45:00 | 1754.50 | 1751.16 | 1767.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1763.90 | 1754.32 | 1765.27 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 15:15:00 | 1785.00 | 1771.92 | 1770.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 1802.20 | 1777.98 | 1773.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1777.10 | 1784.52 | 1778.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 1777.10 | 1784.52 | 1778.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1777.10 | 1784.52 | 1778.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:00:00 | 1777.10 | 1784.52 | 1778.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1768.40 | 1781.30 | 1777.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 1768.40 | 1781.30 | 1777.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 1747.20 | 1774.48 | 1774.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 1732.50 | 1749.83 | 1759.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 11:15:00 | 1752.00 | 1747.89 | 1756.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 11:15:00 | 1752.00 | 1747.89 | 1756.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1752.00 | 1747.89 | 1756.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 1752.00 | 1747.89 | 1756.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1750.80 | 1748.47 | 1756.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:30:00 | 1751.60 | 1748.47 | 1756.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1755.70 | 1749.92 | 1756.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:30:00 | 1756.30 | 1749.92 | 1756.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1747.00 | 1749.33 | 1755.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 1755.30 | 1749.33 | 1755.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1757.80 | 1751.03 | 1755.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1732.60 | 1751.03 | 1755.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:00:00 | 1745.00 | 1744.66 | 1749.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1798.10 | 1756.20 | 1753.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 1798.10 | 1756.20 | 1753.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 10:15:00 | 1808.60 | 1766.68 | 1758.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 1780.70 | 1789.75 | 1776.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1780.70 | 1789.75 | 1776.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1780.70 | 1789.75 | 1776.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1780.70 | 1789.75 | 1776.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1787.80 | 1789.36 | 1777.40 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1758.20 | 1774.08 | 1775.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1743.00 | 1764.06 | 1769.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 1730.30 | 1727.76 | 1741.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 11:15:00 | 1744.60 | 1727.76 | 1741.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 1756.90 | 1733.59 | 1743.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 1756.90 | 1733.59 | 1743.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 1757.80 | 1738.43 | 1744.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:00:00 | 1757.80 | 1738.43 | 1744.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1734.80 | 1735.18 | 1741.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 1743.30 | 1735.18 | 1741.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1741.80 | 1736.50 | 1741.26 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1759.50 | 1746.16 | 1744.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 1775.00 | 1751.93 | 1747.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 1756.30 | 1758.03 | 1752.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 11:15:00 | 1756.30 | 1758.03 | 1752.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1756.30 | 1758.03 | 1752.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:30:00 | 1753.10 | 1758.03 | 1752.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1743.90 | 1755.21 | 1751.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 1743.90 | 1755.21 | 1751.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1749.50 | 1754.06 | 1751.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 1749.50 | 1754.06 | 1751.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1746.20 | 1752.49 | 1750.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 1746.20 | 1752.49 | 1750.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1745.00 | 1750.99 | 1750.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 1759.50 | 1750.99 | 1750.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 1758.40 | 1757.70 | 1754.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 1759.00 | 1757.70 | 1754.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1756.20 | 1763.86 | 1759.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 1751.70 | 1763.86 | 1759.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1750.70 | 1761.23 | 1759.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 1757.70 | 1760.20 | 1758.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 1761.00 | 1760.20 | 1758.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:45:00 | 1759.30 | 1767.22 | 1764.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 14:15:00 | 1746.50 | 1761.95 | 1762.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 1746.50 | 1761.95 | 1762.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1734.30 | 1754.05 | 1758.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 1768.20 | 1747.40 | 1752.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 14:15:00 | 1768.20 | 1747.40 | 1752.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1768.20 | 1747.40 | 1752.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 1768.20 | 1747.40 | 1752.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 1769.30 | 1751.78 | 1753.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1762.70 | 1751.78 | 1753.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 1784.50 | 1760.52 | 1757.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 1784.50 | 1760.52 | 1757.54 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 1713.90 | 1757.06 | 1760.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 1666.90 | 1738.09 | 1750.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 1645.60 | 1643.05 | 1670.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 10:00:00 | 1645.60 | 1643.05 | 1670.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 1665.50 | 1648.49 | 1668.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:00:00 | 1665.50 | 1648.49 | 1668.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1658.90 | 1650.57 | 1667.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:45:00 | 1651.80 | 1651.04 | 1666.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:30:00 | 1651.40 | 1652.13 | 1665.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 15:15:00 | 1650.00 | 1652.13 | 1665.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 1644.30 | 1651.14 | 1662.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1657.80 | 1648.79 | 1656.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 1657.80 | 1648.79 | 1656.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1654.00 | 1649.83 | 1656.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 1669.00 | 1649.83 | 1656.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1674.20 | 1654.71 | 1658.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 1674.20 | 1654.71 | 1658.07 | SL hit (close>static) qty=1.00 sl=1668.80 alert=retest2 |

### Cycle 110 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 1672.50 | 1662.16 | 1660.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 15:15:00 | 1674.00 | 1666.15 | 1663.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1691.60 | 1703.87 | 1689.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1691.60 | 1703.87 | 1689.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1691.60 | 1703.87 | 1689.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1686.60 | 1703.87 | 1689.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1701.20 | 1702.01 | 1690.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 1690.70 | 1702.01 | 1690.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1712.00 | 1730.39 | 1718.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 1712.00 | 1730.39 | 1718.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1716.00 | 1727.51 | 1718.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:30:00 | 1725.80 | 1727.51 | 1718.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1728.30 | 1727.67 | 1719.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 1751.90 | 1722.85 | 1719.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 1736.40 | 1725.29 | 1721.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 1738.00 | 1727.60 | 1722.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 1723.10 | 1733.43 | 1734.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 1723.10 | 1733.43 | 1734.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 1719.00 | 1730.54 | 1733.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 1648.40 | 1642.07 | 1656.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 11:00:00 | 1648.40 | 1642.07 | 1656.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1650.30 | 1643.71 | 1656.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:30:00 | 1656.10 | 1643.71 | 1656.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1654.40 | 1645.85 | 1656.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 1657.30 | 1645.85 | 1656.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1656.00 | 1647.88 | 1656.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 1666.50 | 1647.88 | 1656.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1674.50 | 1653.21 | 1657.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 1674.50 | 1653.21 | 1657.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1665.80 | 1655.72 | 1658.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1662.40 | 1655.72 | 1658.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 13:15:00 | 1665.20 | 1660.66 | 1660.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 1665.20 | 1660.66 | 1660.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 1683.00 | 1665.13 | 1662.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 1672.50 | 1673.48 | 1668.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 12:15:00 | 1672.50 | 1673.48 | 1668.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 1672.50 | 1673.48 | 1668.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:30:00 | 1668.00 | 1673.48 | 1668.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1667.80 | 1672.34 | 1668.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1679.10 | 1669.84 | 1667.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 15:15:00 | 1671.10 | 1679.15 | 1679.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 1671.10 | 1679.15 | 1679.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 1660.90 | 1675.50 | 1678.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 1670.20 | 1670.09 | 1674.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:45:00 | 1669.50 | 1670.09 | 1674.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1659.50 | 1668.02 | 1672.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:15:00 | 1655.00 | 1668.02 | 1672.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1572.25 | 1618.36 | 1633.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1543.80 | 1542.66 | 1559.04 | SL hit (close>ema200) qty=0.50 sl=1542.66 alert=retest2 |

### Cycle 114 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 1249.70 | 1238.43 | 1237.59 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 12:15:00 | 1230.00 | 1236.74 | 1236.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 13:15:00 | 1219.20 | 1233.23 | 1235.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 10:15:00 | 1240.00 | 1229.60 | 1232.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 10:15:00 | 1240.00 | 1229.60 | 1232.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1240.00 | 1229.60 | 1232.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:30:00 | 1234.90 | 1229.60 | 1232.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 11:15:00 | 1267.00 | 1237.08 | 1235.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 1304.40 | 1258.98 | 1246.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 1443.00 | 1447.90 | 1419.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:45:00 | 1444.70 | 1447.90 | 1419.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1422.50 | 1440.86 | 1425.11 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1375.60 | 1412.84 | 1416.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 14:15:00 | 1372.80 | 1379.59 | 1386.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 1331.90 | 1329.42 | 1345.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 1331.90 | 1329.42 | 1345.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1286.60 | 1271.78 | 1282.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 1287.00 | 1271.78 | 1282.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1289.00 | 1275.22 | 1282.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 1289.00 | 1275.22 | 1282.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1296.00 | 1279.38 | 1283.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 1296.00 | 1279.38 | 1283.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 1290.20 | 1283.76 | 1285.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:45:00 | 1293.60 | 1283.76 | 1285.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 1295.00 | 1286.01 | 1285.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 1311.90 | 1291.19 | 1288.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 12:15:00 | 1292.10 | 1292.39 | 1289.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 12:15:00 | 1292.10 | 1292.39 | 1289.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1292.10 | 1292.39 | 1289.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:15:00 | 1280.60 | 1292.39 | 1289.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1284.70 | 1290.85 | 1289.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 1284.40 | 1290.85 | 1289.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 1276.90 | 1288.06 | 1288.10 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 15:15:00 | 1295.40 | 1289.53 | 1288.77 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1281.70 | 1287.96 | 1288.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1203.50 | 1230.49 | 1235.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 09:15:00 | 1197.50 | 1187.01 | 1199.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 1197.50 | 1187.01 | 1199.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1197.50 | 1187.01 | 1199.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 1150.90 | 1188.14 | 1195.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1172.40 | 1190.85 | 1193.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:45:00 | 1170.30 | 1185.96 | 1190.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 11:45:00 | 1171.00 | 1179.43 | 1186.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 1113.78 | 1150.16 | 1168.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1111.78 | 1141.27 | 1162.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1112.45 | 1141.27 | 1162.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 12:15:00 | 1093.36 | 1124.25 | 1150.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1120.50 | 1108.64 | 1125.76 | SL hit (close>ema200) qty=0.50 sl=1108.64 alert=retest2 |

### Cycle 122 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1180.80 | 1136.24 | 1134.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 1204.10 | 1149.81 | 1140.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1176.20 | 1189.93 | 1167.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 10:00:00 | 1176.20 | 1189.93 | 1167.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1184.50 | 1191.62 | 1176.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1184.50 | 1191.62 | 1176.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 1181.00 | 1189.49 | 1177.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1189.20 | 1189.49 | 1177.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 1168.90 | 1183.42 | 1177.44 | SL hit (close<static) qty=1.00 sl=1170.80 alert=retest2 |

### Cycle 123 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 1164.00 | 1173.51 | 1174.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1127.00 | 1164.21 | 1169.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1138.10 | 1124.86 | 1138.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1138.10 | 1124.86 | 1138.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1138.10 | 1124.86 | 1138.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 1138.10 | 1124.86 | 1138.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1138.40 | 1127.57 | 1138.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1140.00 | 1127.57 | 1138.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1133.30 | 1128.71 | 1137.80 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1203.00 | 1144.42 | 1143.41 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1111.40 | 1147.76 | 1149.81 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 1157.50 | 1138.98 | 1137.90 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1113.30 | 1136.98 | 1139.55 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 1150.00 | 1140.56 | 1139.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 1157.30 | 1143.91 | 1141.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 13:15:00 | 1171.50 | 1173.49 | 1162.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:00:00 | 1171.50 | 1173.49 | 1162.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1417.90 | 1419.53 | 1408.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 1398.30 | 1419.53 | 1408.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 1407.20 | 1415.65 | 1410.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 1432.80 | 1413.60 | 1410.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 14:30:00 | 1419.60 | 1423.34 | 1422.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1427.80 | 1421.87 | 1421.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 1388.70 | 1417.52 | 1420.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1388.70 | 1417.52 | 1420.84 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1443.30 | 1420.40 | 1419.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 1459.50 | 1428.22 | 1422.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1511.70 | 1530.62 | 1507.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 1511.70 | 1530.62 | 1507.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1487.10 | 1521.92 | 1505.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 1487.10 | 1521.92 | 1505.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1499.20 | 1517.37 | 1505.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1511.50 | 1502.38 | 1501.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 1505.20 | 1500.89 | 1500.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 1490.00 | 1502.21 | 1502.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 15:15:00 | 1490.00 | 1502.21 | 1502.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 1478.00 | 1497.36 | 1500.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1494.30 | 1485.21 | 1490.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1494.30 | 1485.21 | 1490.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1494.30 | 1485.21 | 1490.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:30:00 | 1502.40 | 1485.21 | 1490.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1498.90 | 1487.95 | 1491.35 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1518.00 | 1493.96 | 1493.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 1553.90 | 1510.50 | 1501.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 1525.80 | 1557.47 | 1540.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 1525.80 | 1557.47 | 1540.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1525.80 | 1557.47 | 1540.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 1525.80 | 1557.47 | 1540.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1505.50 | 1547.07 | 1536.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 1505.50 | 1547.07 | 1536.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 1497.10 | 1530.32 | 1530.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 1489.10 | 1516.85 | 1524.10 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 13:00:00 | 2156.95 | 2024-05-23 10:15:00 | 2120.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-05-23 09:15:00 | 2157.60 | 2024-05-23 10:15:00 | 2120.40 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-05-23 10:15:00 | 2154.50 | 2024-05-23 10:15:00 | 2120.40 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-06-13 11:30:00 | 2146.30 | 2024-06-18 12:15:00 | 2360.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-01 10:15:00 | 2323.00 | 2024-07-04 14:15:00 | 2320.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-07-02 09:30:00 | 2320.00 | 2024-07-04 14:15:00 | 2320.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-07-09 15:00:00 | 2236.35 | 2024-07-10 11:15:00 | 2282.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-07-10 09:30:00 | 2203.00 | 2024-07-10 11:15:00 | 2282.10 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2024-07-10 13:45:00 | 2242.75 | 2024-07-12 15:15:00 | 2130.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-10 13:45:00 | 2242.75 | 2024-07-16 09:15:00 | 2232.85 | STOP_HIT | 0.50 | 0.44% |
| BUY | retest2 | 2024-07-19 13:15:00 | 2231.80 | 2024-07-19 14:15:00 | 2192.35 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-07-22 11:15:00 | 2227.05 | 2024-07-22 12:15:00 | 2167.25 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-07-23 12:15:00 | 2124.45 | 2024-07-24 09:15:00 | 2225.55 | STOP_HIT | 1.00 | -4.76% |
| SELL | retest2 | 2024-07-23 15:15:00 | 2160.00 | 2024-07-24 09:15:00 | 2225.55 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-07-31 12:00:00 | 2370.85 | 2024-08-01 13:15:00 | 2310.50 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-07-31 13:15:00 | 2366.50 | 2024-08-01 13:15:00 | 2310.50 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-08-01 10:00:00 | 2368.90 | 2024-08-01 13:15:00 | 2310.50 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-08-07 14:15:00 | 2151.00 | 2024-08-09 11:15:00 | 2193.15 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-08-12 09:15:00 | 2219.35 | 2024-08-14 09:15:00 | 2126.30 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-08-12 10:15:00 | 2207.90 | 2024-08-14 09:15:00 | 2126.30 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2024-08-13 15:00:00 | 2201.00 | 2024-08-14 09:15:00 | 2126.30 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-08-19 14:30:00 | 2123.00 | 2024-08-19 15:15:00 | 2149.95 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-09-02 12:00:00 | 2371.70 | 2024-09-05 10:15:00 | 2396.20 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-09-06 10:45:00 | 2417.70 | 2024-09-09 09:15:00 | 2341.00 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-09-23 10:45:00 | 2816.05 | 2024-09-24 15:15:00 | 2773.50 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-09-23 12:30:00 | 2812.35 | 2024-09-24 15:15:00 | 2773.50 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-09-24 13:30:00 | 2818.00 | 2024-09-24 15:15:00 | 2773.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-10-03 14:15:00 | 2844.10 | 2024-10-04 12:15:00 | 2792.45 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-10-07 12:30:00 | 2701.80 | 2024-10-14 09:15:00 | 2718.10 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-10-10 11:15:00 | 2700.00 | 2024-10-14 10:15:00 | 2760.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-10-10 13:30:00 | 2705.00 | 2024-10-14 10:15:00 | 2760.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-10-10 14:00:00 | 2705.00 | 2024-10-14 10:15:00 | 2760.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-10-11 14:15:00 | 2662.00 | 2024-10-14 10:15:00 | 2760.00 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2024-10-18 10:15:00 | 2857.00 | 2024-10-22 12:15:00 | 2823.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-10-24 13:15:00 | 2763.55 | 2024-10-25 10:15:00 | 2625.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 14:45:00 | 2762.20 | 2024-10-25 10:15:00 | 2624.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 15:15:00 | 2754.00 | 2024-10-25 10:15:00 | 2616.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 13:15:00 | 2763.55 | 2024-10-28 10:15:00 | 2861.45 | STOP_HIT | 0.50 | -3.54% |
| SELL | retest2 | 2024-10-24 14:45:00 | 2762.20 | 2024-10-28 10:15:00 | 2861.45 | STOP_HIT | 0.50 | -3.59% |
| SELL | retest2 | 2024-10-24 15:15:00 | 2754.00 | 2024-10-28 10:15:00 | 2861.45 | STOP_HIT | 0.50 | -3.90% |
| BUY | retest2 | 2024-11-06 09:15:00 | 2868.35 | 2024-11-08 12:15:00 | 2806.95 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-11-14 12:45:00 | 2626.90 | 2024-11-25 09:15:00 | 2634.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-11-14 13:15:00 | 2629.90 | 2024-11-25 09:15:00 | 2634.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-11-27 09:15:00 | 2634.60 | 2024-12-03 09:15:00 | 2898.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-11 12:00:00 | 2789.35 | 2024-12-16 10:15:00 | 2807.50 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-12-19 14:00:00 | 2660.60 | 2024-12-23 09:15:00 | 2558.02 | PARTIAL | 0.50 | 3.86% |
| SELL | retest2 | 2024-12-20 12:00:00 | 2692.65 | 2024-12-24 13:15:00 | 2527.57 | PARTIAL | 0.50 | 6.13% |
| SELL | retest2 | 2024-12-19 14:00:00 | 2660.60 | 2024-12-31 13:15:00 | 2493.50 | STOP_HIT | 0.50 | 6.28% |
| SELL | retest2 | 2024-12-20 12:00:00 | 2692.65 | 2024-12-31 13:15:00 | 2493.50 | STOP_HIT | 0.50 | 7.40% |
| SELL | retest2 | 2025-01-24 12:30:00 | 1909.00 | 2025-01-27 10:15:00 | 1813.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:15:00 | 1905.90 | 2025-01-27 10:15:00 | 1810.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 1909.00 | 2025-01-27 13:15:00 | 1891.40 | STOP_HIT | 0.50 | 0.92% |
| SELL | retest2 | 2025-01-24 14:15:00 | 1905.90 | 2025-01-27 13:15:00 | 1891.40 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2025-01-28 14:45:00 | 1909.45 | 2025-01-29 09:15:00 | 1918.95 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-01-29 09:15:00 | 1907.90 | 2025-01-29 09:15:00 | 1918.95 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-02-03 14:15:00 | 2091.50 | 2025-02-10 12:15:00 | 2218.15 | STOP_HIT | 1.00 | 6.06% |
| SELL | retest2 | 2025-02-13 10:15:00 | 2104.30 | 2025-02-14 12:15:00 | 1999.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:45:00 | 2093.20 | 2025-02-14 12:15:00 | 1988.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:30:00 | 2097.00 | 2025-02-14 12:15:00 | 1992.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:15:00 | 2100.55 | 2025-02-14 12:15:00 | 1995.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:15:00 | 2104.30 | 2025-02-14 14:15:00 | 2074.25 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2025-02-13 10:45:00 | 2093.20 | 2025-02-14 14:15:00 | 2074.25 | STOP_HIT | 0.50 | 0.91% |
| SELL | retest2 | 2025-02-13 13:30:00 | 2097.00 | 2025-02-14 14:15:00 | 2074.25 | STOP_HIT | 0.50 | 1.08% |
| SELL | retest2 | 2025-02-13 14:15:00 | 2100.55 | 2025-02-14 14:15:00 | 2074.25 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2025-03-03 09:15:00 | 1752.30 | 2025-03-03 13:15:00 | 1932.40 | STOP_HIT | 1.00 | -10.28% |
| SELL | retest2 | 2025-03-27 12:15:00 | 1940.20 | 2025-04-01 14:15:00 | 1962.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-04-01 11:30:00 | 1943.15 | 2025-04-01 14:15:00 | 1962.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-04-01 12:45:00 | 1939.00 | 2025-04-01 14:15:00 | 1962.40 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-04-08 15:15:00 | 1785.10 | 2025-04-09 15:15:00 | 1695.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-08 15:15:00 | 1785.10 | 2025-04-11 09:15:00 | 1758.60 | STOP_HIT | 0.50 | 1.48% |
| SELL | retest2 | 2025-05-23 14:15:00 | 2095.30 | 2025-05-23 15:15:00 | 2192.20 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest2 | 2025-06-18 09:30:00 | 2481.50 | 2025-06-18 11:15:00 | 2455.10 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-06-20 09:15:00 | 2388.70 | 2025-06-27 15:15:00 | 2414.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-06-20 15:00:00 | 2417.80 | 2025-06-27 15:15:00 | 2414.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-07-02 11:30:00 | 2416.30 | 2025-07-03 10:15:00 | 2397.60 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-07-07 12:45:00 | 2284.60 | 2025-07-14 14:15:00 | 2234.40 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2025-07-23 09:15:00 | 2113.90 | 2025-07-23 13:15:00 | 2008.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 2113.90 | 2025-07-24 11:15:00 | 2081.80 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2025-08-08 15:15:00 | 1848.70 | 2025-08-14 09:15:00 | 1756.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 15:15:00 | 1848.70 | 2025-08-14 09:15:00 | 1789.30 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-08-12 11:00:00 | 1842.80 | 2025-08-18 14:15:00 | 1831.70 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-08-22 09:15:00 | 1792.40 | 2025-08-25 10:15:00 | 1844.50 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-08-22 10:45:00 | 1802.20 | 2025-08-25 10:15:00 | 1844.50 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-09-05 09:45:00 | 1834.70 | 2025-09-05 10:15:00 | 1804.50 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-09-09 12:15:00 | 1779.90 | 2025-09-11 09:15:00 | 1808.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-09 15:00:00 | 1780.90 | 2025-09-11 09:15:00 | 1808.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-09-19 10:15:00 | 1917.60 | 2025-09-22 09:15:00 | 1891.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-09-19 10:45:00 | 1921.00 | 2025-09-22 09:15:00 | 1891.70 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-10-28 14:30:00 | 1700.10 | 2025-10-30 09:15:00 | 1863.84 | TARGET_HIT | 1.00 | 9.63% |
| BUY | retest2 | 2025-10-29 09:30:00 | 1694.40 | 2025-10-30 10:15:00 | 1870.11 | TARGET_HIT | 1.00 | 10.37% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1732.60 | 2025-11-19 09:15:00 | 1798.10 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-11-18 15:00:00 | 1745.00 | 2025-11-19 09:15:00 | 1798.10 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-12-01 13:30:00 | 1757.70 | 2025-12-02 14:15:00 | 1746.50 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-01 14:15:00 | 1761.00 | 2025-12-02 14:15:00 | 1746.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-12-02 12:45:00 | 1759.30 | 2025-12-02 14:15:00 | 1746.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-04 09:15:00 | 1762.70 | 2025-12-04 10:15:00 | 1784.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-12-10 13:45:00 | 1651.80 | 2025-12-12 09:15:00 | 1674.20 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-12-10 14:30:00 | 1651.40 | 2025-12-12 09:15:00 | 1674.20 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-10 15:15:00 | 1650.00 | 2025-12-12 09:15:00 | 1674.20 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-12-11 09:30:00 | 1644.30 | 2025-12-12 09:15:00 | 1674.20 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-12 13:00:00 | 1660.00 | 2025-12-12 13:15:00 | 1672.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-12-19 09:15:00 | 1751.90 | 2025-12-23 15:15:00 | 1723.10 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-12-19 10:30:00 | 1736.40 | 2025-12-23 15:15:00 | 1723.10 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-19 12:30:00 | 1738.00 | 2025-12-23 15:15:00 | 1723.10 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1662.40 | 2026-01-01 13:15:00 | 1665.20 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-01-05 09:15:00 | 1679.10 | 2026-01-06 15:15:00 | 1671.10 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-01-08 10:15:00 | 1655.00 | 2026-01-12 09:15:00 | 1572.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:15:00 | 1655.00 | 2026-01-16 09:15:00 | 1543.80 | STOP_HIT | 0.50 | 6.72% |
| SELL | retest2 | 2026-03-12 09:15:00 | 1150.90 | 2026-03-16 09:15:00 | 1113.78 | PARTIAL | 0.50 | 3.23% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1172.40 | 2026-03-16 10:15:00 | 1111.78 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-03-13 09:45:00 | 1170.30 | 2026-03-16 10:15:00 | 1112.45 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2026-03-13 11:45:00 | 1171.00 | 2026-03-16 12:15:00 | 1093.36 | PARTIAL | 0.50 | 6.63% |
| SELL | retest2 | 2026-03-12 09:15:00 | 1150.90 | 2026-03-17 13:15:00 | 1120.50 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1172.40 | 2026-03-17 13:15:00 | 1120.50 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2026-03-13 09:45:00 | 1170.30 | 2026-03-17 13:15:00 | 1120.50 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2026-03-13 11:45:00 | 1171.00 | 2026-03-17 13:15:00 | 1120.50 | STOP_HIT | 0.50 | 4.31% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1189.20 | 2026-03-20 11:15:00 | 1168.90 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-04-21 09:15:00 | 1432.80 | 2026-04-24 10:15:00 | 1388.70 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-04-22 14:30:00 | 1419.60 | 2026-04-24 10:15:00 | 1388.70 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-04-23 09:15:00 | 1427.80 | 2026-04-24 10:15:00 | 1388.70 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1511.50 | 2026-05-04 15:15:00 | 1490.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-05-04 10:15:00 | 1505.20 | 2026-05-04 15:15:00 | 1490.00 | STOP_HIT | 1.00 | -1.01% |
