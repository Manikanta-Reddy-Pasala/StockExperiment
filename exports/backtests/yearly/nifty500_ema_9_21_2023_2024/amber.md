# Amber Enterprises India Ltd. (AMBER)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 8851.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 209 |
| ALERT1 | 145 |
| ALERT2 | 140 |
| ALERT2_SKIP | 71 |
| ALERT3 | 377 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 10 |
| ENTRY2 | 159 |
| PARTIAL | 14 |
| TARGET_HIT | 20 |
| STOP_HIT | 153 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 183 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 133
- **Target hits / Stop hits / Partials:** 20 / 149 / 14
- **Avg / median % per leg:** 0.43% / -1.13%
- **Sum % (uncompounded):** 79.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 92 | 20 | 21.7% | 12 | 79 | 1 | 0.14% | 12.6% |
| BUY @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 8 | 1 | -0.17% | -1.5% |
| BUY @ 3rd Alert (retest2) | 83 | 18 | 21.7% | 12 | 71 | 0 | 0.17% | 14.1% |
| SELL (all) | 91 | 30 | 33.0% | 8 | 70 | 13 | 0.73% | 66.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.73% | -5.5% |
| SELL @ 3rd Alert (retest2) | 89 | 30 | 33.7% | 8 | 68 | 13 | 0.81% | 72.0% |
| retest1 (combined) | 11 | 2 | 18.2% | 0 | 10 | 1 | -0.64% | -7.0% |
| retest2 (combined) | 172 | 48 | 27.9% | 20 | 139 | 13 | 0.50% | 86.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 1853.60 | 1829.46 | 1827.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 1930.30 | 1859.37 | 1843.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 14:15:00 | 1876.05 | 1895.00 | 1871.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-16 15:00:00 | 1876.05 | 1895.00 | 1871.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 15:15:00 | 1880.00 | 1892.00 | 1871.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-17 09:15:00 | 2149.40 | 1892.00 | 1871.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 11:15:00 | 2065.00 | 2075.28 | 2075.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 11:15:00 | 2065.00 | 2075.28 | 2075.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 13:15:00 | 2055.65 | 2069.45 | 2072.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 10:15:00 | 2068.00 | 2057.13 | 2064.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 10:15:00 | 2068.00 | 2057.13 | 2064.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 2068.00 | 2057.13 | 2064.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 10:30:00 | 2056.00 | 2057.13 | 2064.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 11:15:00 | 2101.30 | 2065.96 | 2067.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 12:00:00 | 2101.30 | 2065.96 | 2067.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 12:15:00 | 2130.80 | 2078.93 | 2073.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 09:15:00 | 2172.00 | 2134.31 | 2114.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 12:15:00 | 2136.00 | 2138.89 | 2122.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-26 13:00:00 | 2136.00 | 2138.89 | 2122.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 2125.15 | 2136.02 | 2123.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 15:00:00 | 2125.15 | 2136.02 | 2123.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 15:15:00 | 2124.00 | 2133.62 | 2123.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 09:15:00 | 2154.00 | 2133.62 | 2123.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 10:45:00 | 2131.70 | 2133.36 | 2125.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:15:00 | 2160.95 | 2129.28 | 2126.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 12:15:00 | 2135.00 | 2136.97 | 2135.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 2125.00 | 2134.58 | 2134.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 13:00:00 | 2125.00 | 2134.58 | 2134.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-05-31 13:15:00 | 2125.00 | 2132.66 | 2133.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 13:15:00 | 2125.00 | 2132.66 | 2133.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 14:15:00 | 2117.70 | 2129.67 | 2131.89 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 2162.00 | 2133.33 | 2133.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 11:15:00 | 2171.45 | 2142.58 | 2137.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 15:15:00 | 2145.10 | 2145.46 | 2140.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 15:15:00 | 2145.10 | 2145.46 | 2140.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 2145.10 | 2145.46 | 2140.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 2217.85 | 2145.46 | 2140.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-06 14:15:00 | 2163.50 | 2171.59 | 2172.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 14:15:00 | 2163.50 | 2171.59 | 2172.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-07 10:15:00 | 2148.95 | 2164.73 | 2168.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-09 11:15:00 | 2103.10 | 2091.01 | 2112.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 11:15:00 | 2103.10 | 2091.01 | 2112.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 11:15:00 | 2103.10 | 2091.01 | 2112.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-09 11:45:00 | 2105.10 | 2091.01 | 2112.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 12:15:00 | 2096.00 | 2092.01 | 2110.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-09 12:30:00 | 2105.95 | 2092.01 | 2110.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 2080.05 | 2087.80 | 2102.75 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 09:15:00 | 2136.00 | 2105.21 | 2102.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 09:15:00 | 2150.20 | 2135.14 | 2125.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 14:15:00 | 2127.90 | 2137.86 | 2131.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 14:15:00 | 2127.90 | 2137.86 | 2131.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 2127.90 | 2137.86 | 2131.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 14:45:00 | 2125.65 | 2137.86 | 2131.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 2128.00 | 2135.89 | 2131.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 09:15:00 | 2131.40 | 2135.89 | 2131.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 12:15:00 | 2114.00 | 2130.66 | 2130.27 | SL hit (close<static) qty=1.00 sl=2120.05 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 13:15:00 | 2115.15 | 2127.56 | 2128.89 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 09:15:00 | 2332.10 | 2162.22 | 2143.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 09:15:00 | 2418.90 | 2329.84 | 2276.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 12:15:00 | 2335.70 | 2342.44 | 2296.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 14:00:00 | 2369.00 | 2347.75 | 2303.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 2330.05 | 2343.87 | 2312.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:30:00 | 2323.05 | 2343.87 | 2312.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 12:15:00 | 2304.90 | 2332.77 | 2315.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-23 12:15:00 | 2304.90 | 2332.77 | 2315.18 | SL hit (close<ema400) qty=1.00 sl=2315.18 alert=retest1 |

### Cycle 10 — SELL (started 2023-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 15:15:00 | 2258.50 | 2298.99 | 2302.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 09:15:00 | 2246.90 | 2288.57 | 2297.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 2264.00 | 2252.68 | 2270.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 2264.00 | 2252.68 | 2270.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 2264.00 | 2252.68 | 2270.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:45:00 | 2257.00 | 2252.68 | 2270.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 2313.90 | 2264.92 | 2274.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:30:00 | 2317.00 | 2264.92 | 2274.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 2346.50 | 2281.24 | 2281.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:45:00 | 2369.95 | 2281.24 | 2281.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 12:15:00 | 2323.65 | 2289.72 | 2285.14 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 15:15:00 | 2263.00 | 2280.87 | 2282.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 12:15:00 | 2239.20 | 2263.37 | 2272.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 09:15:00 | 2238.60 | 2234.26 | 2253.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-30 09:45:00 | 2239.65 | 2234.26 | 2253.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 2258.00 | 2239.01 | 2254.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 11:00:00 | 2258.00 | 2239.01 | 2254.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 11:15:00 | 2266.45 | 2244.50 | 2255.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 14:15:00 | 2252.00 | 2252.43 | 2257.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 15:15:00 | 2250.00 | 2253.55 | 2257.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 12:30:00 | 2252.00 | 2255.53 | 2257.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 14:00:00 | 2252.80 | 2254.99 | 2256.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 2257.15 | 2255.42 | 2256.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 10:00:00 | 2248.55 | 2254.94 | 2256.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 09:15:00 | 2243.75 | 2248.93 | 2252.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 12:15:00 | 2269.25 | 2247.06 | 2244.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 12:15:00 | 2269.25 | 2247.06 | 2244.18 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 2214.95 | 2245.05 | 2246.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 13:15:00 | 2202.10 | 2223.56 | 2232.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 2224.90 | 2213.82 | 2225.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 2224.90 | 2213.82 | 2225.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 2224.90 | 2213.82 | 2225.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 2224.90 | 2213.82 | 2225.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 2206.40 | 2212.33 | 2223.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:30:00 | 2230.00 | 2212.33 | 2223.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 2219.00 | 2214.49 | 2220.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:45:00 | 2232.00 | 2214.49 | 2220.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 2219.00 | 2215.39 | 2220.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 2227.25 | 2215.39 | 2220.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 2227.85 | 2217.88 | 2221.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 10:15:00 | 2219.50 | 2217.88 | 2221.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 12:00:00 | 2225.15 | 2222.00 | 2222.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 13:15:00 | 2229.95 | 2224.64 | 2223.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 13:15:00 | 2229.95 | 2224.64 | 2223.96 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 10:15:00 | 2208.00 | 2221.73 | 2223.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 2190.00 | 2211.98 | 2217.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 09:15:00 | 2207.00 | 2204.26 | 2212.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 09:15:00 | 2207.00 | 2204.26 | 2212.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 2207.00 | 2204.26 | 2212.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 10:00:00 | 2207.00 | 2204.26 | 2212.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 2212.75 | 2205.96 | 2212.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 10:30:00 | 2220.10 | 2205.96 | 2212.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 2202.70 | 2205.31 | 2211.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:30:00 | 2212.30 | 2205.31 | 2211.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 2210.80 | 2205.65 | 2210.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:00:00 | 2210.80 | 2205.65 | 2210.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 2218.00 | 2208.12 | 2211.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 15:00:00 | 2218.00 | 2208.12 | 2211.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 2219.50 | 2210.40 | 2211.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 2220.65 | 2210.40 | 2211.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 2220.00 | 2212.32 | 2212.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 10:15:00 | 2213.25 | 2212.32 | 2212.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 11:45:00 | 2208.75 | 2211.68 | 2212.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 13:00:00 | 2212.15 | 2211.77 | 2212.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 13:15:00 | 2223.15 | 2214.05 | 2213.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 13:15:00 | 2223.15 | 2214.05 | 2213.29 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 2208.30 | 2213.23 | 2213.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 10:15:00 | 2201.10 | 2210.83 | 2212.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 11:15:00 | 2203.80 | 2197.91 | 2204.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 11:15:00 | 2203.80 | 2197.91 | 2204.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 2203.80 | 2197.91 | 2204.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:00:00 | 2203.80 | 2197.91 | 2204.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 2197.00 | 2197.73 | 2203.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 14:00:00 | 2196.25 | 2197.43 | 2202.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 15:15:00 | 2195.00 | 2197.52 | 2202.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 11:15:00 | 2221.05 | 2199.97 | 2201.69 | SL hit (close>static) qty=1.00 sl=2208.65 alert=retest2 |

### Cycle 19 — BUY (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 12:15:00 | 2225.50 | 2205.07 | 2203.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 09:15:00 | 2241.85 | 2222.48 | 2213.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 14:15:00 | 2265.60 | 2267.25 | 2248.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 14:15:00 | 2265.60 | 2267.25 | 2248.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 2265.60 | 2267.25 | 2248.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 14:45:00 | 2275.90 | 2267.25 | 2248.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 2233.20 | 2260.24 | 2248.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 10:30:00 | 2274.00 | 2269.88 | 2253.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-27 10:15:00 | 2501.40 | 2394.80 | 2333.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 2348.40 | 2404.20 | 2411.08 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 13:15:00 | 2419.90 | 2407.94 | 2407.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 2510.00 | 2428.56 | 2416.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 09:15:00 | 2469.60 | 2487.71 | 2462.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-07 09:30:00 | 2476.00 | 2487.71 | 2462.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 2457.45 | 2481.66 | 2461.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:00:00 | 2457.45 | 2481.66 | 2461.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 2463.85 | 2478.10 | 2462.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 12:30:00 | 2473.00 | 2479.49 | 2464.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 14:15:00 | 2490.00 | 2509.79 | 2511.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 14:15:00 | 2490.00 | 2509.79 | 2511.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 10:15:00 | 2462.95 | 2491.85 | 2502.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 11:15:00 | 2492.65 | 2492.01 | 2501.36 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 12:45:00 | 2454.00 | 2488.92 | 2499.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 13:30:00 | 2449.95 | 2483.33 | 2495.64 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 2475.40 | 2481.75 | 2493.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 15:00:00 | 2475.40 | 2481.75 | 2493.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 2518.95 | 2487.79 | 2494.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-14 09:15:00 | 2518.95 | 2487.79 | 2494.37 | SL hit (close>ema400) qty=1.00 sl=2494.37 alert=retest1 |

### Cycle 23 — BUY (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 11:15:00 | 2531.00 | 2502.14 | 2500.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 13:15:00 | 2571.85 | 2519.82 | 2508.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 14:15:00 | 2810.80 | 2814.03 | 2748.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-18 15:00:00 | 2810.80 | 2814.03 | 2748.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 2848.95 | 2857.86 | 2833.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 14:30:00 | 2831.05 | 2857.86 | 2833.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 2850.00 | 2853.99 | 2843.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:30:00 | 2843.35 | 2853.99 | 2843.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 2858.65 | 2878.17 | 2861.79 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 2800.00 | 2849.50 | 2851.67 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 2903.35 | 2841.93 | 2837.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 10:15:00 | 2916.00 | 2884.52 | 2863.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 15:15:00 | 2906.00 | 2908.27 | 2885.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:15:00 | 2942.95 | 2908.27 | 2885.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 12:45:00 | 2917.00 | 2912.53 | 2895.42 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 2907.70 | 2908.94 | 2896.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 14:30:00 | 2881.05 | 2908.94 | 2896.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 2895.00 | 2906.15 | 2896.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-30 15:15:00 | 2895.00 | 2906.15 | 2896.45 | SL hit (close<ema400) qty=1.00 sl=2896.45 alert=retest1 |

### Cycle 26 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 2993.25 | 3026.09 | 3028.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 2939.10 | 3001.09 | 3016.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 2883.05 | 2872.82 | 2916.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 2883.05 | 2872.82 | 2916.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 2883.05 | 2872.82 | 2916.09 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 2978.80 | 2918.70 | 2914.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 09:15:00 | 2997.45 | 2960.11 | 2938.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 13:15:00 | 2960.95 | 2972.80 | 2952.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 13:15:00 | 2960.95 | 2972.80 | 2952.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 2960.95 | 2972.80 | 2952.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 14:00:00 | 2960.95 | 2972.80 | 2952.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 2960.00 | 2970.24 | 2953.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 15:15:00 | 2976.05 | 2970.24 | 2953.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 2976.05 | 2971.40 | 2955.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 2942.50 | 2971.40 | 2955.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 2967.00 | 2970.52 | 2956.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:30:00 | 2945.00 | 2970.52 | 2956.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 12:15:00 | 2967.30 | 2970.96 | 2960.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 13:00:00 | 2967.30 | 2970.96 | 2960.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 2966.00 | 2969.97 | 2960.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 14:00:00 | 2966.00 | 2969.97 | 2960.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 2969.10 | 2969.80 | 2961.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 14:30:00 | 2970.25 | 2969.80 | 2961.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 2963.00 | 2968.44 | 2961.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 09:15:00 | 2974.65 | 2968.44 | 2961.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 2969.90 | 2968.73 | 2962.50 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 14:15:00 | 2932.45 | 2957.42 | 2959.34 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 15:15:00 | 2970.00 | 2955.90 | 2955.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 11:15:00 | 2980.80 | 2964.39 | 2959.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 14:15:00 | 2991.20 | 2993.11 | 2980.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 15:00:00 | 2991.20 | 2993.11 | 2980.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 3025.75 | 3000.73 | 2986.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 11:45:00 | 3052.65 | 3017.44 | 2996.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 12:45:00 | 3057.85 | 3024.95 | 3002.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 14:15:00 | 3057.10 | 3030.26 | 3006.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 14:45:00 | 3055.00 | 3035.17 | 3010.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 3003.60 | 3032.48 | 3016.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:00:00 | 3003.60 | 3032.48 | 3016.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 3021.55 | 3030.29 | 3016.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-28 15:15:00 | 2952.00 | 3004.88 | 3008.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 2952.00 | 3004.88 | 3008.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 10:15:00 | 2911.60 | 2968.96 | 2987.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 2972.00 | 2914.98 | 2930.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 2972.00 | 2914.98 | 2930.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 2972.00 | 2914.98 | 2930.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 09:45:00 | 2978.00 | 2914.98 | 2930.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 10:15:00 | 2949.35 | 2921.85 | 2932.18 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 14:15:00 | 2949.35 | 2938.89 | 2938.12 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 10:15:00 | 2918.00 | 2936.92 | 2937.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 11:15:00 | 2910.00 | 2931.53 | 2935.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 13:15:00 | 2845.00 | 2838.83 | 2863.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-10 14:00:00 | 2845.00 | 2838.83 | 2863.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 2864.20 | 2844.25 | 2859.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:30:00 | 2877.30 | 2844.25 | 2859.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 2866.45 | 2848.69 | 2860.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 11:00:00 | 2866.45 | 2848.69 | 2860.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 2896.85 | 2858.32 | 2863.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 11:30:00 | 2898.00 | 2858.32 | 2863.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 13:15:00 | 2896.85 | 2872.52 | 2869.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 3003.35 | 2903.48 | 2884.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 14:15:00 | 2977.95 | 2980.69 | 2954.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 15:00:00 | 2977.95 | 2980.69 | 2954.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 2948.15 | 2973.08 | 2955.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:45:00 | 2944.25 | 2973.08 | 2955.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 10:15:00 | 2933.05 | 2965.07 | 2953.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 10:30:00 | 2935.25 | 2965.07 | 2953.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2023-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 14:15:00 | 2936.90 | 2946.83 | 2947.38 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 09:15:00 | 2979.25 | 2951.58 | 2949.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-20 09:15:00 | 3040.00 | 2979.90 | 2971.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 09:15:00 | 2853.05 | 2993.73 | 2991.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 09:15:00 | 2853.05 | 2993.73 | 2991.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 2853.05 | 2993.73 | 2991.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:30:00 | 2822.00 | 2993.73 | 2991.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2023-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 10:15:00 | 2838.05 | 2962.59 | 2977.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 2819.35 | 2933.94 | 2962.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 2970.70 | 2873.75 | 2914.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 2970.70 | 2873.75 | 2914.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 2970.70 | 2873.75 | 2914.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 10:00:00 | 2970.70 | 2873.75 | 2914.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 2965.00 | 2892.00 | 2919.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 13:00:00 | 2936.05 | 2909.53 | 2922.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 13:30:00 | 2940.05 | 2919.54 | 2926.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-25 15:15:00 | 2959.05 | 2935.17 | 2932.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-25 15:15:00 | 2959.05 | 2935.17 | 2932.63 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-10-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 10:15:00 | 2912.00 | 2928.50 | 2929.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 11:15:00 | 2892.00 | 2921.20 | 2926.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 2934.00 | 2922.24 | 2925.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 14:15:00 | 2934.00 | 2922.24 | 2925.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 2934.00 | 2922.24 | 2925.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 15:00:00 | 2934.00 | 2922.24 | 2925.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 2940.00 | 2925.79 | 2926.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 2958.95 | 2925.79 | 2926.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 09:15:00 | 2943.25 | 2929.29 | 2928.33 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 10:15:00 | 2870.70 | 2917.57 | 2923.09 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 10:15:00 | 2961.90 | 2923.99 | 2921.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 11:15:00 | 2970.10 | 2933.21 | 2925.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 2933.00 | 2967.77 | 2956.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 14:15:00 | 2933.00 | 2967.77 | 2956.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 2933.00 | 2967.77 | 2956.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 15:00:00 | 2933.00 | 2967.77 | 2956.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 2927.00 | 2959.62 | 2953.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 09:15:00 | 2900.20 | 2959.62 | 2953.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 09:15:00 | 2901.10 | 2947.91 | 2948.73 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 10:15:00 | 2955.00 | 2949.33 | 2949.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-01 11:15:00 | 2955.80 | 2950.62 | 2949.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 12:15:00 | 2947.95 | 2950.09 | 2949.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 12:15:00 | 2947.95 | 2950.09 | 2949.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 2947.95 | 2950.09 | 2949.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:00:00 | 2947.95 | 2950.09 | 2949.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 2949.85 | 2950.04 | 2949.72 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 14:15:00 | 2903.25 | 2940.68 | 2945.50 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 11:15:00 | 2976.00 | 2942.95 | 2939.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 12:15:00 | 3016.00 | 2957.56 | 2946.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 14:15:00 | 3328.40 | 3374.15 | 3312.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-08 15:00:00 | 3328.40 | 3374.15 | 3312.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 15:15:00 | 3331.00 | 3365.52 | 3314.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:15:00 | 3270.00 | 3365.52 | 3314.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 3271.15 | 3346.65 | 3310.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:30:00 | 3299.00 | 3346.65 | 3310.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 3258.45 | 3329.01 | 3305.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:00:00 | 3258.45 | 3329.01 | 3305.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2023-11-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 13:15:00 | 3207.65 | 3283.89 | 3289.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 14:15:00 | 3122.30 | 3251.57 | 3274.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 3175.00 | 3101.08 | 3156.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 3175.00 | 3101.08 | 3156.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 3175.00 | 3101.08 | 3156.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 19:00:00 | 3175.00 | 3101.08 | 3156.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 3083.00 | 3097.46 | 3149.76 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 09:15:00 | 3140.25 | 3133.77 | 3132.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 11:15:00 | 3250.50 | 3162.82 | 3146.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 10:15:00 | 3225.95 | 3226.18 | 3192.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 15:15:00 | 3200.00 | 3219.95 | 3202.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 3200.00 | 3219.95 | 3202.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:15:00 | 3249.30 | 3219.95 | 3202.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 3244.30 | 3224.82 | 3206.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 10:45:00 | 3308.25 | 3239.85 | 3214.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-22 13:15:00 | 3246.15 | 3276.14 | 3277.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2023-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 13:15:00 | 3246.15 | 3276.14 | 3277.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 3190.55 | 3247.50 | 3259.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 10:15:00 | 3078.80 | 3073.12 | 3124.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-30 11:00:00 | 3078.80 | 3073.12 | 3124.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 11:15:00 | 3114.50 | 3074.73 | 3095.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 12:00:00 | 3114.50 | 3074.73 | 3095.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 12:15:00 | 3125.00 | 3084.79 | 3098.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 12:30:00 | 3129.00 | 3084.79 | 3098.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2023-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 15:15:00 | 3112.50 | 3107.27 | 3106.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 09:15:00 | 3159.00 | 3122.37 | 3114.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 10:15:00 | 3114.35 | 3120.76 | 3114.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 10:15:00 | 3114.35 | 3120.76 | 3114.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 10:15:00 | 3114.35 | 3120.76 | 3114.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:00:00 | 3114.35 | 3120.76 | 3114.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 3115.00 | 3119.61 | 3114.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 3108.10 | 3119.61 | 3114.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 3137.45 | 3123.18 | 3116.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 13:30:00 | 3158.20 | 3128.54 | 3119.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 10:15:00 | 3107.85 | 3125.78 | 3121.88 | SL hit (close<static) qty=1.00 sl=3110.80 alert=retest2 |

### Cycle 50 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 3101.20 | 3118.77 | 3119.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 13:15:00 | 3099.45 | 3114.91 | 3117.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 11:15:00 | 3111.80 | 3103.55 | 3110.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 11:15:00 | 3111.80 | 3103.55 | 3110.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 11:15:00 | 3111.80 | 3103.55 | 3110.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 12:00:00 | 3111.80 | 3103.55 | 3110.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 12:15:00 | 3092.55 | 3101.35 | 3108.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 12:45:00 | 3105.00 | 3101.35 | 3108.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 14:15:00 | 3091.85 | 3096.99 | 3105.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 14:30:00 | 3101.25 | 3096.99 | 3105.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 3092.60 | 3095.31 | 3102.84 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 3127.70 | 3104.79 | 3104.43 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 11:15:00 | 3092.20 | 3103.71 | 3104.09 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 14:15:00 | 3140.00 | 3109.13 | 3106.24 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 3082.25 | 3114.56 | 3115.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 3059.15 | 3103.48 | 3109.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 3139.70 | 3087.36 | 3094.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 3139.70 | 3087.36 | 3094.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 3139.70 | 3087.36 | 3094.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:00:00 | 3139.70 | 3087.36 | 3094.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 3108.70 | 3091.63 | 3095.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 11:15:00 | 3100.95 | 3091.63 | 3095.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-14 12:15:00 | 3155.00 | 3108.78 | 3103.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 12:15:00 | 3155.00 | 3108.78 | 3103.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 3308.85 | 3156.23 | 3127.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 09:15:00 | 3245.85 | 3278.35 | 3219.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 10:00:00 | 3245.85 | 3278.35 | 3219.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 3225.00 | 3260.10 | 3221.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 12:00:00 | 3225.00 | 3260.10 | 3221.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 3234.10 | 3254.90 | 3222.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 12:45:00 | 3232.20 | 3254.90 | 3222.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 13:15:00 | 3252.80 | 3254.48 | 3225.24 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2023-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 13:15:00 | 3196.10 | 3214.30 | 3216.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 10:15:00 | 3177.50 | 3204.85 | 3210.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 12:15:00 | 3098.25 | 3097.50 | 3138.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 13:00:00 | 3098.25 | 3097.50 | 3138.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 3130.40 | 3064.47 | 3072.99 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 3149.00 | 3081.37 | 3079.90 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 3095.60 | 3114.52 | 3116.67 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 15:15:00 | 3120.00 | 3114.75 | 3114.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 3162.05 | 3124.21 | 3118.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 3300.00 | 3320.23 | 3269.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 15:00:00 | 3300.00 | 3320.23 | 3269.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 3507.00 | 3526.17 | 3479.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 09:45:00 | 3554.45 | 3531.78 | 3485.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-18 11:15:00 | 3909.89 | 3768.65 | 3743.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 09:15:00 | 3697.70 | 3760.81 | 3765.13 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 13:15:00 | 3789.70 | 3751.77 | 3749.92 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 3618.45 | 3725.10 | 3737.97 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 3820.15 | 3734.55 | 3727.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 10:15:00 | 3875.15 | 3787.23 | 3755.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 15:15:00 | 4347.60 | 4448.81 | 4354.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 15:15:00 | 4347.60 | 4448.81 | 4354.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 4347.60 | 4448.81 | 4354.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:15:00 | 4276.00 | 4448.81 | 4354.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 4218.35 | 4402.72 | 4341.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:45:00 | 4184.60 | 4402.72 | 4341.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 4237.80 | 4369.74 | 4332.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 11:15:00 | 4211.90 | 4369.74 | 4332.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 4192.15 | 4286.47 | 4299.28 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 14:15:00 | 4562.10 | 4341.59 | 4323.18 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 13:15:00 | 4416.80 | 4442.24 | 4442.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 11:15:00 | 4335.00 | 4406.80 | 4424.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 3613.20 | 3553.12 | 3746.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 15:00:00 | 3613.20 | 3553.12 | 3746.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 3700.00 | 3582.50 | 3741.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 3581.05 | 3582.50 | 3741.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:45:00 | 3585.00 | 3583.40 | 3727.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 13:15:00 | 3577.40 | 3589.34 | 3694.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 14:15:00 | 3856.80 | 3665.32 | 3712.16 | SL hit (close>static) qty=1.00 sl=3743.80 alert=retest2 |

### Cycle 67 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 3941.90 | 3743.16 | 3740.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 4034.15 | 3904.49 | 3837.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 14:15:00 | 3944.70 | 3950.82 | 3890.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 15:00:00 | 3944.70 | 3950.82 | 3890.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 4076.95 | 3972.29 | 3914.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 4115.00 | 4030.81 | 3969.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 3912.05 | 3997.94 | 4000.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 3912.05 | 3997.94 | 4000.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 14:15:00 | 3880.15 | 3940.93 | 3966.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 14:15:00 | 3720.65 | 3695.25 | 3747.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 14:15:00 | 3720.65 | 3695.25 | 3747.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 14:15:00 | 3720.65 | 3695.25 | 3747.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 14:45:00 | 3771.20 | 3695.25 | 3747.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 3594.00 | 3678.54 | 3730.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 10:45:00 | 3571.00 | 3661.64 | 3718.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 15:15:00 | 3573.00 | 3615.67 | 3652.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 09:15:00 | 3573.50 | 3634.27 | 3649.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-04 12:15:00 | 3693.95 | 3659.65 | 3657.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 12:15:00 | 3693.95 | 3659.65 | 3657.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 10:15:00 | 3773.00 | 3706.14 | 3683.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 12:15:00 | 3704.05 | 3708.11 | 3688.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 13:00:00 | 3704.05 | 3708.11 | 3688.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 3728.00 | 3710.40 | 3693.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 09:30:00 | 3679.50 | 3702.09 | 3691.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 3647.30 | 3691.13 | 3687.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:00:00 | 3647.30 | 3691.13 | 3687.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 3639.65 | 3680.83 | 3683.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 09:15:00 | 3592.15 | 3648.31 | 3662.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 12:15:00 | 3720.00 | 3652.44 | 3659.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 12:15:00 | 3720.00 | 3652.44 | 3659.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 12:15:00 | 3720.00 | 3652.44 | 3659.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 12:45:00 | 3707.60 | 3652.44 | 3659.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 3678.60 | 3657.67 | 3661.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 13:30:00 | 3722.15 | 3657.67 | 3661.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 3625.50 | 3651.24 | 3658.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 15:15:00 | 3615.00 | 3651.24 | 3658.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 10:15:00 | 3605.00 | 3650.39 | 3656.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:15:00 | 3434.25 | 3536.08 | 3586.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:15:00 | 3424.75 | 3536.08 | 3586.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-13 14:15:00 | 3554.00 | 3512.80 | 3561.07 | SL hit (close>ema200) qty=0.50 sl=3512.80 alert=retest2 |

### Cycle 71 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 3327.90 | 3256.76 | 3255.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 3375.20 | 3280.45 | 3266.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 3479.00 | 3499.27 | 3422.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 12:00:00 | 3576.50 | 3518.37 | 3444.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 09:15:00 | 3755.33 | 3659.07 | 3617.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-01 13:15:00 | 3660.20 | 3670.89 | 3637.50 | SL hit (close<ema200) qty=0.50 sl=3670.89 alert=retest1 |

### Cycle 72 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 3745.10 | 3789.40 | 3789.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 13:15:00 | 3717.95 | 3748.81 | 3767.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 10:15:00 | 3647.50 | 3633.23 | 3667.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 11:00:00 | 3647.50 | 3633.23 | 3667.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 3647.25 | 3629.58 | 3649.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:45:00 | 3636.20 | 3630.89 | 3648.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 11:45:00 | 3638.40 | 3631.96 | 3647.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 3622.35 | 3633.57 | 3646.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 3685.60 | 3643.41 | 3637.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 3685.60 | 3643.41 | 3637.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 13:15:00 | 3709.95 | 3678.13 | 3658.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-22 14:15:00 | 3651.90 | 3672.89 | 3657.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 14:15:00 | 3651.90 | 3672.89 | 3657.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 3651.90 | 3672.89 | 3657.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 15:00:00 | 3651.90 | 3672.89 | 3657.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 15:15:00 | 3660.00 | 3670.31 | 3657.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 09:15:00 | 3747.85 | 3670.31 | 3657.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 09:15:00 | 3740.35 | 3785.18 | 3785.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 3740.35 | 3785.18 | 3785.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 12:15:00 | 3725.00 | 3760.97 | 3773.20 | Break + close below crossover candle low |

### Cycle 75 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 3917.20 | 3782.44 | 3777.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 13:15:00 | 3947.35 | 3911.35 | 3883.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 3992.40 | 4031.55 | 3988.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 11:00:00 | 3992.40 | 4031.55 | 3988.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 3979.00 | 4021.04 | 3987.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:30:00 | 3980.00 | 4021.04 | 3987.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 3935.25 | 4003.88 | 3982.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 13:00:00 | 3935.25 | 4003.88 | 3982.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 3940.20 | 3991.15 | 3978.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 14:00:00 | 3940.20 | 3991.15 | 3978.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 3985.00 | 3987.96 | 3979.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:15:00 | 3953.85 | 3987.96 | 3979.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 3935.00 | 3977.37 | 3975.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:00:00 | 3935.00 | 3977.37 | 3975.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 10:15:00 | 3915.00 | 3964.89 | 3969.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 11:15:00 | 3878.90 | 3947.69 | 3961.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-09 09:15:00 | 4021.00 | 3937.58 | 3947.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 4021.00 | 3937.58 | 3947.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 4021.00 | 3937.58 | 3947.42 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 13:15:00 | 3981.00 | 3953.34 | 3952.78 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 14:15:00 | 3866.65 | 3936.00 | 3944.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 09:15:00 | 3839.30 | 3907.70 | 3929.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 14:15:00 | 3800.00 | 3779.32 | 3823.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 14:45:00 | 3796.20 | 3779.32 | 3823.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 3765.50 | 3778.64 | 3815.80 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 3891.25 | 3819.77 | 3810.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 3951.90 | 3846.20 | 3823.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 3855.00 | 3932.67 | 3898.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 3855.00 | 3932.67 | 3898.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 3855.00 | 3932.67 | 3898.78 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 3815.00 | 3868.97 | 3875.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 15:15:00 | 3809.00 | 3847.70 | 3864.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 3825.95 | 3810.32 | 3834.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 14:15:00 | 3825.95 | 3810.32 | 3834.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 3825.95 | 3810.32 | 3834.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 3825.95 | 3810.32 | 3834.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 3837.60 | 3817.65 | 3833.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 3853.05 | 3817.65 | 3833.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 3837.80 | 3821.68 | 3833.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:45:00 | 3844.45 | 3821.68 | 3833.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 3842.35 | 3825.81 | 3834.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:30:00 | 3845.05 | 3825.81 | 3834.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 3839.35 | 3828.52 | 3835.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 15:00:00 | 3813.00 | 3828.48 | 3834.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:15:00 | 3622.35 | 3707.06 | 3747.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-31 09:15:00 | 3629.65 | 3558.68 | 3590.84 | SL hit (close>ema200) qty=0.50 sl=3558.68 alert=retest2 |

### Cycle 81 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 3721.65 | 3621.65 | 3611.22 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 3464.00 | 3634.47 | 3639.70 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 3669.50 | 3625.49 | 3624.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 3726.35 | 3650.42 | 3636.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 3733.20 | 3744.84 | 3713.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 14:45:00 | 3750.00 | 3744.84 | 3713.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 4045.15 | 4067.04 | 4023.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:45:00 | 4061.55 | 4067.04 | 4023.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 4029.50 | 4059.53 | 4023.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 3999.20 | 4059.53 | 4023.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 4057.90 | 4059.20 | 4026.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 3989.05 | 4059.20 | 4026.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 4183.90 | 4084.14 | 4041.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 12:15:00 | 4207.00 | 4105.91 | 4054.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 15:15:00 | 3989.00 | 4107.27 | 4074.70 | SL hit (close<static) qty=1.00 sl=4036.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 3994.00 | 4053.80 | 4055.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 3982.55 | 4039.55 | 4048.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 11:15:00 | 4035.05 | 4008.19 | 4025.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 11:15:00 | 4035.05 | 4008.19 | 4025.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 4035.05 | 4008.19 | 4025.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 4035.05 | 4008.19 | 4025.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 4030.00 | 4012.55 | 4026.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 14:45:00 | 4023.00 | 4010.82 | 4023.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:00:00 | 3994.60 | 4005.85 | 4018.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 13:00:00 | 4017.95 | 4011.70 | 4018.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 4019.20 | 4015.84 | 4019.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 3968.30 | 4006.33 | 4014.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 4089.45 | 4016.82 | 4017.81 | SL hit (close>static) qty=1.00 sl=4040.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 4065.65 | 4026.59 | 4022.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 14:15:00 | 4096.10 | 4060.67 | 4041.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 13:15:00 | 4092.50 | 4093.33 | 4069.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 13:30:00 | 4085.00 | 4093.33 | 4069.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 4053.60 | 4085.38 | 4067.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 4053.60 | 4085.38 | 4067.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 4054.55 | 4079.22 | 4066.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 4122.30 | 4079.22 | 4066.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-28 11:15:00 | 4534.53 | 4357.39 | 4275.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 4552.10 | 4610.75 | 4611.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 4481.05 | 4584.81 | 4599.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 13:15:00 | 4222.00 | 4219.27 | 4301.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 13:30:00 | 4230.50 | 4219.27 | 4301.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 4282.30 | 4231.88 | 4299.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:45:00 | 4305.00 | 4231.88 | 4299.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 4323.15 | 4232.25 | 4257.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 4308.05 | 4232.25 | 4257.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 4315.40 | 4248.88 | 4262.81 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 4317.75 | 4274.35 | 4272.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 4367.90 | 4307.77 | 4290.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 4368.55 | 4417.79 | 4378.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 14:15:00 | 4368.55 | 4417.79 | 4378.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 4368.55 | 4417.79 | 4378.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 14:45:00 | 4378.00 | 4417.79 | 4378.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 4400.00 | 4414.23 | 4380.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 4464.65 | 4414.23 | 4380.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 10:45:00 | 4420.00 | 4415.48 | 4386.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 4416.00 | 4399.40 | 4389.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 14:15:00 | 4343.00 | 4383.77 | 4386.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 4343.00 | 4383.77 | 4386.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 4330.00 | 4356.62 | 4369.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 09:15:00 | 4369.00 | 4303.05 | 4325.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 4369.00 | 4303.05 | 4325.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 4369.00 | 4303.05 | 4325.64 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 11:15:00 | 4525.00 | 4358.12 | 4347.41 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 4158.00 | 4328.03 | 4343.85 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 4333.15 | 4285.27 | 4280.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 4338.35 | 4295.89 | 4285.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 15:15:00 | 4325.00 | 4328.15 | 4308.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 15:15:00 | 4325.00 | 4328.15 | 4308.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 4325.00 | 4328.15 | 4308.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 4307.00 | 4324.68 | 4308.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 4311.55 | 4322.05 | 4308.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 12:30:00 | 4322.25 | 4319.11 | 4309.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 14:45:00 | 4320.00 | 4322.96 | 4313.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 4179.00 | 4293.38 | 4301.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 4179.00 | 4293.38 | 4301.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 11:15:00 | 4158.00 | 4245.77 | 4277.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 13:15:00 | 4062.20 | 4052.28 | 4112.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 14:00:00 | 4062.20 | 4052.28 | 4112.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 4068.60 | 4055.55 | 4108.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 15:00:00 | 4068.60 | 4055.55 | 4108.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 4078.75 | 4060.90 | 4101.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:45:00 | 4064.20 | 4061.14 | 4098.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 13:15:00 | 4119.20 | 4081.28 | 4098.84 | SL hit (close>static) qty=1.00 sl=4117.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 4168.55 | 4115.09 | 4111.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 4192.55 | 4139.01 | 4123.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 4131.20 | 4158.30 | 4141.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 4131.20 | 4158.30 | 4141.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 4131.20 | 4158.30 | 4141.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 4131.20 | 4158.30 | 4141.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 4095.10 | 4145.66 | 4137.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 4095.10 | 4145.66 | 4137.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 4118.00 | 4140.13 | 4135.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 12:15:00 | 4127.10 | 4140.13 | 4135.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-29 10:15:00 | 4539.81 | 4413.59 | 4349.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 4449.40 | 4521.89 | 4529.06 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 4497.60 | 4458.78 | 4457.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 4616.20 | 4499.24 | 4476.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 4548.90 | 4586.32 | 4543.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 4548.90 | 4586.32 | 4543.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 4548.90 | 4586.32 | 4543.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 4548.90 | 4586.32 | 4543.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 4525.00 | 4574.06 | 4541.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 4525.00 | 4574.06 | 4541.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 4516.30 | 4562.51 | 4539.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 4516.30 | 4562.51 | 4539.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 4550.00 | 4550.13 | 4539.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 4515.95 | 4550.13 | 4539.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 4505.75 | 4541.26 | 4536.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 4505.75 | 4541.26 | 4536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 4486.35 | 4530.27 | 4531.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 4478.25 | 4507.87 | 4518.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 4400.00 | 4372.01 | 4423.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 4400.00 | 4372.01 | 4423.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 4377.95 | 4373.20 | 4418.97 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 4545.10 | 4407.25 | 4405.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 11:15:00 | 4636.00 | 4453.00 | 4426.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 4757.95 | 4871.62 | 4729.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 14:00:00 | 4757.95 | 4871.62 | 4729.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 4721.05 | 4841.50 | 4728.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:30:00 | 4695.25 | 4841.50 | 4728.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 4740.50 | 4821.30 | 4729.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 4692.95 | 4795.63 | 4726.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 4697.05 | 4775.92 | 4723.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 4654.15 | 4775.92 | 4723.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 4820.95 | 4767.48 | 4730.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:45:00 | 4731.00 | 4767.48 | 4730.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 4783.85 | 4782.55 | 4747.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 4731.00 | 4782.55 | 4747.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 4741.65 | 4774.37 | 4747.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 4741.65 | 4774.37 | 4747.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 4763.75 | 4772.24 | 4748.83 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 14:15:00 | 4660.00 | 4737.69 | 4737.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 09:15:00 | 4646.65 | 4712.34 | 4725.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 13:15:00 | 4700.05 | 4661.23 | 4692.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 13:15:00 | 4700.05 | 4661.23 | 4692.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 4700.05 | 4661.23 | 4692.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 4700.05 | 4661.23 | 4692.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 4652.85 | 4659.56 | 4688.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 4619.00 | 4659.56 | 4688.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 14:15:00 | 4812.80 | 4693.18 | 4690.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 14:15:00 | 4812.80 | 4693.18 | 4690.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 10:15:00 | 4857.00 | 4753.74 | 4721.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 5018.15 | 5054.68 | 4974.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 13:00:00 | 5018.15 | 5054.68 | 4974.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 4921.90 | 5021.71 | 4972.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:30:00 | 4916.05 | 5021.71 | 4972.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 4902.00 | 4997.77 | 4966.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 5049.00 | 4997.77 | 4966.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 4874.85 | 4968.06 | 4957.89 | SL hit (close<static) qty=1.00 sl=4890.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 12:15:00 | 4875.90 | 4945.43 | 4949.10 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 09:15:00 | 5084.00 | 4962.93 | 4954.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 14:15:00 | 5155.55 | 5104.15 | 5054.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 5098.00 | 5125.17 | 5086.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 12:30:00 | 5114.60 | 5125.17 | 5086.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 5125.70 | 5124.17 | 5092.50 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 5060.00 | 5083.87 | 5083.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 14:15:00 | 5020.45 | 5071.18 | 5078.12 | Break + close below crossover candle low |

### Cycle 103 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 5265.45 | 5100.26 | 5089.53 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 5067.75 | 5096.06 | 5097.59 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 5158.15 | 5107.48 | 5102.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 5223.00 | 5138.51 | 5117.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 09:15:00 | 5427.75 | 5451.12 | 5375.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 5427.75 | 5451.12 | 5375.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 5427.75 | 5451.12 | 5375.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 5738.85 | 5413.86 | 5387.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-21 14:15:00 | 6312.74 | 5950.02 | 5700.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 12:15:00 | 5994.20 | 6190.70 | 6197.39 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 6280.45 | 6193.82 | 6188.59 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 10:15:00 | 6056.80 | 6200.39 | 6213.73 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 6104.30 | 6067.26 | 6063.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 6196.70 | 6100.95 | 6079.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 6218.80 | 6224.51 | 6171.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 6218.80 | 6224.51 | 6171.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 6211.80 | 6226.41 | 6189.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 6211.80 | 6226.41 | 6189.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 6219.65 | 6222.27 | 6193.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:30:00 | 6206.05 | 6222.27 | 6193.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 6138.00 | 6202.34 | 6189.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 6033.35 | 6202.34 | 6189.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 6109.40 | 6183.75 | 6182.20 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 10:15:00 | 6113.75 | 6169.75 | 6175.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 15:15:00 | 6057.00 | 6116.24 | 6145.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 6137.30 | 6120.45 | 6144.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 6137.30 | 6120.45 | 6144.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 6137.30 | 6120.45 | 6144.44 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 13:15:00 | 6211.35 | 6158.79 | 6155.93 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 6110.25 | 6149.08 | 6151.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 5978.35 | 6110.28 | 6133.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 10:15:00 | 6129.25 | 6114.07 | 6133.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 10:15:00 | 6129.25 | 6114.07 | 6133.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 6129.25 | 6114.07 | 6133.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:45:00 | 6121.85 | 6114.07 | 6133.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 6126.10 | 6116.48 | 6132.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 12:15:00 | 6141.80 | 6116.48 | 6132.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 12:15:00 | 6124.40 | 6118.06 | 6131.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:30:00 | 6066.50 | 6108.40 | 6124.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 15:00:00 | 6067.60 | 6108.40 | 6124.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 6075.20 | 6106.72 | 6122.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 6205.90 | 6131.65 | 6128.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 09:15:00 | 6205.90 | 6131.65 | 6128.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 11:15:00 | 6313.55 | 6180.73 | 6152.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 15:15:00 | 6176.00 | 6199.79 | 6172.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 15:15:00 | 6176.00 | 6199.79 | 6172.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 6176.00 | 6199.79 | 6172.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 6244.85 | 6199.79 | 6172.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 14:15:00 | 6153.00 | 6231.43 | 6206.31 | SL hit (close<static) qty=1.00 sl=6165.90 alert=retest2 |

### Cycle 114 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 6475.00 | 6494.15 | 6495.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 09:15:00 | 6350.00 | 6458.12 | 6477.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 15:15:00 | 6075.00 | 6071.89 | 6174.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 09:15:00 | 6160.00 | 6071.89 | 6174.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 6127.65 | 6083.04 | 6170.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 6050.65 | 6082.56 | 6137.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 10:15:00 | 6013.05 | 6073.20 | 6123.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:15:00 | 5748.12 | 5780.53 | 5830.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:15:00 | 5712.40 | 5780.53 | 5830.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-10 12:15:00 | 5772.50 | 5766.84 | 5810.95 | SL hit (close>ema200) qty=0.50 sl=5766.84 alert=retest2 |

### Cycle 115 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 5797.35 | 5756.12 | 5755.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 6050.00 | 5814.90 | 5782.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 15:15:00 | 6000.00 | 6002.62 | 5944.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 09:15:00 | 5940.55 | 6002.62 | 5944.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 5994.55 | 6001.01 | 5949.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 12:00:00 | 6175.05 | 6035.03 | 5995.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-23 13:15:00 | 6792.56 | 6434.75 | 6254.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 11:15:00 | 7638.70 | 7788.88 | 7789.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 12:15:00 | 7589.60 | 7749.02 | 7771.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 15:15:00 | 7765.75 | 7729.04 | 7754.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 15:15:00 | 7765.75 | 7729.04 | 7754.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 7765.75 | 7729.04 | 7754.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 7807.50 | 7729.04 | 7754.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 7928.50 | 7768.94 | 7770.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 7928.50 | 7768.94 | 7770.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 10:15:00 | 8050.95 | 7825.34 | 7795.99 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 7590.95 | 7783.87 | 7792.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 10:15:00 | 7457.35 | 7718.57 | 7761.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 11:15:00 | 7005.00 | 6973.67 | 7168.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 11:45:00 | 7059.45 | 6973.67 | 7168.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 7002.90 | 7007.09 | 7115.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:45:00 | 6967.45 | 6970.25 | 7079.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 10:45:00 | 6935.85 | 6842.03 | 6900.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 14:15:00 | 7058.95 | 6949.35 | 6938.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 7058.95 | 6949.35 | 6938.42 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 6688.70 | 6907.24 | 6921.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 6666.60 | 6859.11 | 6898.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 6790.70 | 6456.69 | 6570.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 6790.70 | 6456.69 | 6570.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 6790.70 | 6456.69 | 6570.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 6847.35 | 6456.69 | 6570.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 6916.00 | 6548.55 | 6601.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 6929.60 | 6548.55 | 6601.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 6913.00 | 6681.95 | 6656.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 10:15:00 | 7046.05 | 6816.42 | 6735.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 6655.00 | 6874.96 | 6815.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 6655.00 | 6874.96 | 6815.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 6655.00 | 6874.96 | 6815.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:00:00 | 6655.00 | 6874.96 | 6815.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 6628.65 | 6825.69 | 6798.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:30:00 | 6607.15 | 6825.69 | 6798.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 12:15:00 | 6682.50 | 6772.25 | 6777.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 6649.15 | 6747.63 | 6765.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 6732.95 | 6594.21 | 6662.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 6732.95 | 6594.21 | 6662.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 6732.95 | 6594.21 | 6662.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 6732.95 | 6594.21 | 6662.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 6748.45 | 6625.06 | 6670.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:45:00 | 6774.50 | 6625.06 | 6670.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 6687.70 | 6637.58 | 6671.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:15:00 | 6635.00 | 6637.58 | 6671.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 7098.10 | 6729.27 | 6707.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 7098.10 | 6729.27 | 6707.53 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 13:15:00 | 6592.65 | 6695.22 | 6699.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 09:15:00 | 6507.20 | 6630.70 | 6666.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 6483.00 | 6443.28 | 6529.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 10:45:00 | 6498.60 | 6443.28 | 6529.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 6521.50 | 6458.92 | 6528.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:45:00 | 6506.00 | 6458.92 | 6528.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 6478.95 | 6462.93 | 6523.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 13:15:00 | 6462.25 | 6462.93 | 6523.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 6626.35 | 6500.34 | 6521.63 | SL hit (close>static) qty=1.00 sl=6529.65 alert=retest2 |

### Cycle 125 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 6619.95 | 6548.71 | 6540.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 14:15:00 | 6669.00 | 6572.77 | 6552.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 6561.15 | 6573.20 | 6556.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 6561.15 | 6573.20 | 6556.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 6561.15 | 6573.20 | 6556.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 6728.00 | 6613.66 | 6576.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:30:00 | 6720.75 | 6643.73 | 6613.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 6628.10 | 6846.68 | 6865.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 6628.10 | 6846.68 | 6865.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 6583.00 | 6793.95 | 6839.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 6209.95 | 6146.46 | 6350.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 10:00:00 | 6209.95 | 6146.46 | 6350.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 5643.00 | 5545.52 | 5609.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:45:00 | 5671.00 | 5545.52 | 5609.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 5684.15 | 5573.24 | 5616.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:30:00 | 5663.95 | 5573.24 | 5616.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 5736.05 | 5605.80 | 5626.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:45:00 | 5728.15 | 5605.80 | 5626.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 5728.00 | 5658.03 | 5648.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 5840.00 | 5694.43 | 5666.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 5763.70 | 5892.47 | 5816.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 5763.70 | 5892.47 | 5816.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 5763.70 | 5892.47 | 5816.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 5763.70 | 5892.47 | 5816.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 5808.90 | 5875.75 | 5815.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 5828.75 | 5875.75 | 5815.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 5755.70 | 5934.88 | 5946.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 5755.70 | 5934.88 | 5946.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 5692.00 | 5857.08 | 5907.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 5701.05 | 5617.98 | 5711.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 14:15:00 | 5701.05 | 5617.98 | 5711.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 5701.05 | 5617.98 | 5711.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 5701.05 | 5617.98 | 5711.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 5619.00 | 5618.18 | 5703.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 5649.90 | 5618.18 | 5703.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 5648.60 | 5624.26 | 5698.44 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 5766.80 | 5729.84 | 5726.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 5806.25 | 5752.65 | 5738.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 15:15:00 | 6375.00 | 6407.79 | 6286.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-11 09:15:00 | 6223.70 | 6407.79 | 6286.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 6282.95 | 6382.82 | 6286.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 6193.30 | 6382.82 | 6286.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 6334.75 | 6373.21 | 6290.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:45:00 | 6422.95 | 6372.31 | 6309.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 15:00:00 | 6418.25 | 6381.50 | 6319.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:45:00 | 6409.80 | 6389.76 | 6334.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 12:30:00 | 6440.05 | 6406.39 | 6355.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 6388.30 | 6434.77 | 6398.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:45:00 | 6378.00 | 6434.77 | 6398.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 6393.90 | 6426.59 | 6397.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:15:00 | 6394.10 | 6426.59 | 6397.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 6388.00 | 6418.88 | 6396.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:15:00 | 6359.55 | 6418.88 | 6396.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 6370.05 | 6409.11 | 6394.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:45:00 | 6364.30 | 6409.11 | 6394.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 6353.45 | 6397.98 | 6390.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 6542.05 | 6397.98 | 6390.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-20 09:15:00 | 7065.25 | 6858.88 | 6720.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 6865.90 | 6939.45 | 6948.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 6826.95 | 6899.32 | 6925.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 6874.85 | 6859.64 | 6887.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 6874.85 | 6859.64 | 6887.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 6874.85 | 6859.64 | 6887.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 6869.95 | 6859.64 | 6887.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 6890.40 | 6865.79 | 6888.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 6886.05 | 6865.79 | 6888.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 6933.40 | 6879.31 | 6892.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:30:00 | 6876.00 | 6879.31 | 6892.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 6961.30 | 6895.71 | 6898.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 6961.30 | 6895.71 | 6898.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 7138.50 | 6944.27 | 6920.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 7297.85 | 7043.10 | 6971.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 6905.05 | 7117.32 | 7062.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 6905.05 | 7117.32 | 7062.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 6905.05 | 7117.32 | 7062.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 6905.05 | 7117.32 | 7062.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 6885.35 | 7070.92 | 7046.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:45:00 | 6887.50 | 7070.92 | 7046.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 12:15:00 | 6894.50 | 7007.43 | 7020.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 15:15:00 | 6865.00 | 6948.07 | 6987.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 6949.55 | 6948.36 | 6983.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 6949.55 | 6948.36 | 6983.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 6949.55 | 6948.36 | 6983.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:00:00 | 6949.55 | 6948.36 | 6983.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 6967.60 | 6953.22 | 6979.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 6967.60 | 6953.22 | 6979.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 6982.35 | 6959.05 | 6980.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 6982.35 | 6959.05 | 6980.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 6972.10 | 6961.66 | 6979.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 6992.35 | 6961.66 | 6979.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 6996.50 | 6968.63 | 6981.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 6996.50 | 6968.63 | 6981.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 6982.00 | 6971.30 | 6981.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 7005.00 | 6971.30 | 6981.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 6959.25 | 6967.88 | 6977.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:30:00 | 6971.95 | 6967.88 | 6977.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 6913.25 | 6939.81 | 6959.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 15:00:00 | 6913.25 | 6939.81 | 6959.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 6736.00 | 6898.28 | 6937.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 6691.75 | 6898.28 | 6937.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 11:00:00 | 6713.60 | 6861.34 | 6917.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 6022.57 | 6580.07 | 6740.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 6532.70 | 6393.40 | 6378.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 6856.00 | 6537.71 | 6455.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 6765.00 | 6791.73 | 6667.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 6765.00 | 6791.73 | 6667.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 6723.00 | 6763.33 | 6711.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:30:00 | 6705.50 | 6763.33 | 6711.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 6748.00 | 6760.27 | 6714.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:45:00 | 6775.00 | 6739.84 | 6719.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-22 09:15:00 | 6628.00 | 6737.66 | 6734.48 | SL hit (close<static) qty=1.00 sl=6701.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 10:15:00 | 6625.50 | 6715.23 | 6724.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 12:15:00 | 6603.00 | 6677.70 | 6704.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 10:15:00 | 6673.00 | 6657.90 | 6682.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 10:15:00 | 6673.00 | 6657.90 | 6682.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 6673.00 | 6657.90 | 6682.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-23 15:15:00 | 6590.00 | 6652.18 | 6672.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 11:15:00 | 6260.50 | 6385.52 | 6484.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 6389.50 | 6386.31 | 6476.00 | SL hit (close>ema200) qty=0.50 sl=6386.31 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 6252.00 | 6074.55 | 6065.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 6386.50 | 6136.94 | 6095.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 6120.00 | 6155.00 | 6111.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 6120.00 | 6155.00 | 6111.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 6120.00 | 6155.00 | 6111.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 6120.00 | 6155.00 | 6111.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 6025.50 | 6129.10 | 6103.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 6025.50 | 6129.10 | 6103.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 6005.00 | 6104.28 | 6094.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 5906.50 | 6104.28 | 6094.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 5870.00 | 6057.43 | 6074.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 11:15:00 | 5827.00 | 5980.15 | 6034.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 6123.00 | 5928.24 | 5977.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 6123.00 | 5928.24 | 5977.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6123.00 | 5928.24 | 5977.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 6153.50 | 5928.24 | 5977.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 6210.00 | 6029.60 | 6017.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 6256.00 | 6140.10 | 6079.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 6203.00 | 6218.94 | 6155.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 6203.00 | 6218.94 | 6155.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 6266.00 | 6227.72 | 6170.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 6284.50 | 6227.72 | 6170.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:00:00 | 6285.00 | 6364.65 | 6349.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:45:00 | 6288.50 | 6348.12 | 6343.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 12:15:00 | 6286.50 | 6335.80 | 6338.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 12:15:00 | 6286.50 | 6335.80 | 6338.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 6251.00 | 6312.23 | 6326.53 | Break + close below crossover candle low |

### Cycle 139 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 6502.50 | 6340.01 | 6336.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 10:15:00 | 6510.50 | 6374.11 | 6351.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 6579.00 | 6607.40 | 6539.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 11:00:00 | 6579.00 | 6607.40 | 6539.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 6564.50 | 6598.82 | 6542.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 6539.00 | 6598.82 | 6542.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 6529.00 | 6584.85 | 6540.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 6534.00 | 6584.85 | 6540.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 6548.50 | 6577.58 | 6541.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:15:00 | 6505.00 | 6577.58 | 6541.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 6578.50 | 6577.77 | 6544.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 6523.00 | 6577.77 | 6544.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 6539.50 | 6570.11 | 6544.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 6605.50 | 6570.11 | 6544.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 6590.50 | 6575.31 | 6551.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 6408.50 | 6547.08 | 6545.29 | SL hit (close<static) qty=1.00 sl=6525.50 alert=retest2 |

### Cycle 140 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 6413.50 | 6520.36 | 6533.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 6333.00 | 6463.39 | 6503.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 11:15:00 | 6454.50 | 6442.67 | 6486.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 12:00:00 | 6454.50 | 6442.67 | 6486.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 6580.00 | 6456.37 | 6473.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 6580.00 | 6456.37 | 6473.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 6590.00 | 6483.10 | 6484.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 6610.50 | 6483.10 | 6484.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 6538.50 | 6494.18 | 6489.18 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 6434.50 | 6478.98 | 6483.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 6390.50 | 6429.38 | 6451.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 6425.00 | 6411.56 | 6436.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 6425.00 | 6411.56 | 6436.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 6425.00 | 6411.56 | 6436.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 6446.00 | 6411.56 | 6436.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 6436.00 | 6416.45 | 6436.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 6436.00 | 6416.45 | 6436.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 6482.50 | 6429.66 | 6440.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 6482.00 | 6429.66 | 6440.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 6490.00 | 6441.73 | 6444.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 6478.00 | 6441.73 | 6444.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 6490.00 | 6451.38 | 6449.08 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 6424.00 | 6450.28 | 6450.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 11:15:00 | 6398.00 | 6439.82 | 6445.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 6321.00 | 6286.47 | 6326.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 6321.00 | 6286.47 | 6326.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 6321.00 | 6286.47 | 6326.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 6320.50 | 6286.47 | 6326.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 6382.00 | 6305.57 | 6331.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:15:00 | 6304.50 | 6329.94 | 6338.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 6304.50 | 6314.48 | 6327.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 12:30:00 | 6306.00 | 6315.36 | 6325.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 6403.00 | 6338.11 | 6334.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 6403.00 | 6338.11 | 6334.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 6660.00 | 6413.93 | 6371.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 13:15:00 | 6557.50 | 6561.20 | 6499.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:00:00 | 6557.50 | 6561.20 | 6499.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 6567.50 | 6597.05 | 6552.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 6567.50 | 6597.05 | 6552.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 6647.00 | 6607.04 | 6560.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 6573.50 | 6607.04 | 6560.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 6575.00 | 6599.03 | 6571.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 6575.00 | 6599.03 | 6571.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 6592.00 | 6597.63 | 6573.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 6575.00 | 6597.63 | 6573.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 6493.50 | 6576.80 | 6566.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 6493.50 | 6576.80 | 6566.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 6520.00 | 6565.44 | 6562.21 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 6516.00 | 6555.55 | 6558.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 6471.50 | 6538.74 | 6550.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 6445.00 | 6430.03 | 6468.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 6445.00 | 6430.03 | 6468.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 6469.00 | 6437.82 | 6468.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 6469.00 | 6437.82 | 6468.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 6450.50 | 6440.36 | 6467.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 6499.00 | 6440.36 | 6467.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 6483.00 | 6448.89 | 6468.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 6511.50 | 6448.89 | 6468.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 6548.50 | 6468.81 | 6475.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 6548.50 | 6468.81 | 6475.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 6535.50 | 6485.22 | 6482.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 6582.50 | 6520.78 | 6501.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 6549.00 | 6616.94 | 6574.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 10:15:00 | 6549.00 | 6616.94 | 6574.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 6549.00 | 6616.94 | 6574.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 6549.00 | 6616.94 | 6574.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 6525.00 | 6598.55 | 6569.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 6525.00 | 6598.55 | 6569.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 6444.00 | 6567.64 | 6558.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 6444.00 | 6567.64 | 6558.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 6462.00 | 6546.51 | 6549.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 6419.50 | 6503.27 | 6528.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 6486.00 | 6483.93 | 6514.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 6486.00 | 6483.93 | 6514.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 6511.50 | 6482.79 | 6503.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 6511.50 | 6482.79 | 6503.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 6519.00 | 6490.03 | 6504.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 6575.00 | 6504.32 | 6509.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 6586.50 | 6520.76 | 6516.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 6710.50 | 6558.71 | 6534.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 6790.00 | 6792.82 | 6735.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 6790.00 | 6792.82 | 6735.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 6743.00 | 6783.39 | 6745.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:00:00 | 6743.00 | 6783.39 | 6745.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 6772.00 | 6781.11 | 6748.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 6916.50 | 6783.91 | 6755.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 6815.00 | 6842.18 | 6816.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:00:00 | 6830.00 | 6831.88 | 6815.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 6778.50 | 6803.08 | 6804.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 6778.50 | 6803.08 | 6804.90 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 6847.00 | 6811.86 | 6808.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 6883.50 | 6826.19 | 6815.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 7311.50 | 7317.86 | 7219.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 7311.50 | 7317.86 | 7219.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 7452.00 | 7347.92 | 7270.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 7480.00 | 7408.39 | 7342.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 10:45:00 | 7478.00 | 7416.31 | 7352.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:15:00 | 7472.00 | 7416.31 | 7352.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 7482.00 | 7428.25 | 7363.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 7470.00 | 7673.41 | 7616.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 7470.00 | 7673.41 | 7616.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 7471.00 | 7632.93 | 7602.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 12:15:00 | 7399.00 | 7555.27 | 7570.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 7399.00 | 7555.27 | 7570.88 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 7620.00 | 7557.59 | 7555.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 15:15:00 | 7644.00 | 7574.87 | 7563.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 7636.00 | 7748.91 | 7686.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 7636.00 | 7748.91 | 7686.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 7636.00 | 7748.91 | 7686.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 7636.00 | 7748.91 | 7686.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 7644.50 | 7728.03 | 7682.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 7644.50 | 7728.03 | 7682.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 7602.00 | 7702.82 | 7675.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 7595.00 | 7702.82 | 7675.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 15:15:00 | 7590.50 | 7651.30 | 7657.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 7570.50 | 7635.14 | 7649.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 7646.00 | 7612.49 | 7626.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 7646.00 | 7612.49 | 7626.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 7646.00 | 7612.49 | 7626.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 7680.00 | 7612.49 | 7626.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 7644.50 | 7618.89 | 7628.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 7644.50 | 7618.89 | 7628.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 7612.00 | 7617.51 | 7626.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 7637.50 | 7617.51 | 7626.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 7610.00 | 7600.76 | 7615.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 7624.50 | 7600.76 | 7615.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 7488.50 | 7492.19 | 7536.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 7530.50 | 7492.19 | 7536.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 7358.00 | 7311.88 | 7356.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 7342.50 | 7311.88 | 7356.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 7378.00 | 7325.11 | 7358.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:45:00 | 7306.00 | 7330.53 | 7353.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 7444.00 | 7347.16 | 7340.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 7444.00 | 7347.16 | 7340.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 7645.50 | 7464.02 | 7413.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 7850.00 | 7924.47 | 7767.71 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:45:00 | 7969.00 | 7930.97 | 7784.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:30:00 | 7993.50 | 7940.48 | 7802.52 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 14:30:00 | 7975.00 | 7955.31 | 7844.66 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 09:15:00 | 7979.50 | 7956.15 | 7855.10 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 7903.50 | 7978.14 | 7910.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 7903.50 | 7978.14 | 7910.16 | SL hit (close<ema400) qty=1.00 sl=7910.16 alert=retest1 |

### Cycle 156 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 7899.00 | 7911.77 | 7913.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 7672.00 | 7863.82 | 7891.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 7705.50 | 7669.79 | 7724.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 7705.50 | 7669.79 | 7724.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 7765.00 | 7688.83 | 7727.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 7677.00 | 7688.83 | 7727.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 7628.50 | 7676.76 | 7718.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 7606.00 | 7655.45 | 7701.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 7610.00 | 7647.06 | 7693.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-11 09:15:00 | 6845.40 | 7434.59 | 7573.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 7420.00 | 7010.18 | 7001.35 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 7296.00 | 7331.35 | 7333.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 7251.00 | 7315.28 | 7325.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 7310.00 | 7305.22 | 7318.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 7310.00 | 7305.22 | 7318.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 7310.00 | 7305.22 | 7318.97 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 15:15:00 | 7377.50 | 7333.53 | 7328.12 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 7270.00 | 7317.45 | 7321.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 7197.00 | 7293.36 | 7310.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 7340.00 | 7280.13 | 7295.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 7340.00 | 7280.13 | 7295.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 7340.00 | 7280.13 | 7295.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 7340.00 | 7280.13 | 7295.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 7308.00 | 7285.71 | 7297.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 7293.00 | 7285.71 | 7297.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:15:00 | 7295.00 | 7279.55 | 7289.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 11:15:00 | 7321.50 | 7295.85 | 7295.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 7321.50 | 7295.85 | 7295.48 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 7271.50 | 7296.47 | 7296.51 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 7425.00 | 7318.42 | 7306.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 7573.00 | 7369.33 | 7330.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 7604.00 | 7634.20 | 7571.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 7604.00 | 7634.20 | 7571.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 7530.00 | 7717.26 | 7698.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 7530.00 | 7717.26 | 7698.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 7529.50 | 7679.71 | 7682.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 7458.00 | 7563.53 | 7620.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 7540.50 | 7528.85 | 7587.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:45:00 | 7565.50 | 7528.85 | 7587.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 7640.50 | 7551.18 | 7592.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:45:00 | 7640.50 | 7551.18 | 7592.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 7690.00 | 7578.95 | 7601.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 7690.00 | 7578.95 | 7601.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 7781.50 | 7619.46 | 7617.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 7823.00 | 7660.17 | 7636.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 7917.50 | 7918.71 | 7872.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:15:00 | 8007.00 | 7918.71 | 7872.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 8205.00 | 8241.23 | 8190.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 8215.00 | 8241.23 | 8190.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 8199.00 | 8232.78 | 8190.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 8199.00 | 8232.78 | 8190.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 8199.00 | 8226.02 | 8191.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 8199.00 | 8226.02 | 8191.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 8200.00 | 8220.82 | 8192.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 8265.00 | 8220.82 | 8192.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 8193.00 | 8328.97 | 8347.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 8193.00 | 8328.97 | 8347.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 09:15:00 | 8017.00 | 8186.73 | 8263.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 8135.50 | 8131.79 | 8215.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 8135.50 | 8131.79 | 8215.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 8270.00 | 8159.44 | 8220.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 8275.00 | 8159.44 | 8220.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 8230.00 | 8173.55 | 8221.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 8244.00 | 8173.55 | 8221.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 8200.00 | 8178.84 | 8219.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 8122.50 | 8178.84 | 8219.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:45:00 | 8165.00 | 8181.64 | 8213.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:00:00 | 8187.00 | 8150.70 | 8173.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 8232.00 | 8192.60 | 8187.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 8232.00 | 8192.60 | 8187.57 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 8136.00 | 8193.90 | 8194.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 10:15:00 | 8110.50 | 8177.22 | 8187.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 14:15:00 | 8175.00 | 8166.07 | 8177.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 14:15:00 | 8175.00 | 8166.07 | 8177.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 8175.00 | 8166.07 | 8177.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:45:00 | 8175.50 | 8166.07 | 8177.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 8175.00 | 8167.85 | 8177.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 8210.00 | 8167.85 | 8177.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 8255.00 | 8185.28 | 8184.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 10:15:00 | 8324.00 | 8213.03 | 8196.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 8337.00 | 8347.86 | 8285.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:30:00 | 8359.00 | 8347.86 | 8285.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 8274.00 | 8333.09 | 8284.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 8274.00 | 8333.09 | 8284.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 8330.00 | 8332.47 | 8288.56 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 8120.50 | 8249.29 | 8264.27 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 8329.00 | 8269.46 | 8262.00 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 8186.50 | 8273.97 | 8284.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 8170.00 | 8253.18 | 8274.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 8268.50 | 8256.24 | 8273.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 8268.50 | 8256.24 | 8273.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 8268.50 | 8256.24 | 8273.82 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 8355.50 | 8266.19 | 8260.72 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 8255.00 | 8270.34 | 8271.15 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 8278.00 | 8271.87 | 8271.78 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 12:15:00 | 8263.00 | 8270.10 | 8270.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 8223.50 | 8251.02 | 8260.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 12:15:00 | 8308.00 | 8252.32 | 8256.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 12:15:00 | 8308.00 | 8252.32 | 8256.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 8308.00 | 8252.32 | 8256.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:45:00 | 8297.50 | 8252.32 | 8256.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 8312.00 | 8264.26 | 8261.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 8387.00 | 8298.52 | 8278.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 8393.00 | 8421.31 | 8375.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 8393.00 | 8421.31 | 8375.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 8393.00 | 8421.31 | 8375.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 8536.50 | 8444.25 | 8403.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 8308.50 | 8421.58 | 8418.66 | SL hit (close<static) qty=1.00 sl=8353.50 alert=retest2 |

### Cycle 178 — SELL (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 15:15:00 | 8310.00 | 8399.26 | 8408.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 8142.50 | 8347.91 | 8384.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 7974.00 | 7877.36 | 7950.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 7974.00 | 7877.36 | 7950.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 7974.00 | 7877.36 | 7950.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 7974.00 | 7877.36 | 7950.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 7950.50 | 7891.99 | 7950.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 7950.50 | 7891.99 | 7950.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 7803.00 | 7874.19 | 7937.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 7959.00 | 7874.19 | 7937.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 7268.00 | 7160.25 | 7265.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 7281.00 | 7160.25 | 7265.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 7195.00 | 7167.20 | 7258.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:45:00 | 7147.00 | 7161.96 | 7248.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:30:00 | 7158.00 | 7153.45 | 7229.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:30:00 | 7175.00 | 7158.08 | 7207.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 7168.00 | 7170.47 | 7201.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 7254.00 | 7170.05 | 7183.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 7295.00 | 7170.05 | 7183.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 7294.00 | 7194.84 | 7193.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 7294.00 | 7194.84 | 7193.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 11:15:00 | 7327.50 | 7221.37 | 7205.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 7360.50 | 7403.87 | 7347.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 7360.50 | 7403.87 | 7347.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 7360.50 | 7403.87 | 7347.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 7360.50 | 7403.87 | 7347.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 7356.00 | 7394.29 | 7348.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 7356.00 | 7394.29 | 7348.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 7361.00 | 7387.64 | 7349.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:15:00 | 7348.50 | 7387.64 | 7349.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 7355.50 | 7381.21 | 7350.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 14:00:00 | 7374.00 | 7379.77 | 7352.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 7382.50 | 7369.71 | 7352.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 7370.00 | 7367.61 | 7354.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 7376.00 | 7366.69 | 7354.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 7354.50 | 7382.15 | 7369.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 7354.50 | 7382.15 | 7369.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 7356.50 | 7377.02 | 7367.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 7356.50 | 7377.02 | 7367.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 7345.00 | 7370.61 | 7365.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 7345.00 | 7370.61 | 7365.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 12:15:00 | 7288.50 | 7354.19 | 7358.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 7288.50 | 7354.19 | 7358.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 13:15:00 | 7264.00 | 7336.15 | 7350.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 12:15:00 | 7088.00 | 7081.19 | 7141.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:45:00 | 7076.00 | 7081.19 | 7141.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 7146.50 | 7099.50 | 7139.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 7198.50 | 7099.50 | 7139.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 7140.00 | 7107.60 | 7139.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 7170.50 | 7107.60 | 7139.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 7172.00 | 7120.48 | 7142.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 7181.50 | 7120.48 | 7142.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 7182.00 | 7132.78 | 7145.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 7182.00 | 7132.78 | 7145.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 7157.50 | 7137.73 | 7146.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 7155.00 | 7137.73 | 7146.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 7202.00 | 7150.58 | 7151.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 7202.00 | 7150.58 | 7151.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 7291.00 | 7178.67 | 7164.62 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 7106.00 | 7160.21 | 7167.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 7068.50 | 7127.19 | 7149.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 7164.00 | 7107.38 | 7127.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 7164.00 | 7107.38 | 7127.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 7164.00 | 7107.38 | 7127.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 7164.00 | 7107.38 | 7127.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 7220.00 | 7129.90 | 7136.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 7133.50 | 7129.90 | 7136.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 7062.00 | 7076.90 | 7098.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 7027.00 | 7073.12 | 7092.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 7034.00 | 7041.24 | 7057.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:30:00 | 7036.00 | 7027.97 | 7044.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 6675.65 | 6826.07 | 6932.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 6682.30 | 6826.07 | 6932.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 6684.20 | 6826.07 | 6932.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 6503.00 | 6482.89 | 6581.09 | SL hit (close>ema200) qty=0.50 sl=6482.89 alert=retest2 |

### Cycle 183 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 6720.50 | 6631.16 | 6620.66 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 6560.00 | 6619.00 | 6620.04 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 6625.00 | 6590.92 | 6586.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 6721.50 | 6617.04 | 6599.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 13:15:00 | 6767.50 | 6772.97 | 6722.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:00:00 | 6767.50 | 6772.97 | 6722.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 6725.50 | 6764.81 | 6731.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 6725.50 | 6764.81 | 6731.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 6715.00 | 6754.85 | 6729.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 6717.00 | 6754.85 | 6729.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 6648.00 | 6733.48 | 6722.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 6648.00 | 6733.48 | 6722.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 6583.50 | 6703.48 | 6709.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 6556.50 | 6600.80 | 6641.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 6621.50 | 6601.48 | 6631.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 6621.50 | 6601.48 | 6631.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 6621.50 | 6601.48 | 6631.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 6656.50 | 6601.48 | 6631.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 6635.00 | 6608.18 | 6631.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 6635.00 | 6608.18 | 6631.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 6645.50 | 6615.65 | 6632.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 6645.50 | 6615.65 | 6632.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 6626.50 | 6617.82 | 6632.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 6682.50 | 6617.82 | 6632.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 6651.00 | 6624.45 | 6633.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 6651.00 | 6624.45 | 6633.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 6678.00 | 6635.16 | 6637.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 6678.00 | 6635.16 | 6637.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 6685.00 | 6645.13 | 6642.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 6715.00 | 6659.10 | 6648.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 6692.00 | 6698.29 | 6677.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 6663.00 | 6698.29 | 6677.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 6698.50 | 6698.33 | 6679.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 6636.00 | 6698.33 | 6679.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 6697.50 | 6697.39 | 6682.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 6675.00 | 6697.39 | 6682.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 6659.50 | 6689.81 | 6679.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 6659.50 | 6689.81 | 6679.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 6678.50 | 6687.55 | 6679.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 6755.00 | 6675.73 | 6675.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:45:00 | 6689.50 | 6696.84 | 6689.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 6724.50 | 6695.27 | 6689.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 6668.50 | 6686.03 | 6686.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 6668.50 | 6686.03 | 6686.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 15:15:00 | 6629.00 | 6668.22 | 6677.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 6358.00 | 6343.14 | 6428.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:45:00 | 6353.50 | 6343.14 | 6428.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 6392.50 | 6358.83 | 6421.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 6392.50 | 6358.83 | 6421.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 6401.50 | 6371.55 | 6416.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 6433.50 | 6371.55 | 6416.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 6412.00 | 6379.64 | 6415.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 6410.00 | 6379.64 | 6415.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 6390.00 | 6381.71 | 6413.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:30:00 | 6411.00 | 6381.71 | 6413.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 6408.50 | 6389.76 | 6411.87 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 6514.00 | 6430.52 | 6426.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 6668.50 | 6509.65 | 6471.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 6642.50 | 6659.26 | 6590.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 6642.50 | 6659.26 | 6590.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 6650.00 | 6666.86 | 6640.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 6667.50 | 6665.89 | 6642.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:45:00 | 6664.00 | 6658.53 | 6643.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 6569.50 | 6640.72 | 6636.32 | SL hit (close<static) qty=1.00 sl=6620.00 alert=retest2 |

### Cycle 190 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 6552.00 | 6622.98 | 6628.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 6454.00 | 6543.84 | 6583.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 6154.50 | 6127.68 | 6223.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 6154.50 | 6127.68 | 6223.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 6223.00 | 6141.20 | 6173.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:45:00 | 6135.00 | 6159.11 | 6173.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 6129.00 | 6152.13 | 6167.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 6135.00 | 6159.79 | 6168.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 5828.25 | 5933.71 | 6017.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 5822.55 | 5933.71 | 6017.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 5828.25 | 5933.71 | 6017.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 09:15:00 | 5521.50 | 5595.60 | 5709.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 191 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 5832.00 | 5593.38 | 5582.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 5882.00 | 5738.42 | 5676.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 5831.50 | 5881.83 | 5798.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:00:00 | 5831.50 | 5881.83 | 5798.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 7873.00 | 7781.31 | 7716.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:15:00 | 7898.00 | 7781.31 | 7716.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:15:00 | 7880.00 | 7805.72 | 7773.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 7698.00 | 7776.94 | 7787.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 7698.00 | 7776.94 | 7787.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 12:15:00 | 7628.50 | 7716.64 | 7753.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 7807.00 | 7711.64 | 7736.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 7807.00 | 7711.64 | 7736.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 7807.00 | 7711.64 | 7736.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 7846.00 | 7711.64 | 7736.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 7845.00 | 7738.31 | 7746.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 7845.00 | 7738.31 | 7746.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 7823.00 | 7755.25 | 7753.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 11:15:00 | 7850.00 | 7796.64 | 7776.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 7997.00 | 7997.37 | 7933.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:45:00 | 7984.00 | 7997.37 | 7933.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 7966.50 | 7991.19 | 7936.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 7957.00 | 7991.19 | 7936.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 7966.50 | 7980.14 | 7940.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 8041.00 | 7961.25 | 7938.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:30:00 | 7981.50 | 7966.16 | 7944.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:00:00 | 7996.00 | 7966.16 | 7944.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:30:00 | 7986.00 | 7977.42 | 7955.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 7972.50 | 7976.43 | 7957.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 7972.50 | 7976.43 | 7957.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 7930.00 | 7966.12 | 7955.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 7930.00 | 7966.12 | 7955.72 | SL hit (close<static) qty=1.00 sl=7936.50 alert=retest2 |

### Cycle 194 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 7851.00 | 7943.09 | 7946.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 7791.00 | 7912.68 | 7932.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 7897.00 | 7881.51 | 7909.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 7897.00 | 7881.51 | 7909.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 7890.00 | 7883.20 | 7908.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 7647.50 | 7883.20 | 7908.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 7848.50 | 7732.15 | 7740.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 7849.00 | 7764.94 | 7754.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 7849.00 | 7764.94 | 7754.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 7926.50 | 7797.25 | 7770.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 7650.00 | 7801.40 | 7785.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 7650.00 | 7801.40 | 7785.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 7650.00 | 7801.40 | 7785.13 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 7312.50 | 7687.40 | 7735.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 09:15:00 | 7238.50 | 7444.29 | 7583.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 13:15:00 | 7445.00 | 7428.28 | 7529.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 13:45:00 | 7441.50 | 7428.28 | 7529.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 7505.00 | 7443.63 | 7527.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 7472.00 | 7443.63 | 7527.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 7530.00 | 7460.90 | 7527.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 7434.50 | 7460.90 | 7527.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 7062.77 | 7295.77 | 7404.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-13 09:15:00 | 6691.05 | 6980.30 | 7174.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 197 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 6867.50 | 6717.61 | 6714.24 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 6677.00 | 6743.52 | 6750.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6369.50 | 6633.18 | 6685.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 6325.00 | 6304.98 | 6437.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 6325.00 | 6304.98 | 6437.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 6465.00 | 6336.99 | 6439.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 6479.50 | 6336.99 | 6439.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 6485.50 | 6366.69 | 6444.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 6498.50 | 6366.69 | 6444.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 6450.00 | 6398.28 | 6445.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 6708.50 | 6398.28 | 6445.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 6738.00 | 6466.22 | 6472.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 6738.00 | 6466.22 | 6472.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 6750.00 | 6522.98 | 6497.73 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 6503.00 | 6573.15 | 6579.17 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 12:15:00 | 6622.50 | 6583.80 | 6583.01 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 6575.50 | 6582.14 | 6582.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 6535.00 | 6568.05 | 6575.60 | Break + close below crossover candle low |

### Cycle 203 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 6711.00 | 6596.64 | 6587.91 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 6247.50 | 6528.39 | 6564.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 09:15:00 | 6219.00 | 6321.22 | 6421.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 6324.50 | 6316.08 | 6401.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:45:00 | 6345.00 | 6316.08 | 6401.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 6398.00 | 6341.87 | 6392.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 6398.00 | 6341.87 | 6392.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 6430.00 | 6359.49 | 6395.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 6348.50 | 6359.49 | 6395.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 11:00:00 | 6380.00 | 6364.95 | 6392.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 6435.00 | 6406.27 | 6404.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 6435.00 | 6406.27 | 6404.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 6793.00 | 6483.61 | 6439.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 6789.50 | 6829.10 | 6681.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 6789.50 | 6829.10 | 6681.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 7863.50 | 7924.67 | 7863.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:45:00 | 7865.00 | 7924.67 | 7863.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 7843.50 | 7908.43 | 7862.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 7890.00 | 7908.43 | 7862.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 7848.00 | 7896.35 | 7860.78 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 15:15:00 | 7803.00 | 7852.30 | 7852.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 7758.50 | 7833.54 | 7844.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 7834.50 | 7811.73 | 7824.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 7834.50 | 7811.73 | 7824.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 7834.50 | 7811.73 | 7824.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:45:00 | 7782.00 | 7802.68 | 7819.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:45:00 | 7785.00 | 7806.95 | 7820.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:30:00 | 7784.50 | 7800.06 | 7815.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:00:00 | 7772.50 | 7800.06 | 7815.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 7972.00 | 7816.36 | 7815.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 7972.00 | 7816.36 | 7815.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 8171.00 | 7914.91 | 7862.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 11:15:00 | 8138.00 | 8140.72 | 8070.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 11:45:00 | 8130.50 | 8140.72 | 8070.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 8081.00 | 8127.62 | 8076.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 8081.00 | 8127.62 | 8076.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 8077.00 | 8117.50 | 8076.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 8096.50 | 8117.50 | 8076.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 8081.00 | 8110.20 | 8077.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 8018.00 | 8110.20 | 8077.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 7878.50 | 8063.86 | 8059.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 7878.50 | 8063.86 | 8059.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 7939.00 | 8038.89 | 8048.14 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 8122.50 | 8032.03 | 8027.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 8245.00 | 8090.26 | 8056.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 8726.00 | 8768.51 | 8638.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 8721.50 | 8768.51 | 8638.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-17 09:15:00 | 2149.40 | 2023-05-23 11:15:00 | 2065.00 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2023-05-29 09:15:00 | 2154.00 | 2023-05-31 13:15:00 | 2125.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-05-29 10:45:00 | 2131.70 | 2023-05-31 13:15:00 | 2125.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-05-30 09:15:00 | 2160.95 | 2023-05-31 13:15:00 | 2125.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-05-31 12:15:00 | 2135.00 | 2023-05-31 13:15:00 | 2125.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2023-06-02 09:15:00 | 2217.85 | 2023-06-06 14:15:00 | 2163.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2023-06-19 09:15:00 | 2131.40 | 2023-06-19 12:15:00 | 2114.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest1 | 2023-06-22 14:00:00 | 2369.00 | 2023-06-23 12:15:00 | 2304.90 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2023-06-30 14:15:00 | 2252.00 | 2023-07-06 12:15:00 | 2269.25 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-06-30 15:15:00 | 2250.00 | 2023-07-06 12:15:00 | 2269.25 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-07-03 12:30:00 | 2252.00 | 2023-07-06 12:15:00 | 2269.25 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-07-03 14:00:00 | 2252.80 | 2023-07-06 12:15:00 | 2269.25 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-07-04 10:00:00 | 2248.55 | 2023-07-06 12:15:00 | 2269.25 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-07-05 09:15:00 | 2243.75 | 2023-07-06 12:15:00 | 2269.25 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-07-12 10:15:00 | 2219.50 | 2023-07-12 13:15:00 | 2229.95 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-07-12 12:00:00 | 2225.15 | 2023-07-12 13:15:00 | 2229.95 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2023-07-17 10:15:00 | 2213.25 | 2023-07-17 13:15:00 | 2223.15 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-07-17 11:45:00 | 2208.75 | 2023-07-17 13:15:00 | 2223.15 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-07-17 13:00:00 | 2212.15 | 2023-07-17 13:15:00 | 2223.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-07-20 14:00:00 | 2196.25 | 2023-07-21 11:15:00 | 2221.05 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-07-20 15:15:00 | 2195.00 | 2023-07-21 11:15:00 | 2221.05 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2023-07-26 10:30:00 | 2274.00 | 2023-07-27 10:15:00 | 2501.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-07 12:30:00 | 2473.00 | 2023-08-10 14:15:00 | 2490.00 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest1 | 2023-08-11 12:45:00 | 2454.00 | 2023-08-14 09:15:00 | 2518.95 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest1 | 2023-08-11 13:30:00 | 2449.95 | 2023-08-14 09:15:00 | 2518.95 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest1 | 2023-08-30 09:15:00 | 2942.95 | 2023-08-30 15:15:00 | 2895.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest1 | 2023-08-30 12:45:00 | 2917.00 | 2023-08-30 15:15:00 | 2895.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-08-31 09:15:00 | 2912.10 | 2023-09-12 09:15:00 | 2993.25 | STOP_HIT | 1.00 | 2.79% |
| BUY | retest2 | 2023-08-31 11:15:00 | 2909.75 | 2023-09-12 09:15:00 | 2993.25 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest2 | 2023-09-01 09:15:00 | 3028.95 | 2023-09-12 09:15:00 | 2993.25 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-09-27 11:45:00 | 3052.65 | 2023-09-28 15:15:00 | 2952.00 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2023-09-27 12:45:00 | 3057.85 | 2023-09-28 15:15:00 | 2952.00 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2023-09-27 14:15:00 | 3057.10 | 2023-09-28 15:15:00 | 2952.00 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2023-09-27 14:45:00 | 3055.00 | 2023-09-28 15:15:00 | 2952.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2023-10-25 13:00:00 | 2936.05 | 2023-10-25 15:15:00 | 2959.05 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-10-25 13:30:00 | 2940.05 | 2023-10-25 15:15:00 | 2959.05 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-11-20 10:45:00 | 3308.25 | 2023-11-22 13:15:00 | 3246.15 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2023-12-05 13:30:00 | 3158.20 | 2023-12-06 10:15:00 | 3107.85 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2023-12-14 11:15:00 | 3100.95 | 2023-12-14 12:15:00 | 3155.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-01-12 09:45:00 | 3554.45 | 2024-01-18 11:15:00 | 3909.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-14 09:15:00 | 3581.05 | 2024-02-14 14:15:00 | 3856.80 | STOP_HIT | 1.00 | -7.70% |
| SELL | retest2 | 2024-02-14 09:45:00 | 3585.00 | 2024-02-14 14:15:00 | 3856.80 | STOP_HIT | 1.00 | -7.58% |
| SELL | retest2 | 2024-02-14 13:15:00 | 3577.40 | 2024-02-14 14:15:00 | 3856.80 | STOP_HIT | 1.00 | -7.81% |
| BUY | retest2 | 2024-02-20 09:15:00 | 4115.00 | 2024-02-21 13:15:00 | 3912.05 | STOP_HIT | 1.00 | -4.93% |
| SELL | retest2 | 2024-02-29 10:45:00 | 3571.00 | 2024-03-04 12:15:00 | 3693.95 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2024-03-01 15:15:00 | 3573.00 | 2024-03-04 12:15:00 | 3693.95 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-03-04 09:15:00 | 3573.50 | 2024-03-04 12:15:00 | 3693.95 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-03-11 15:15:00 | 3615.00 | 2024-03-13 11:15:00 | 3434.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-12 10:15:00 | 3605.00 | 2024-03-13 11:15:00 | 3424.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 15:15:00 | 3615.00 | 2024-03-13 14:15:00 | 3554.00 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2024-03-12 10:15:00 | 3605.00 | 2024-03-13 14:15:00 | 3554.00 | STOP_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2024-03-26 12:00:00 | 3576.50 | 2024-04-01 09:15:00 | 3755.33 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-03-26 12:00:00 | 3576.50 | 2024-04-01 13:15:00 | 3660.20 | STOP_HIT | 0.50 | 2.34% |
| BUY | retest2 | 2024-04-05 12:00:00 | 3816.15 | 2024-04-05 14:15:00 | 3743.75 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-04-05 13:00:00 | 3808.80 | 2024-04-05 14:15:00 | 3743.75 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-04-08 09:15:00 | 3825.50 | 2024-04-09 13:15:00 | 3745.10 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-04-09 09:15:00 | 3826.30 | 2024-04-09 13:15:00 | 3745.10 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-04-18 10:45:00 | 3636.20 | 2024-04-22 09:15:00 | 3685.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-04-18 11:45:00 | 3638.40 | 2024-04-22 09:15:00 | 3685.60 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-04-18 13:15:00 | 3622.35 | 2024-04-22 09:15:00 | 3685.60 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-04-23 09:15:00 | 3747.85 | 2024-04-29 09:15:00 | 3740.35 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-05-23 15:00:00 | 3813.00 | 2024-05-28 11:15:00 | 3622.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-23 15:00:00 | 3813.00 | 2024-05-31 09:15:00 | 3629.65 | STOP_HIT | 0.50 | 4.81% |
| BUY | retest2 | 2024-06-18 12:15:00 | 4207.00 | 2024-06-18 15:15:00 | 3989.00 | STOP_HIT | 1.00 | -5.18% |
| SELL | retest2 | 2024-06-20 14:45:00 | 4023.00 | 2024-06-24 09:15:00 | 4089.45 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-06-21 10:00:00 | 3994.60 | 2024-06-24 09:15:00 | 4089.45 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-06-21 13:00:00 | 4017.95 | 2024-06-24 09:15:00 | 4089.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-06-21 14:15:00 | 4019.20 | 2024-06-24 09:15:00 | 4089.45 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-06-26 09:15:00 | 4122.30 | 2024-06-28 11:15:00 | 4534.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-29 09:15:00 | 4464.65 | 2024-07-30 14:15:00 | 4343.00 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-07-29 10:45:00 | 4420.00 | 2024-07-30 14:15:00 | 4343.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-07-30 09:15:00 | 4416.00 | 2024-07-30 14:15:00 | 4343.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-08-09 12:30:00 | 4322.25 | 2024-08-12 09:15:00 | 4179.00 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2024-08-09 14:45:00 | 4320.00 | 2024-08-12 09:15:00 | 4179.00 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2024-08-16 10:45:00 | 4064.20 | 2024-08-16 13:15:00 | 4119.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-08-20 12:15:00 | 4127.10 | 2024-08-29 10:15:00 | 4539.81 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-27 15:15:00 | 4619.00 | 2024-09-30 14:15:00 | 4812.80 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2024-10-07 09:15:00 | 5049.00 | 2024-10-07 10:15:00 | 4874.85 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2024-10-07 11:15:00 | 4930.50 | 2024-10-07 12:15:00 | 4875.90 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-10-07 12:00:00 | 4941.80 | 2024-10-07 12:15:00 | 4875.90 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-10-21 09:15:00 | 5738.85 | 2024-10-21 14:15:00 | 6312.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-22 12:15:00 | 5534.00 | 2024-10-23 09:15:00 | 6087.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-13 14:30:00 | 6066.50 | 2024-11-18 09:15:00 | 6205.90 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-11-13 15:00:00 | 6067.60 | 2024-11-18 09:15:00 | 6205.90 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-11-14 09:15:00 | 6075.20 | 2024-11-18 09:15:00 | 6205.90 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-11-19 09:15:00 | 6244.85 | 2024-11-19 14:15:00 | 6153.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-11-21 09:15:00 | 6361.00 | 2024-11-27 11:15:00 | 6475.00 | STOP_HIT | 1.00 | 1.79% |
| SELL | retest2 | 2024-12-02 15:15:00 | 6050.65 | 2024-12-10 09:15:00 | 5748.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-03 10:15:00 | 6013.05 | 2024-12-10 09:15:00 | 5712.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-02 15:15:00 | 6050.65 | 2024-12-10 12:15:00 | 5772.50 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2024-12-03 10:15:00 | 6013.05 | 2024-12-10 12:15:00 | 5772.50 | STOP_HIT | 0.50 | 4.00% |
| BUY | retest2 | 2024-12-19 12:00:00 | 6175.05 | 2024-12-23 13:15:00 | 6792.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-16 11:45:00 | 6967.45 | 2025-01-20 14:15:00 | 7058.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-01-20 10:45:00 | 6935.85 | 2025-01-20 14:15:00 | 7058.95 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-01-28 15:15:00 | 6635.00 | 2025-01-29 09:15:00 | 7098.10 | STOP_HIT | 1.00 | -6.98% |
| SELL | retest2 | 2025-01-31 13:15:00 | 6462.25 | 2025-02-01 09:15:00 | 6626.35 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-02-03 10:45:00 | 6728.00 | 2025-02-11 09:15:00 | 6628.10 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-02-04 09:30:00 | 6720.75 | 2025-02-11 09:15:00 | 6628.10 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-02-24 11:15:00 | 5828.75 | 2025-02-28 09:15:00 | 5755.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-03-11 13:45:00 | 6422.95 | 2025-03-20 09:15:00 | 7065.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-11 15:00:00 | 6418.25 | 2025-03-20 09:15:00 | 7060.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-12 09:45:00 | 6409.80 | 2025-03-20 09:15:00 | 7050.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-12 12:30:00 | 6440.05 | 2025-03-20 09:15:00 | 7084.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 09:15:00 | 6542.05 | 2025-03-20 09:15:00 | 7196.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 10:15:00 | 6691.75 | 2025-04-07 09:15:00 | 6022.57 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 11:00:00 | 6713.60 | 2025-04-07 09:15:00 | 6042.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 09:45:00 | 6775.00 | 2025-04-22 09:15:00 | 6628.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-04-23 15:15:00 | 6590.00 | 2025-04-25 11:15:00 | 6260.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-23 15:15:00 | 6590.00 | 2025-04-25 12:15:00 | 6389.50 | STOP_HIT | 0.50 | 3.04% |
| BUY | retest2 | 2025-05-14 10:15:00 | 6284.50 | 2025-05-19 12:15:00 | 6286.50 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-05-19 11:00:00 | 6285.00 | 2025-05-19 12:15:00 | 6286.50 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-05-19 11:45:00 | 6288.50 | 2025-05-19 12:15:00 | 6286.50 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-05-23 09:15:00 | 6605.50 | 2025-05-23 13:15:00 | 6408.50 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-05-23 11:15:00 | 6590.50 | 2025-05-23 13:15:00 | 6408.50 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-06-05 14:15:00 | 6304.50 | 2025-06-06 14:15:00 | 6403.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-06-06 11:15:00 | 6304.50 | 2025-06-06 14:15:00 | 6403.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-06-06 12:30:00 | 6306.00 | 2025-06-06 14:15:00 | 6403.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-06-27 09:15:00 | 6916.50 | 2025-07-01 09:15:00 | 6778.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-06-30 12:15:00 | 6815.00 | 2025-07-01 09:15:00 | 6778.50 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-06-30 14:00:00 | 6830.00 | 2025-07-01 09:15:00 | 6778.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-08 10:15:00 | 7480.00 | 2025-07-11 12:15:00 | 7399.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-08 10:45:00 | 7478.00 | 2025-07-11 12:15:00 | 7399.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-08 11:15:00 | 7472.00 | 2025-07-11 12:15:00 | 7399.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-08 11:45:00 | 7482.00 | 2025-07-11 12:15:00 | 7399.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-25 10:45:00 | 7306.00 | 2025-07-28 11:15:00 | 7444.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest1 | 2025-07-31 10:45:00 | 7969.00 | 2025-08-01 13:15:00 | 7903.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest1 | 2025-07-31 11:30:00 | 7993.50 | 2025-08-01 13:15:00 | 7903.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest1 | 2025-07-31 14:30:00 | 7975.00 | 2025-08-01 13:15:00 | 7903.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2025-08-01 09:15:00 | 7979.50 | 2025-08-01 13:15:00 | 7903.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-04 12:45:00 | 7950.00 | 2025-08-05 13:15:00 | 7845.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-04 13:30:00 | 7945.00 | 2025-08-05 13:15:00 | 7845.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-05 11:30:00 | 7946.00 | 2025-08-05 13:15:00 | 7845.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-05 12:15:00 | 7943.50 | 2025-08-05 13:15:00 | 7845.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-08-08 12:00:00 | 7606.00 | 2025-08-11 09:15:00 | 6845.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-08 13:15:00 | 7610.00 | 2025-08-11 09:15:00 | 6849.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-28 12:15:00 | 7293.00 | 2025-08-29 11:15:00 | 7321.50 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-08-29 10:15:00 | 7295.00 | 2025-08-29 11:15:00 | 7321.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-09-18 14:15:00 | 8265.00 | 2025-09-26 11:15:00 | 8193.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-30 09:15:00 | 8122.50 | 2025-10-01 15:15:00 | 8232.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-30 10:45:00 | 8165.00 | 2025-10-01 15:15:00 | 8232.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-10-01 12:00:00 | 8187.00 | 2025-10-01 15:15:00 | 8232.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-28 15:00:00 | 8536.50 | 2025-10-29 14:15:00 | 8308.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-11-11 14:45:00 | 7147.00 | 2025-11-14 10:15:00 | 7294.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-11-12 09:30:00 | 7158.00 | 2025-11-14 10:15:00 | 7294.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-11-12 13:30:00 | 7175.00 | 2025-11-14 10:15:00 | 7294.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-11-13 10:00:00 | 7168.00 | 2025-11-14 10:15:00 | 7294.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-11-18 14:00:00 | 7374.00 | 2025-11-20 12:15:00 | 7288.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-19 09:15:00 | 7382.50 | 2025-11-20 12:15:00 | 7288.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-11-19 10:30:00 | 7370.00 | 2025-11-20 12:15:00 | 7288.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-19 12:15:00 | 7376.00 | 2025-11-20 12:15:00 | 7288.50 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-02 12:15:00 | 7027.00 | 2025-12-05 09:15:00 | 6675.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 14:15:00 | 7034.00 | 2025-12-05 09:15:00 | 6682.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 11:30:00 | 7036.00 | 2025-12-05 09:15:00 | 6684.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 12:15:00 | 7027.00 | 2025-12-09 11:15:00 | 6503.00 | STOP_HIT | 0.50 | 7.46% |
| SELL | retest2 | 2025-12-03 14:15:00 | 7034.00 | 2025-12-09 11:15:00 | 6503.00 | STOP_HIT | 0.50 | 7.55% |
| SELL | retest2 | 2025-12-04 11:30:00 | 7036.00 | 2025-12-09 11:15:00 | 6503.00 | STOP_HIT | 0.50 | 7.58% |
| BUY | retest2 | 2025-12-24 09:15:00 | 6755.00 | 2025-12-26 10:15:00 | 6668.50 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-12-24 14:45:00 | 6689.50 | 2025-12-26 10:15:00 | 6668.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-12-26 09:15:00 | 6724.50 | 2025-12-26 10:15:00 | 6668.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-07 14:45:00 | 6667.50 | 2026-01-08 10:15:00 | 6569.50 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-08 09:45:00 | 6664.00 | 2026-01-08 10:15:00 | 6569.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-01-16 13:45:00 | 6135.00 | 2026-01-21 13:15:00 | 5828.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 6129.00 | 2026-01-21 13:15:00 | 5822.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 12:00:00 | 6135.00 | 2026-01-21 13:15:00 | 5828.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:45:00 | 6135.00 | 2026-01-27 09:15:00 | 5521.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 6129.00 | 2026-01-27 09:15:00 | 5516.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 12:00:00 | 6135.00 | 2026-01-27 09:15:00 | 5521.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-17 10:15:00 | 7898.00 | 2026-02-19 15:15:00 | 7698.00 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-02-18 13:15:00 | 7880.00 | 2026-02-19 15:15:00 | 7698.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-02-27 09:15:00 | 8041.00 | 2026-03-02 09:15:00 | 7930.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-02-27 10:30:00 | 7981.50 | 2026-03-02 09:15:00 | 7930.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-27 11:00:00 | 7996.00 | 2026-03-02 09:15:00 | 7930.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-02-27 13:30:00 | 7986.00 | 2026-03-02 09:15:00 | 7930.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-03-04 09:15:00 | 7647.50 | 2026-03-06 11:15:00 | 7849.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-03-06 09:45:00 | 7848.50 | 2026-03-06 11:15:00 | 7849.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-03-11 09:15:00 | 7434.50 | 2026-03-12 09:15:00 | 7062.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 09:15:00 | 7434.50 | 2026-03-13 09:15:00 | 6691.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-07 09:15:00 | 6348.50 | 2026-04-07 15:15:00 | 6435.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-04-07 11:00:00 | 6380.00 | 2026-04-07 15:15:00 | 6435.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-04-24 10:45:00 | 7782.00 | 2026-04-27 09:15:00 | 7972.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-04-24 11:45:00 | 7785.00 | 2026-04-27 09:15:00 | 7972.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-04-24 12:30:00 | 7784.50 | 2026-04-27 09:15:00 | 7972.00 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-04-24 13:00:00 | 7772.50 | 2026-04-27 09:15:00 | 7972.00 | STOP_HIT | 1.00 | -2.57% |
