# Astral Ltd. (ASTRAL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1567.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 144 |
| ALERT1 | 96 |
| ALERT2 | 97 |
| ALERT2_SKIP | 45 |
| ALERT3 | 282 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 89 |
| PARTIAL | 12 |
| TARGET_HIT | 1 |
| STOP_HIT | 87 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 100 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 52 / 48
- **Target hits / Stop hits / Partials:** 1 / 87 / 12
- **Avg / median % per leg:** 0.92% / 0.20%
- **Sum % (uncompounded):** 92.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 12 | 37.5% | 0 | 32 | 0 | 0.12% | 3.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.11% | -1.1% |
| BUY @ 3rd Alert (retest2) | 31 | 12 | 38.7% | 0 | 31 | 0 | 0.15% | 4.8% |
| SELL (all) | 68 | 40 | 58.8% | 1 | 55 | 12 | 1.31% | 88.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 68 | 40 | 58.8% | 1 | 55 | 12 | 1.31% | 88.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.11% | -1.1% |
| retest2 (combined) | 99 | 52 | 52.5% | 1 | 86 | 12 | 0.94% | 93.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-18 12:15:00 | 2189.80 | 2248.04 | 2249.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 09:15:00 | 2094.05 | 2217.24 | 2235.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 13:15:00 | 2107.30 | 2107.12 | 2142.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 14:00:00 | 2107.30 | 2107.12 | 2142.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 2144.95 | 2114.53 | 2124.32 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 12:15:00 | 2152.70 | 2132.19 | 2130.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 2180.55 | 2148.27 | 2139.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 15:15:00 | 2168.50 | 2169.35 | 2156.05 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:15:00 | 2187.85 | 2169.35 | 2156.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 2163.50 | 2173.39 | 2164.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-28 13:15:00 | 2163.50 | 2173.39 | 2164.03 | SL hit (close<ema400) qty=1.00 sl=2164.03 alert=retest1 |

### Cycle 3 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 2137.45 | 2158.12 | 2158.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 11:15:00 | 2128.15 | 2148.13 | 2153.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 2094.40 | 2092.90 | 2107.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 15:00:00 | 2094.40 | 2092.90 | 2107.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 2102.00 | 2094.72 | 2106.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 2136.10 | 2094.72 | 2106.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2169.80 | 2109.74 | 2112.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 2169.80 | 2109.74 | 2112.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 2159.40 | 2119.67 | 2116.59 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 2024.25 | 2105.42 | 2113.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1988.85 | 2082.11 | 2102.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 2076.95 | 2046.85 | 2071.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 2076.95 | 2046.85 | 2071.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 2076.95 | 2046.85 | 2071.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 2076.95 | 2046.85 | 2071.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 2064.35 | 2050.35 | 2070.94 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 2111.95 | 2083.04 | 2081.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 2156.30 | 2097.69 | 2088.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 2129.70 | 2148.93 | 2137.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 10:15:00 | 2129.70 | 2148.93 | 2137.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 2129.70 | 2148.93 | 2137.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:00:00 | 2129.70 | 2148.93 | 2137.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 2145.00 | 2148.14 | 2138.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 12:30:00 | 2148.90 | 2147.54 | 2138.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 13:15:00 | 2152.90 | 2147.54 | 2138.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 14:30:00 | 2156.00 | 2151.30 | 2141.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 10:15:00 | 2219.00 | 2245.94 | 2247.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 2219.00 | 2245.94 | 2247.49 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 2280.40 | 2242.00 | 2237.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 11:15:00 | 2285.00 | 2250.60 | 2241.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 10:15:00 | 2403.45 | 2406.72 | 2383.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 10:45:00 | 2408.75 | 2406.72 | 2383.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 2385.20 | 2397.99 | 2384.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 2385.20 | 2397.99 | 2384.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 2412.60 | 2400.92 | 2387.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 2394.50 | 2400.92 | 2387.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 2385.00 | 2402.23 | 2390.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:00:00 | 2385.00 | 2402.23 | 2390.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 2387.30 | 2399.24 | 2390.27 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 2367.95 | 2386.36 | 2387.08 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 2432.60 | 2389.29 | 2386.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 10:15:00 | 2446.95 | 2400.82 | 2391.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 10:15:00 | 2393.00 | 2405.28 | 2399.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 10:15:00 | 2393.00 | 2405.28 | 2399.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 2393.00 | 2405.28 | 2399.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:00:00 | 2393.00 | 2405.28 | 2399.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 2400.25 | 2404.27 | 2399.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:45:00 | 2388.15 | 2404.27 | 2399.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 2390.95 | 2401.61 | 2398.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:45:00 | 2392.35 | 2401.61 | 2398.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 2378.20 | 2396.93 | 2396.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 2376.20 | 2396.93 | 2396.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 2407.85 | 2399.66 | 2398.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 2367.45 | 2399.66 | 2398.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 09:15:00 | 2382.70 | 2396.27 | 2396.72 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 2403.80 | 2393.58 | 2392.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 2410.00 | 2397.01 | 2394.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 2371.75 | 2393.72 | 2393.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 10:15:00 | 2371.75 | 2393.72 | 2393.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 2371.75 | 2393.72 | 2393.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 2371.75 | 2393.72 | 2393.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 2324.70 | 2379.91 | 2387.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 13:15:00 | 2309.60 | 2356.59 | 2374.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 14:15:00 | 2336.10 | 2333.62 | 2350.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 15:00:00 | 2336.10 | 2333.62 | 2350.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 2320.60 | 2330.64 | 2345.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 2315.10 | 2330.64 | 2345.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:15:00 | 2315.65 | 2332.88 | 2339.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:15:00 | 2312.80 | 2328.80 | 2334.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 2294.70 | 2260.04 | 2259.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 2294.70 | 2260.04 | 2259.89 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 2237.90 | 2258.94 | 2261.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 11:15:00 | 2229.85 | 2250.81 | 2256.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 2209.75 | 2192.14 | 2206.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 2209.75 | 2192.14 | 2206.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 2209.75 | 2192.14 | 2206.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 2209.75 | 2192.14 | 2206.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 2211.60 | 2196.03 | 2206.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:45:00 | 2215.00 | 2196.03 | 2206.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 2211.70 | 2199.17 | 2207.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:15:00 | 2209.60 | 2199.17 | 2207.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 2206.70 | 2200.67 | 2207.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:45:00 | 2203.20 | 2203.74 | 2207.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 15:15:00 | 2226.50 | 2210.05 | 2210.09 | SL hit (close>static) qty=1.00 sl=2215.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 2230.25 | 2214.09 | 2211.92 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 12:15:00 | 2193.85 | 2209.66 | 2210.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 10:15:00 | 2189.10 | 2196.14 | 2201.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 15:15:00 | 2194.00 | 2192.32 | 2197.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 15:15:00 | 2194.00 | 2192.32 | 2197.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 2194.00 | 2192.32 | 2197.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 2192.30 | 2192.32 | 2197.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 2201.15 | 2194.09 | 2197.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:45:00 | 2204.60 | 2194.09 | 2197.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 2193.35 | 2193.94 | 2197.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 12:00:00 | 2184.25 | 2192.00 | 2195.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 12:15:00 | 2075.04 | 2114.61 | 2143.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 2139.90 | 2115.41 | 2133.91 | SL hit (close>ema200) qty=0.50 sl=2115.41 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 1927.45 | 1898.20 | 1896.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 10:15:00 | 1942.40 | 1907.04 | 1900.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 14:15:00 | 1946.60 | 1948.68 | 1934.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 15:00:00 | 1946.60 | 1948.68 | 1934.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1960.05 | 1966.72 | 1956.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 1961.75 | 1966.72 | 1956.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 1958.55 | 1965.09 | 1956.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:30:00 | 1958.00 | 1965.09 | 1956.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1965.25 | 1965.12 | 1957.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 1973.10 | 1960.23 | 1956.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 10:15:00 | 1970.95 | 1960.68 | 1957.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 10:45:00 | 1972.90 | 1962.50 | 1958.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 14:45:00 | 1966.00 | 1965.13 | 1961.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1954.75 | 1963.06 | 1960.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-28 15:15:00 | 1954.75 | 1963.06 | 1960.66 | SL hit (close<static) qty=1.00 sl=1956.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 1939.85 | 1955.47 | 1957.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 1926.70 | 1949.72 | 1954.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 15:15:00 | 1920.40 | 1919.06 | 1926.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 15:15:00 | 1920.40 | 1919.06 | 1926.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 1920.40 | 1919.06 | 1926.94 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 1929.75 | 1928.50 | 1928.47 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 1925.65 | 1929.53 | 1929.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1919.80 | 1927.65 | 1928.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 11:15:00 | 1910.00 | 1909.25 | 1915.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 11:15:00 | 1910.00 | 1909.25 | 1915.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1910.00 | 1909.25 | 1915.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:45:00 | 1913.60 | 1909.25 | 1915.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 1914.20 | 1910.24 | 1915.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 12:45:00 | 1914.30 | 1910.24 | 1915.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1917.05 | 1911.60 | 1915.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:00:00 | 1917.05 | 1911.60 | 1915.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1918.15 | 1912.91 | 1916.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:30:00 | 1916.15 | 1912.91 | 1916.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1915.55 | 1913.44 | 1916.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 1927.70 | 1913.44 | 1916.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1919.30 | 1914.61 | 1916.38 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 1938.05 | 1919.30 | 1918.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 1947.60 | 1927.90 | 1922.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 1927.05 | 1941.10 | 1934.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 1927.05 | 1941.10 | 1934.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1927.05 | 1941.10 | 1934.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 1927.05 | 1941.10 | 1934.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1924.30 | 1937.74 | 1933.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1924.30 | 1937.74 | 1933.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1939.70 | 1936.20 | 1933.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 11:30:00 | 1944.60 | 1937.54 | 1934.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 13:45:00 | 1941.95 | 1939.16 | 1935.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:30:00 | 1940.00 | 1939.33 | 1936.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 15:00:00 | 1940.00 | 1939.33 | 1936.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 1937.70 | 1939.00 | 1936.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 1952.00 | 1939.00 | 1936.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1949.75 | 1941.15 | 1937.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 1928.25 | 1935.32 | 1936.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 1928.25 | 1935.32 | 1936.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 10:15:00 | 1900.70 | 1928.39 | 1933.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 14:15:00 | 1910.45 | 1909.58 | 1916.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 14:30:00 | 1908.90 | 1909.58 | 1916.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1913.00 | 1909.86 | 1915.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:45:00 | 1913.95 | 1909.86 | 1915.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1906.25 | 1909.14 | 1914.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:15:00 | 1900.75 | 1909.14 | 1914.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:45:00 | 1905.00 | 1896.78 | 1904.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 1925.95 | 1904.22 | 1903.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 1925.95 | 1904.22 | 1903.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 1933.55 | 1910.09 | 1906.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 11:15:00 | 2015.20 | 2015.79 | 2001.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 11:45:00 | 2018.45 | 2015.79 | 2001.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 2016.60 | 2011.91 | 2004.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 11:15:00 | 2025.60 | 2012.42 | 2005.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 11:15:00 | 1991.95 | 2003.09 | 2003.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1991.95 | 2003.09 | 2003.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 1970.00 | 1989.84 | 1996.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 15:15:00 | 1893.00 | 1888.43 | 1907.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:15:00 | 1896.00 | 1888.43 | 1907.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 1899.00 | 1891.76 | 1905.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:00:00 | 1892.45 | 1891.89 | 1904.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 1910.55 | 1897.41 | 1902.35 | SL hit (close>static) qty=1.00 sl=1905.95 alert=retest2 |

### Cycle 26 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 1905.40 | 1898.48 | 1898.20 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 1891.80 | 1899.66 | 1899.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 1887.35 | 1897.20 | 1898.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 1894.55 | 1894.23 | 1896.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 15:00:00 | 1894.55 | 1894.23 | 1896.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 1890.00 | 1893.38 | 1896.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 1885.95 | 1893.38 | 1896.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 1791.65 | 1833.28 | 1851.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 14:15:00 | 1791.45 | 1789.95 | 1805.18 | SL hit (close>ema200) qty=0.50 sl=1789.95 alert=retest2 |

### Cycle 28 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 1806.40 | 1800.72 | 1800.44 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 1793.30 | 1799.23 | 1799.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1785.55 | 1795.82 | 1798.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 09:15:00 | 1755.05 | 1753.75 | 1765.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 10:00:00 | 1755.05 | 1753.75 | 1765.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 1765.75 | 1756.80 | 1763.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:45:00 | 1766.90 | 1756.80 | 1763.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1769.95 | 1759.43 | 1763.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:15:00 | 1768.50 | 1759.43 | 1763.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 1768.50 | 1761.24 | 1764.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1780.00 | 1765.00 | 1765.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 1778.05 | 1767.61 | 1766.67 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 1739.10 | 1761.91 | 1764.17 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 11:15:00 | 1778.40 | 1763.08 | 1760.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1783.55 | 1771.69 | 1766.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 1791.00 | 1793.31 | 1782.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 10:15:00 | 1791.00 | 1793.31 | 1782.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1791.00 | 1793.31 | 1782.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 1791.05 | 1793.31 | 1782.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 1785.55 | 1791.62 | 1784.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:00:00 | 1785.55 | 1791.62 | 1784.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1789.95 | 1791.29 | 1785.06 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1771.00 | 1781.59 | 1782.08 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 14:15:00 | 1789.90 | 1783.21 | 1782.62 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1752.65 | 1777.79 | 1780.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 1730.65 | 1765.96 | 1774.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1740.80 | 1740.16 | 1756.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:00:00 | 1740.80 | 1740.16 | 1756.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1728.00 | 1738.93 | 1748.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 10:30:00 | 1727.55 | 1736.14 | 1746.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 1726.00 | 1732.76 | 1740.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:00:00 | 1725.20 | 1731.48 | 1738.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 1702.05 | 1733.61 | 1737.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1703.15 | 1727.52 | 1734.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-19 11:15:00 | 1750.30 | 1733.88 | 1731.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 1750.30 | 1733.88 | 1731.88 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 1717.85 | 1730.74 | 1731.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 1709.05 | 1726.41 | 1729.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1728.65 | 1720.56 | 1723.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 1728.65 | 1720.56 | 1723.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1728.65 | 1720.56 | 1723.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:30:00 | 1722.95 | 1720.56 | 1723.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1739.00 | 1724.25 | 1725.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 1739.00 | 1724.25 | 1725.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 1756.35 | 1730.67 | 1728.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 12:15:00 | 1762.90 | 1737.12 | 1731.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 1797.00 | 1800.19 | 1784.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:15:00 | 1787.55 | 1800.19 | 1784.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1801.80 | 1800.52 | 1786.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:30:00 | 1817.20 | 1803.29 | 1793.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 14:15:00 | 1812.75 | 1804.83 | 1797.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 12:15:00 | 1787.75 | 1795.98 | 1796.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 12:15:00 | 1787.75 | 1795.98 | 1796.16 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 1815.00 | 1798.00 | 1796.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 1838.15 | 1814.18 | 1805.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 13:15:00 | 1839.95 | 1842.26 | 1831.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 14:15:00 | 1837.05 | 1842.26 | 1831.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1836.75 | 1841.16 | 1831.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:45:00 | 1833.15 | 1841.16 | 1831.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1815.35 | 1834.85 | 1830.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 1815.35 | 1834.85 | 1830.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1815.40 | 1830.96 | 1829.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 1815.40 | 1830.96 | 1829.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 12:15:00 | 1817.75 | 1826.88 | 1827.43 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 1834.90 | 1828.71 | 1828.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 1845.75 | 1834.66 | 1831.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 1835.05 | 1838.89 | 1834.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 1835.05 | 1838.89 | 1834.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1835.05 | 1838.89 | 1834.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 1835.05 | 1838.89 | 1834.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1842.30 | 1839.57 | 1835.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 1849.50 | 1839.57 | 1835.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 13:00:00 | 1852.20 | 1850.90 | 1845.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:30:00 | 1848.35 | 1849.17 | 1846.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 11:00:00 | 1845.70 | 1850.21 | 1848.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1846.30 | 1849.42 | 1848.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-12 13:15:00 | 1838.45 | 1846.23 | 1847.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 1838.45 | 1846.23 | 1847.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 1834.80 | 1843.95 | 1846.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 1825.45 | 1825.43 | 1834.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 1825.45 | 1825.43 | 1834.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1832.70 | 1826.78 | 1833.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1832.70 | 1826.78 | 1833.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 1835.00 | 1828.43 | 1833.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 1830.75 | 1828.43 | 1833.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1839.70 | 1830.68 | 1834.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:15:00 | 1826.85 | 1831.03 | 1834.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 13:15:00 | 1845.85 | 1836.04 | 1835.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 1845.85 | 1836.04 | 1835.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 1850.65 | 1838.96 | 1837.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 11:15:00 | 1843.60 | 1843.85 | 1840.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 12:00:00 | 1843.60 | 1843.85 | 1840.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 1842.85 | 1843.65 | 1840.75 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 1830.05 | 1838.43 | 1838.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 1822.10 | 1831.19 | 1834.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1743.55 | 1734.28 | 1753.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 1743.55 | 1734.28 | 1753.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1659.10 | 1653.58 | 1667.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:45:00 | 1666.55 | 1653.58 | 1667.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1648.35 | 1650.96 | 1657.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 10:00:00 | 1641.40 | 1649.05 | 1653.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 11:15:00 | 1641.00 | 1647.90 | 1652.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 12:00:00 | 1640.85 | 1646.49 | 1651.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 12:45:00 | 1640.60 | 1645.59 | 1650.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 1647.35 | 1646.34 | 1650.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 14:30:00 | 1649.00 | 1646.34 | 1650.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 1647.75 | 1646.62 | 1649.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 1645.40 | 1646.62 | 1649.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1635.35 | 1644.36 | 1648.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 10:15:00 | 1631.30 | 1644.36 | 1648.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 12:15:00 | 1634.00 | 1643.51 | 1647.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 14:15:00 | 1559.33 | 1588.36 | 1612.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 14:15:00 | 1558.95 | 1588.36 | 1612.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 14:15:00 | 1558.81 | 1588.36 | 1612.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 14:15:00 | 1558.57 | 1588.36 | 1612.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 14:15:00 | 1549.73 | 1588.36 | 1612.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 14:15:00 | 1552.30 | 1588.36 | 1612.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-07 10:15:00 | 1582.75 | 1581.66 | 1603.05 | SL hit (close>ema200) qty=0.50 sl=1581.66 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 1513.10 | 1500.92 | 1500.56 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 1495.90 | 1499.92 | 1500.13 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 11:15:00 | 1505.60 | 1501.05 | 1500.63 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 1484.20 | 1498.29 | 1499.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 1476.55 | 1484.07 | 1489.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1456.85 | 1456.26 | 1468.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 14:30:00 | 1459.95 | 1456.26 | 1468.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1479.30 | 1461.60 | 1469.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1479.30 | 1461.60 | 1469.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1477.10 | 1464.70 | 1469.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1478.60 | 1464.70 | 1469.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 1497.90 | 1473.22 | 1472.96 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 1463.15 | 1474.66 | 1475.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1437.35 | 1465.20 | 1470.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 13:15:00 | 1452.85 | 1451.04 | 1461.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 13:45:00 | 1450.90 | 1451.04 | 1461.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 1464.75 | 1453.78 | 1461.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 1464.75 | 1453.78 | 1461.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 1464.40 | 1455.90 | 1461.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 1464.90 | 1455.90 | 1461.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 1456.30 | 1455.98 | 1461.22 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 12:15:00 | 1491.35 | 1469.21 | 1466.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 1523.00 | 1485.45 | 1479.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 1505.05 | 1509.44 | 1502.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 1505.05 | 1509.44 | 1502.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1505.05 | 1509.44 | 1502.93 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 1489.35 | 1497.74 | 1498.83 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 1508.45 | 1499.20 | 1499.05 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 11:15:00 | 1496.10 | 1498.58 | 1498.78 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 1505.00 | 1499.86 | 1499.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 13:15:00 | 1525.85 | 1509.31 | 1504.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 1500.90 | 1511.11 | 1507.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 1500.90 | 1511.11 | 1507.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1500.90 | 1511.11 | 1507.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1506.00 | 1511.11 | 1507.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1500.15 | 1508.92 | 1506.54 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 1487.85 | 1504.70 | 1504.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 12:15:00 | 1479.00 | 1499.56 | 1502.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 1361.00 | 1340.64 | 1358.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 12:15:00 | 1361.00 | 1340.64 | 1358.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 1361.00 | 1340.64 | 1358.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 1358.70 | 1340.64 | 1358.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1376.00 | 1347.71 | 1360.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:00:00 | 1376.00 | 1347.71 | 1360.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1385.90 | 1355.35 | 1362.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1385.90 | 1355.35 | 1362.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 1358.15 | 1361.43 | 1364.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 1358.15 | 1361.43 | 1364.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 12:15:00 | 1352.00 | 1359.15 | 1362.68 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 1371.70 | 1364.61 | 1363.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 1373.15 | 1366.31 | 1364.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1376.95 | 1387.13 | 1379.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1376.95 | 1387.13 | 1379.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1376.95 | 1387.13 | 1379.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1376.95 | 1387.13 | 1379.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1383.45 | 1386.40 | 1379.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:30:00 | 1402.00 | 1390.53 | 1384.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:30:00 | 1397.30 | 1398.96 | 1392.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 1380.50 | 1389.42 | 1390.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 1380.50 | 1389.42 | 1390.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 1370.80 | 1384.52 | 1387.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 1348.45 | 1345.88 | 1358.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 1348.45 | 1345.88 | 1358.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 1330.30 | 1316.49 | 1327.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 1330.30 | 1316.49 | 1327.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 1319.35 | 1317.06 | 1326.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 13:30:00 | 1317.70 | 1317.18 | 1326.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 14:00:00 | 1317.65 | 1317.18 | 1326.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 15:00:00 | 1316.85 | 1317.11 | 1325.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 15:15:00 | 1333.00 | 1327.24 | 1326.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 15:15:00 | 1333.00 | 1327.24 | 1326.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 1339.90 | 1329.77 | 1328.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 1332.60 | 1333.97 | 1331.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 15:15:00 | 1332.60 | 1333.97 | 1331.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1332.60 | 1333.97 | 1331.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 1324.35 | 1333.97 | 1331.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1323.25 | 1331.83 | 1330.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 1323.25 | 1331.83 | 1330.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 1312.85 | 1328.03 | 1329.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 1303.55 | 1323.13 | 1326.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1253.50 | 1250.21 | 1263.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 10:00:00 | 1253.50 | 1250.21 | 1263.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1253.45 | 1245.09 | 1253.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 1252.90 | 1245.09 | 1253.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1254.10 | 1246.89 | 1253.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:00:00 | 1254.10 | 1246.89 | 1253.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 1259.25 | 1249.36 | 1254.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 1259.25 | 1249.36 | 1254.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 1253.30 | 1250.15 | 1253.97 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 1270.55 | 1256.46 | 1255.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 1294.85 | 1272.97 | 1265.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 1315.00 | 1317.01 | 1300.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 15:00:00 | 1315.00 | 1317.01 | 1300.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1298.50 | 1312.35 | 1300.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1298.50 | 1312.35 | 1300.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1292.30 | 1308.34 | 1300.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 1289.65 | 1308.34 | 1300.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1289.30 | 1304.53 | 1299.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 1289.90 | 1304.53 | 1299.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1291.00 | 1296.90 | 1296.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 1292.70 | 1296.90 | 1296.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 1287.70 | 1295.06 | 1295.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 1278.00 | 1286.63 | 1290.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 1294.95 | 1284.80 | 1287.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 1294.95 | 1284.80 | 1287.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1294.95 | 1284.80 | 1287.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1294.95 | 1284.80 | 1287.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1290.60 | 1285.96 | 1287.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1299.85 | 1285.96 | 1287.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1294.65 | 1288.12 | 1288.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:45:00 | 1297.40 | 1288.12 | 1288.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 1290.95 | 1288.68 | 1288.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 1285.00 | 1288.68 | 1288.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 12:15:00 | 1296.80 | 1290.31 | 1289.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 12:15:00 | 1296.80 | 1290.31 | 1289.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 1306.00 | 1294.34 | 1291.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1337.45 | 1345.13 | 1332.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:00:00 | 1337.45 | 1345.13 | 1332.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 1330.95 | 1340.87 | 1332.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 1330.95 | 1340.87 | 1332.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 1332.65 | 1339.22 | 1332.73 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1276.85 | 1326.12 | 1328.74 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 14:15:00 | 1296.00 | 1274.58 | 1271.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 15:15:00 | 1296.20 | 1278.91 | 1274.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1300.60 | 1303.85 | 1292.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1300.60 | 1303.85 | 1292.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1300.60 | 1303.85 | 1292.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 1299.10 | 1303.85 | 1292.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1359.90 | 1384.06 | 1378.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1359.90 | 1384.06 | 1378.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1337.40 | 1374.73 | 1374.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1337.40 | 1374.73 | 1374.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1347.20 | 1369.22 | 1372.08 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 1371.70 | 1367.38 | 1367.04 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 1361.40 | 1366.48 | 1366.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 1353.40 | 1362.41 | 1364.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1336.30 | 1330.83 | 1342.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:00:00 | 1336.30 | 1330.83 | 1342.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1339.80 | 1333.13 | 1341.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1339.80 | 1333.13 | 1341.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1340.50 | 1334.60 | 1341.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:00:00 | 1340.50 | 1334.60 | 1341.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 1337.30 | 1335.14 | 1341.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 1320.40 | 1334.35 | 1339.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1254.38 | 1286.79 | 1298.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 1275.90 | 1274.64 | 1285.89 | SL hit (close>ema200) qty=0.50 sl=1274.64 alert=retest2 |

### Cycle 70 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1328.30 | 1294.47 | 1293.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1345.10 | 1325.13 | 1311.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 1320.00 | 1336.56 | 1326.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 1320.00 | 1336.56 | 1326.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1320.00 | 1336.56 | 1326.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:45:00 | 1314.50 | 1336.56 | 1326.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 1322.10 | 1333.67 | 1325.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:45:00 | 1312.90 | 1333.67 | 1325.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 1324.00 | 1331.73 | 1325.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 1320.20 | 1331.73 | 1325.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 1328.60 | 1331.11 | 1326.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 1319.90 | 1331.11 | 1326.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 1325.60 | 1330.00 | 1326.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1341.10 | 1329.10 | 1326.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 09:15:00 | 1358.50 | 1374.03 | 1375.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 1358.50 | 1374.03 | 1375.13 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 1426.90 | 1382.58 | 1377.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 1438.50 | 1399.46 | 1386.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 10:15:00 | 1511.80 | 1515.90 | 1496.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 11:00:00 | 1511.80 | 1515.90 | 1496.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1503.70 | 1516.38 | 1506.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 1503.70 | 1516.38 | 1506.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1501.00 | 1513.31 | 1506.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:45:00 | 1500.00 | 1513.31 | 1506.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1503.20 | 1510.07 | 1505.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 1501.80 | 1510.07 | 1505.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1500.10 | 1508.08 | 1505.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1500.10 | 1508.08 | 1505.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1503.20 | 1507.10 | 1505.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1503.60 | 1507.10 | 1505.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1513.80 | 1507.46 | 1505.54 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 1500.10 | 1504.39 | 1504.91 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 1508.60 | 1504.03 | 1503.94 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 1492.50 | 1503.44 | 1503.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 1484.40 | 1499.64 | 1502.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 1494.90 | 1494.56 | 1498.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 1494.90 | 1494.56 | 1498.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1504.10 | 1496.47 | 1498.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 1504.10 | 1496.47 | 1498.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1510.00 | 1499.18 | 1499.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:45:00 | 1511.20 | 1499.18 | 1499.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 1521.20 | 1503.58 | 1501.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1534.30 | 1513.23 | 1506.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 1514.40 | 1519.87 | 1512.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 13:15:00 | 1514.40 | 1519.87 | 1512.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1514.40 | 1519.87 | 1512.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 1514.60 | 1519.87 | 1512.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1510.60 | 1518.02 | 1512.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 1510.60 | 1518.02 | 1512.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1509.90 | 1516.40 | 1512.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 1508.60 | 1516.40 | 1512.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1506.40 | 1514.40 | 1511.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 1508.70 | 1514.40 | 1511.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1509.00 | 1513.32 | 1511.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:00:00 | 1518.00 | 1513.48 | 1511.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1513.00 | 1523.93 | 1524.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 1513.00 | 1523.93 | 1524.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1499.70 | 1516.63 | 1521.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 1508.10 | 1507.99 | 1514.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 10:45:00 | 1512.50 | 1507.99 | 1514.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 1510.10 | 1508.41 | 1514.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:30:00 | 1516.80 | 1508.41 | 1514.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 1513.50 | 1509.43 | 1514.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:45:00 | 1512.40 | 1509.43 | 1514.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1524.10 | 1512.36 | 1515.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 1524.10 | 1512.36 | 1515.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1529.50 | 1515.79 | 1516.56 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 15:15:00 | 1528.80 | 1518.39 | 1517.67 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 1506.10 | 1515.93 | 1516.62 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1538.70 | 1520.59 | 1518.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 1540.80 | 1524.64 | 1520.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 1560.00 | 1571.34 | 1561.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 15:15:00 | 1560.00 | 1571.34 | 1561.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1560.00 | 1571.34 | 1561.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1558.00 | 1571.34 | 1561.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1533.00 | 1563.67 | 1558.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 1533.00 | 1563.67 | 1558.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1543.00 | 1559.54 | 1557.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 1538.00 | 1559.54 | 1557.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1538.00 | 1555.23 | 1555.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1524.00 | 1548.98 | 1552.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1547.70 | 1535.41 | 1543.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1547.70 | 1535.41 | 1543.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1547.70 | 1535.41 | 1543.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1547.70 | 1535.41 | 1543.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1552.60 | 1538.85 | 1543.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1553.80 | 1538.85 | 1543.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1561.00 | 1547.20 | 1546.82 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 1495.50 | 1537.31 | 1542.41 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 14:15:00 | 1523.00 | 1513.34 | 1512.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1538.00 | 1519.34 | 1515.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1515.50 | 1520.48 | 1517.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 12:15:00 | 1515.50 | 1520.48 | 1517.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1515.50 | 1520.48 | 1517.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:45:00 | 1518.60 | 1520.48 | 1517.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1512.00 | 1518.78 | 1516.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:00:00 | 1512.00 | 1518.78 | 1516.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 1500.70 | 1515.16 | 1515.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 1496.20 | 1508.94 | 1512.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 1509.00 | 1507.29 | 1510.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 14:15:00 | 1509.00 | 1507.29 | 1510.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1509.00 | 1507.29 | 1510.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:30:00 | 1511.00 | 1507.29 | 1510.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1496.60 | 1505.09 | 1508.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 11:00:00 | 1495.70 | 1503.21 | 1507.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:30:00 | 1494.50 | 1500.27 | 1505.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:00:00 | 1494.90 | 1494.77 | 1499.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 1493.00 | 1494.28 | 1496.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1498.50 | 1494.52 | 1496.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 1503.40 | 1494.52 | 1496.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1503.50 | 1496.32 | 1496.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 1506.00 | 1496.32 | 1496.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1493.80 | 1495.81 | 1496.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 1496.10 | 1495.81 | 1496.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1486.20 | 1484.74 | 1487.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1486.20 | 1484.74 | 1487.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1486.10 | 1485.01 | 1487.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1484.80 | 1485.01 | 1487.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1477.00 | 1483.41 | 1486.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 1475.30 | 1483.41 | 1486.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 1491.60 | 1483.46 | 1484.89 | SL hit (close>static) qty=1.00 sl=1488.70 alert=retest2 |

### Cycle 86 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 1493.70 | 1486.33 | 1485.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 1501.90 | 1491.59 | 1488.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 1493.10 | 1493.67 | 1490.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 1493.10 | 1493.67 | 1490.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1489.00 | 1492.73 | 1490.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 1489.00 | 1492.73 | 1490.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1493.00 | 1492.79 | 1490.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 1493.00 | 1492.79 | 1490.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1492.90 | 1492.81 | 1490.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1491.40 | 1492.81 | 1490.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1481.00 | 1490.45 | 1490.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 1481.00 | 1490.45 | 1490.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1475.20 | 1487.40 | 1488.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 11:15:00 | 1469.40 | 1476.07 | 1481.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 12:15:00 | 1478.00 | 1476.46 | 1480.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 13:00:00 | 1478.00 | 1476.46 | 1480.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1493.40 | 1479.85 | 1481.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 1493.40 | 1479.85 | 1481.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1496.50 | 1483.18 | 1483.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 1496.50 | 1483.18 | 1483.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 1494.70 | 1485.48 | 1484.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 1498.40 | 1492.40 | 1488.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 1495.20 | 1497.21 | 1493.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 12:45:00 | 1494.50 | 1497.21 | 1493.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1496.80 | 1497.13 | 1493.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:45:00 | 1492.20 | 1497.13 | 1493.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1493.20 | 1496.35 | 1493.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:45:00 | 1493.30 | 1496.35 | 1493.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1490.00 | 1495.08 | 1493.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1505.00 | 1495.08 | 1493.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 14:15:00 | 1500.10 | 1503.96 | 1504.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 1500.10 | 1503.96 | 1504.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1496.20 | 1502.21 | 1503.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 1478.00 | 1467.71 | 1478.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 1478.00 | 1467.71 | 1478.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1478.00 | 1467.71 | 1478.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 1478.00 | 1467.71 | 1478.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 1475.40 | 1469.25 | 1478.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 1459.20 | 1469.25 | 1478.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1467.70 | 1468.94 | 1477.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 1455.00 | 1465.44 | 1471.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 12:15:00 | 1452.30 | 1463.90 | 1469.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 14:15:00 | 1428.40 | 1416.13 | 1415.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1428.40 | 1416.13 | 1415.20 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 1412.90 | 1417.43 | 1417.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1403.90 | 1414.73 | 1416.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1416.50 | 1409.89 | 1413.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 1416.50 | 1409.89 | 1413.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1416.50 | 1409.89 | 1413.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1416.50 | 1409.89 | 1413.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1414.00 | 1410.71 | 1413.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1408.00 | 1410.71 | 1413.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 09:15:00 | 1337.60 | 1362.11 | 1379.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-12 14:15:00 | 1267.20 | 1303.95 | 1340.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 92 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 1320.30 | 1304.57 | 1304.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1326.20 | 1312.89 | 1308.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 15:15:00 | 1422.00 | 1424.08 | 1408.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:15:00 | 1417.30 | 1424.08 | 1408.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1414.70 | 1419.51 | 1413.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1404.60 | 1419.51 | 1413.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1397.10 | 1415.03 | 1411.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1397.00 | 1415.03 | 1411.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1390.60 | 1410.15 | 1409.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 1390.00 | 1410.15 | 1409.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1393.60 | 1406.84 | 1408.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1370.10 | 1392.20 | 1400.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 1366.60 | 1366.20 | 1378.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:45:00 | 1365.20 | 1366.20 | 1378.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1385.30 | 1367.02 | 1373.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1383.90 | 1367.02 | 1373.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1386.50 | 1370.92 | 1374.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1386.50 | 1370.92 | 1374.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 1394.00 | 1378.99 | 1378.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 1407.40 | 1387.47 | 1382.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 1421.80 | 1423.01 | 1413.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 09:15:00 | 1413.80 | 1423.01 | 1413.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1411.00 | 1420.61 | 1413.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 1411.00 | 1420.61 | 1413.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1410.30 | 1418.55 | 1413.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 1409.50 | 1418.55 | 1413.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1411.80 | 1417.20 | 1412.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 1414.00 | 1416.56 | 1413.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 1438.60 | 1446.78 | 1447.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 1438.60 | 1446.78 | 1447.43 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 1456.70 | 1448.44 | 1447.93 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 1440.80 | 1446.40 | 1447.14 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1466.80 | 1449.46 | 1448.34 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 15:15:00 | 1457.60 | 1462.70 | 1462.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 1456.00 | 1461.36 | 1462.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 1457.80 | 1457.22 | 1459.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 15:00:00 | 1457.80 | 1457.22 | 1459.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1462.70 | 1458.32 | 1459.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 1466.60 | 1458.32 | 1459.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1458.60 | 1458.38 | 1459.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 1456.90 | 1457.74 | 1459.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1384.06 | 1395.60 | 1409.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 1366.10 | 1364.57 | 1377.25 | SL hit (close>ema200) qty=0.50 sl=1364.57 alert=retest2 |

### Cycle 100 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 1385.10 | 1374.87 | 1373.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 15:15:00 | 1389.00 | 1381.15 | 1377.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1404.60 | 1406.86 | 1396.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 11:00:00 | 1404.60 | 1406.86 | 1396.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1406.20 | 1406.40 | 1398.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:30:00 | 1401.80 | 1406.40 | 1398.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1400.00 | 1404.21 | 1399.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 1399.10 | 1403.19 | 1399.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1398.70 | 1402.29 | 1399.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 1398.90 | 1402.29 | 1399.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1399.80 | 1401.79 | 1399.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 1395.70 | 1401.79 | 1399.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1400.90 | 1401.61 | 1399.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 1397.40 | 1401.61 | 1399.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1398.00 | 1402.20 | 1400.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1393.30 | 1402.20 | 1400.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1411.90 | 1404.14 | 1401.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 1394.80 | 1404.14 | 1401.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1408.40 | 1417.67 | 1411.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 1408.40 | 1417.67 | 1411.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1407.60 | 1415.66 | 1411.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:15:00 | 1404.30 | 1415.66 | 1411.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1408.60 | 1414.25 | 1410.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 1412.20 | 1414.16 | 1411.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:15:00 | 1412.30 | 1413.75 | 1412.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 1431.30 | 1444.55 | 1446.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 1431.30 | 1444.55 | 1446.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 1415.00 | 1430.65 | 1437.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 1435.70 | 1431.66 | 1437.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 1435.70 | 1431.66 | 1437.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1440.00 | 1433.33 | 1437.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1432.80 | 1433.33 | 1437.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1447.00 | 1436.06 | 1438.64 | SL hit (close>static) qty=1.00 sl=1440.60 alert=retest2 |

### Cycle 102 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 1444.40 | 1440.44 | 1440.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 1454.90 | 1446.47 | 1443.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1454.50 | 1457.84 | 1452.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:00:00 | 1454.50 | 1457.84 | 1452.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 1467.00 | 1459.60 | 1454.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:30:00 | 1454.50 | 1459.60 | 1454.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1462.80 | 1462.08 | 1457.46 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 1447.60 | 1455.05 | 1455.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1442.60 | 1452.56 | 1454.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 1461.60 | 1454.37 | 1455.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 1461.60 | 1454.37 | 1455.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1461.60 | 1454.37 | 1455.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1465.00 | 1454.37 | 1455.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1458.40 | 1455.17 | 1455.40 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 1465.00 | 1457.14 | 1456.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1480.10 | 1461.73 | 1458.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 1467.00 | 1469.55 | 1465.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 13:15:00 | 1467.00 | 1469.55 | 1465.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1467.00 | 1469.55 | 1465.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 1467.00 | 1469.55 | 1465.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1467.50 | 1469.14 | 1465.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 1464.50 | 1469.14 | 1465.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1466.00 | 1468.51 | 1465.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1523.00 | 1468.51 | 1465.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 13:15:00 | 1552.10 | 1562.81 | 1563.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 1552.10 | 1562.81 | 1563.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 1520.10 | 1551.42 | 1557.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 1448.50 | 1447.19 | 1462.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 1448.50 | 1447.19 | 1462.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 1460.90 | 1452.41 | 1460.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 1462.10 | 1452.41 | 1460.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 1461.70 | 1454.27 | 1460.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:30:00 | 1462.00 | 1454.27 | 1460.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 1461.70 | 1455.75 | 1460.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 1461.10 | 1455.75 | 1460.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1463.40 | 1457.28 | 1460.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 1463.40 | 1457.28 | 1460.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1463.00 | 1458.43 | 1461.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 1464.70 | 1458.43 | 1461.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1457.30 | 1458.20 | 1460.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 1460.10 | 1458.20 | 1460.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1450.40 | 1452.60 | 1456.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 1455.70 | 1452.60 | 1456.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1449.70 | 1450.90 | 1454.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:30:00 | 1452.80 | 1450.90 | 1454.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1450.90 | 1450.90 | 1454.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:15:00 | 1448.60 | 1450.90 | 1454.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1474.30 | 1455.58 | 1455.93 | SL hit (close>static) qty=1.00 sl=1455.80 alert=retest2 |

### Cycle 106 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 1486.00 | 1461.66 | 1458.66 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 1446.30 | 1463.34 | 1464.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 1439.60 | 1454.36 | 1459.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1439.90 | 1438.95 | 1447.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 1439.90 | 1438.95 | 1447.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1433.70 | 1418.33 | 1424.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1433.70 | 1418.33 | 1424.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1430.60 | 1420.78 | 1425.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1434.60 | 1420.78 | 1425.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1438.80 | 1426.18 | 1427.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 1438.80 | 1426.18 | 1427.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 1444.50 | 1429.85 | 1428.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 1447.90 | 1439.40 | 1434.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1433.00 | 1447.01 | 1441.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1433.00 | 1447.01 | 1441.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1433.00 | 1447.01 | 1441.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1433.00 | 1447.01 | 1441.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1433.70 | 1444.35 | 1440.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1433.70 | 1444.35 | 1440.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 1428.00 | 1436.53 | 1437.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 1412.80 | 1427.20 | 1431.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 12:15:00 | 1404.30 | 1402.28 | 1412.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 12:45:00 | 1403.40 | 1402.28 | 1412.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1411.60 | 1405.05 | 1411.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 1411.50 | 1405.05 | 1411.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1412.50 | 1406.54 | 1411.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 1417.20 | 1406.54 | 1411.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1413.00 | 1407.83 | 1411.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 1413.00 | 1407.83 | 1411.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1413.50 | 1408.97 | 1411.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 1409.20 | 1409.05 | 1411.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 1417.60 | 1411.55 | 1412.33 | SL hit (close>static) qty=1.00 sl=1416.40 alert=retest2 |

### Cycle 110 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 1418.90 | 1412.92 | 1412.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 1425.50 | 1416.96 | 1414.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 1444.00 | 1451.20 | 1439.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 1444.00 | 1451.20 | 1439.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1432.50 | 1447.46 | 1438.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 1432.50 | 1447.46 | 1438.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1435.90 | 1445.15 | 1438.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 1435.90 | 1445.15 | 1438.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1435.30 | 1442.64 | 1438.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 1435.30 | 1442.64 | 1438.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1429.90 | 1440.09 | 1437.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 1431.30 | 1440.09 | 1437.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1414.80 | 1432.62 | 1434.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 1407.20 | 1421.72 | 1428.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 1417.40 | 1415.22 | 1421.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 13:45:00 | 1416.70 | 1415.22 | 1421.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1423.00 | 1416.78 | 1421.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1423.00 | 1416.78 | 1421.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1420.30 | 1417.48 | 1421.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1429.80 | 1417.48 | 1421.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1429.30 | 1419.85 | 1421.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:30:00 | 1423.00 | 1420.72 | 1422.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:15:00 | 1351.85 | 1368.46 | 1380.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 1379.50 | 1365.14 | 1372.84 | SL hit (close>ema200) qty=0.50 sl=1365.14 alert=retest2 |

### Cycle 112 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1389.30 | 1378.46 | 1377.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 15:15:00 | 1391.50 | 1381.07 | 1378.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1482.90 | 1485.78 | 1468.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1482.90 | 1485.78 | 1468.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1482.40 | 1494.90 | 1485.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1482.40 | 1494.90 | 1485.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1472.40 | 1490.40 | 1483.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1472.40 | 1490.40 | 1483.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1475.70 | 1487.46 | 1483.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1475.70 | 1487.46 | 1483.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1475.50 | 1482.96 | 1481.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 1465.80 | 1482.96 | 1481.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 1476.50 | 1480.67 | 1480.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 1466.60 | 1477.85 | 1479.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 1442.00 | 1440.68 | 1451.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 1442.00 | 1440.68 | 1451.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1434.90 | 1439.52 | 1449.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:45:00 | 1435.30 | 1439.52 | 1449.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1447.30 | 1437.62 | 1443.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 1449.20 | 1437.62 | 1443.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1467.30 | 1443.55 | 1445.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 1467.30 | 1443.55 | 1445.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 1463.40 | 1449.55 | 1447.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1491.00 | 1463.04 | 1455.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 1462.00 | 1468.16 | 1461.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 15:00:00 | 1462.00 | 1468.16 | 1461.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1462.00 | 1466.93 | 1461.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1449.00 | 1466.93 | 1461.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1447.30 | 1463.00 | 1460.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1447.30 | 1463.00 | 1460.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1442.50 | 1458.90 | 1458.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 1440.70 | 1458.90 | 1458.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1436.10 | 1454.34 | 1456.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 1427.80 | 1449.03 | 1453.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1390.60 | 1370.67 | 1392.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1390.60 | 1370.67 | 1392.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1390.60 | 1370.67 | 1392.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1390.60 | 1370.67 | 1392.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 1392.20 | 1376.60 | 1391.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:00:00 | 1392.20 | 1376.60 | 1391.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1391.00 | 1379.48 | 1391.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:45:00 | 1392.00 | 1379.48 | 1391.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1399.60 | 1383.50 | 1392.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 1402.50 | 1383.50 | 1392.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1406.70 | 1388.14 | 1393.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 1406.70 | 1388.14 | 1393.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 1411.40 | 1397.09 | 1396.85 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1389.40 | 1395.67 | 1396.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1384.10 | 1393.36 | 1395.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 15:15:00 | 1399.50 | 1393.47 | 1394.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 15:15:00 | 1399.50 | 1393.47 | 1394.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1399.50 | 1393.47 | 1394.88 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 1428.80 | 1400.20 | 1396.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 1445.50 | 1409.26 | 1401.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1465.90 | 1473.55 | 1459.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1465.90 | 1473.55 | 1459.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1465.90 | 1473.55 | 1459.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1465.90 | 1473.55 | 1459.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1467.00 | 1472.24 | 1460.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1450.00 | 1472.24 | 1460.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1460.50 | 1468.86 | 1460.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 1460.00 | 1468.86 | 1460.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1461.70 | 1467.43 | 1460.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 1465.10 | 1467.43 | 1460.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1454.00 | 1464.74 | 1460.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 1453.90 | 1464.74 | 1460.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1428.70 | 1457.54 | 1457.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:45:00 | 1424.40 | 1457.54 | 1457.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1438.60 | 1453.75 | 1455.74 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1467.50 | 1455.87 | 1455.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1486.70 | 1462.03 | 1458.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 1492.60 | 1494.81 | 1483.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 12:45:00 | 1488.60 | 1494.81 | 1483.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1484.30 | 1498.43 | 1494.17 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 1476.00 | 1488.86 | 1490.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 14:15:00 | 1470.80 | 1485.25 | 1488.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1480.80 | 1480.28 | 1485.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:45:00 | 1480.00 | 1480.28 | 1485.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 1482.20 | 1480.67 | 1484.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:45:00 | 1482.50 | 1480.67 | 1484.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 1489.20 | 1482.37 | 1485.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 1489.20 | 1482.37 | 1485.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 1492.40 | 1484.38 | 1486.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:30:00 | 1496.80 | 1484.38 | 1486.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 1505.00 | 1490.24 | 1488.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 1526.20 | 1500.72 | 1493.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 15:15:00 | 1588.00 | 1589.63 | 1567.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:15:00 | 1567.00 | 1589.63 | 1567.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1569.50 | 1585.61 | 1567.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1569.80 | 1585.61 | 1567.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1576.50 | 1583.78 | 1568.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 1588.20 | 1585.19 | 1570.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 1587.10 | 1586.66 | 1574.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 1617.40 | 1627.64 | 1628.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1617.40 | 1627.64 | 1628.19 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 1641.00 | 1630.80 | 1629.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1643.00 | 1635.12 | 1632.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 14:15:00 | 1634.10 | 1636.09 | 1634.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 1634.10 | 1636.09 | 1634.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1634.10 | 1636.09 | 1634.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 1636.60 | 1636.09 | 1634.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1634.00 | 1635.67 | 1634.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 1638.10 | 1635.67 | 1634.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1635.40 | 1635.62 | 1634.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 1648.40 | 1638.17 | 1635.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:45:00 | 1647.30 | 1642.43 | 1638.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1651.70 | 1668.40 | 1668.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1651.70 | 1668.40 | 1668.62 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 1694.70 | 1658.64 | 1655.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 1698.50 | 1672.43 | 1662.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 1677.60 | 1681.77 | 1671.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 1634.80 | 1681.77 | 1671.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1623.50 | 1670.11 | 1666.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 1623.50 | 1670.11 | 1666.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 1618.60 | 1659.81 | 1662.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 11:15:00 | 1608.30 | 1649.51 | 1657.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 1653.00 | 1629.41 | 1641.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 1653.00 | 1629.41 | 1641.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1653.00 | 1629.41 | 1641.94 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 1658.00 | 1647.87 | 1647.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 1665.00 | 1651.30 | 1649.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1664.00 | 1698.35 | 1680.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1664.00 | 1698.35 | 1680.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1664.00 | 1698.35 | 1680.76 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1637.70 | 1673.55 | 1676.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 1633.70 | 1665.58 | 1672.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1642.40 | 1622.96 | 1635.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1642.40 | 1622.96 | 1635.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1642.40 | 1622.96 | 1635.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 1640.80 | 1622.96 | 1635.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1649.90 | 1628.34 | 1636.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:30:00 | 1640.60 | 1631.18 | 1636.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 1652.40 | 1641.59 | 1640.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 1652.40 | 1641.59 | 1640.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 1662.60 | 1650.75 | 1645.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1639.30 | 1658.30 | 1652.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1639.30 | 1658.30 | 1652.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1639.30 | 1658.30 | 1652.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1637.70 | 1658.30 | 1652.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1628.80 | 1652.40 | 1650.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1628.80 | 1652.40 | 1650.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1630.10 | 1647.94 | 1648.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 15:15:00 | 1625.00 | 1635.46 | 1639.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1588.00 | 1576.24 | 1592.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1588.00 | 1576.24 | 1592.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1588.00 | 1576.24 | 1592.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 1591.30 | 1576.24 | 1592.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1597.00 | 1580.39 | 1593.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1595.70 | 1580.39 | 1593.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1593.20 | 1582.95 | 1593.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 1598.80 | 1582.95 | 1593.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1592.80 | 1584.92 | 1593.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1644.60 | 1584.92 | 1593.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1656.10 | 1599.16 | 1598.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1673.60 | 1614.05 | 1605.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1630.60 | 1639.13 | 1625.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1630.60 | 1639.13 | 1625.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1630.60 | 1639.13 | 1625.22 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1608.80 | 1622.47 | 1623.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 13:15:00 | 1582.90 | 1604.68 | 1611.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 1550.40 | 1539.07 | 1565.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 1550.40 | 1539.07 | 1565.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1570.00 | 1545.26 | 1565.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1491.20 | 1545.26 | 1565.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1568.90 | 1542.40 | 1541.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1568.90 | 1542.40 | 1541.45 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 13:15:00 | 1516.20 | 1537.04 | 1539.61 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 11:15:00 | 1562.30 | 1542.16 | 1540.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 12:15:00 | 1563.00 | 1546.33 | 1542.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 1622.80 | 1626.44 | 1613.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 13:30:00 | 1621.80 | 1626.44 | 1613.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1574.70 | 1615.53 | 1611.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 1574.70 | 1615.53 | 1611.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 1570.60 | 1606.54 | 1608.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 1556.80 | 1596.60 | 1603.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 10:15:00 | 1586.30 | 1580.96 | 1590.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 10:15:00 | 1586.30 | 1580.96 | 1590.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 1586.30 | 1580.96 | 1590.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:00:00 | 1586.30 | 1580.96 | 1590.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1608.40 | 1586.31 | 1589.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 15:00:00 | 1608.40 | 1586.31 | 1589.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 1628.00 | 1594.65 | 1593.21 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1590.90 | 1593.01 | 1593.21 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1600.60 | 1594.53 | 1593.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1604.90 | 1596.60 | 1594.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 11:15:00 | 1592.50 | 1595.78 | 1594.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 11:15:00 | 1592.50 | 1595.78 | 1594.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1592.50 | 1595.78 | 1594.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:00:00 | 1592.50 | 1595.78 | 1594.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 1585.80 | 1593.78 | 1593.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 1581.00 | 1589.66 | 1591.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 10:15:00 | 1593.10 | 1589.25 | 1591.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 10:15:00 | 1593.10 | 1589.25 | 1591.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 1593.10 | 1589.25 | 1591.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:00:00 | 1593.10 | 1589.25 | 1591.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 1588.80 | 1589.16 | 1590.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:00:00 | 1588.80 | 1589.16 | 1590.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 1580.00 | 1587.33 | 1589.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:30:00 | 1576.20 | 1587.46 | 1589.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 1573.90 | 1587.46 | 1589.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:00:00 | 1576.70 | 1579.94 | 1584.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:30:00 | 1576.10 | 1579.09 | 1583.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1581.50 | 1579.72 | 1583.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:30:00 | 1561.60 | 1576.45 | 1581.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 1566.00 | 1567.69 | 1573.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:00:00 | 1561.80 | 1566.40 | 1571.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:00:00 | 1564.40 | 1554.36 | 1556.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1559.40 | 1555.94 | 1556.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:45:00 | 1559.00 | 1555.94 | 1556.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1548.50 | 1554.45 | 1556.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1531.80 | 1554.59 | 1555.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1569.50 | 1541.44 | 1545.08 | SL hit (close>static) qty=1.00 sl=1559.60 alert=retest2 |

### Cycle 142 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1553.70 | 1547.81 | 1547.58 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 1539.20 | 1547.30 | 1548.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 13:15:00 | 1534.10 | 1544.66 | 1547.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1542.40 | 1540.28 | 1544.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1542.40 | 1540.28 | 1544.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1542.40 | 1540.28 | 1544.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:15:00 | 1547.60 | 1540.28 | 1544.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1564.50 | 1545.13 | 1545.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 1564.50 | 1545.13 | 1545.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1571.80 | 1550.46 | 1548.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1574.20 | 1561.27 | 1554.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 1569.70 | 1575.98 | 1567.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 14:15:00 | 1569.70 | 1575.98 | 1567.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 1569.70 | 1575.98 | 1567.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 1569.70 | 1575.98 | 1567.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1588.10 | 1578.76 | 1570.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:45:00 | 1590.90 | 1581.66 | 1573.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:15:00 | 1590.60 | 1581.66 | 1573.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-28 09:15:00 | 2187.85 | 2024-05-28 13:15:00 | 2163.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-06-10 12:30:00 | 2148.90 | 2024-06-19 10:15:00 | 2219.00 | STOP_HIT | 1.00 | 3.26% |
| BUY | retest2 | 2024-06-10 13:15:00 | 2152.90 | 2024-06-19 10:15:00 | 2219.00 | STOP_HIT | 1.00 | 3.07% |
| BUY | retest2 | 2024-06-10 14:30:00 | 2156.00 | 2024-06-19 10:15:00 | 2219.00 | STOP_HIT | 1.00 | 2.92% |
| SELL | retest2 | 2024-07-10 10:15:00 | 2315.10 | 2024-07-22 09:15:00 | 2294.70 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2024-07-11 11:15:00 | 2315.65 | 2024-07-22 09:15:00 | 2294.70 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2024-07-12 12:15:00 | 2312.80 | 2024-07-22 09:15:00 | 2294.70 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2024-07-26 13:45:00 | 2203.20 | 2024-07-26 15:15:00 | 2226.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-08-01 12:00:00 | 2184.25 | 2024-08-05 12:15:00 | 2075.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 12:00:00 | 2184.25 | 2024-08-06 09:15:00 | 2139.90 | STOP_HIT | 0.50 | 2.03% |
| BUY | retest2 | 2024-08-28 09:15:00 | 1973.10 | 2024-08-28 15:15:00 | 1954.75 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-08-28 10:15:00 | 1970.95 | 2024-08-28 15:15:00 | 1954.75 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-08-28 10:45:00 | 1972.90 | 2024-08-28 15:15:00 | 1954.75 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-08-28 14:45:00 | 1966.00 | 2024-08-28 15:15:00 | 1954.75 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-09-12 11:30:00 | 1944.60 | 2024-09-16 09:15:00 | 1928.25 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-09-12 13:45:00 | 1941.95 | 2024-09-16 09:15:00 | 1928.25 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-12 14:30:00 | 1940.00 | 2024-09-16 09:15:00 | 1928.25 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-09-12 15:00:00 | 1940.00 | 2024-09-16 09:15:00 | 1928.25 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-09-18 11:15:00 | 1900.75 | 2024-09-20 09:15:00 | 1925.95 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-09-19 09:45:00 | 1905.00 | 2024-09-20 09:15:00 | 1925.95 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-09-27 11:15:00 | 2025.60 | 2024-09-30 11:15:00 | 1991.95 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-10-08 12:00:00 | 1892.45 | 2024-10-09 09:15:00 | 1910.55 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-10-10 11:30:00 | 1891.45 | 2024-10-14 13:15:00 | 1908.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-10-11 10:30:00 | 1892.80 | 2024-10-14 13:15:00 | 1908.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-10-11 11:15:00 | 1887.00 | 2024-10-14 13:15:00 | 1908.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-10-14 10:45:00 | 1890.65 | 2024-10-14 14:15:00 | 1905.40 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-10-14 12:15:00 | 1889.00 | 2024-10-14 14:15:00 | 1905.40 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-10-17 09:15:00 | 1885.95 | 2024-10-22 14:15:00 | 1791.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:15:00 | 1885.95 | 2024-10-24 14:15:00 | 1791.45 | STOP_HIT | 0.50 | 5.01% |
| SELL | retest2 | 2024-11-13 10:30:00 | 1727.55 | 2024-11-19 11:15:00 | 1750.30 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-11-14 09:15:00 | 1726.00 | 2024-11-19 11:15:00 | 1750.30 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-11-14 11:00:00 | 1725.20 | 2024-11-19 11:15:00 | 1750.30 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-11-18 09:15:00 | 1702.05 | 2024-11-19 11:15:00 | 1750.30 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-11-28 09:30:00 | 1817.20 | 2024-11-29 12:15:00 | 1787.75 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-11-28 14:15:00 | 1812.75 | 2024-11-29 12:15:00 | 1787.75 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-12-09 11:15:00 | 1849.50 | 2024-12-12 13:15:00 | 1838.45 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-12-10 13:00:00 | 1852.20 | 2024-12-12 13:15:00 | 1838.45 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-12-11 09:30:00 | 1848.35 | 2024-12-12 13:15:00 | 1838.45 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-12-12 11:00:00 | 1845.70 | 2024-12-12 13:15:00 | 1838.45 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-12-16 11:15:00 | 1826.85 | 2024-12-16 13:15:00 | 1845.85 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-01-02 10:00:00 | 1641.40 | 2025-01-06 14:15:00 | 1559.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-02 11:15:00 | 1641.00 | 2025-01-06 14:15:00 | 1558.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-02 12:00:00 | 1640.85 | 2025-01-06 14:15:00 | 1558.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-02 12:45:00 | 1640.60 | 2025-01-06 14:15:00 | 1558.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 10:15:00 | 1631.30 | 2025-01-06 14:15:00 | 1549.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 12:15:00 | 1634.00 | 2025-01-06 14:15:00 | 1552.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-02 10:00:00 | 1641.40 | 2025-01-07 10:15:00 | 1582.75 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-01-02 11:15:00 | 1641.00 | 2025-01-07 10:15:00 | 1582.75 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2025-01-02 12:00:00 | 1640.85 | 2025-01-07 10:15:00 | 1582.75 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-01-02 12:45:00 | 1640.60 | 2025-01-07 10:15:00 | 1582.75 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2025-01-03 10:15:00 | 1631.30 | 2025-01-07 10:15:00 | 1582.75 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2025-01-03 12:15:00 | 1634.00 | 2025-01-07 10:15:00 | 1582.75 | STOP_HIT | 0.50 | 3.14% |
| BUY | retest2 | 2025-02-24 10:30:00 | 1402.00 | 2025-02-27 10:15:00 | 1380.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-02-25 09:30:00 | 1397.30 | 2025-02-27 10:15:00 | 1380.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-03-05 13:30:00 | 1317.70 | 2025-03-06 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-03-05 14:00:00 | 1317.65 | 2025-03-06 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-03-05 15:00:00 | 1316.85 | 2025-03-06 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-03-28 12:15:00 | 1285.00 | 2025-03-28 12:15:00 | 1296.80 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-06 09:45:00 | 1320.40 | 2025-05-09 09:15:00 | 1254.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:45:00 | 1320.40 | 2025-05-09 15:15:00 | 1275.90 | STOP_HIT | 0.50 | 3.37% |
| BUY | retest2 | 2025-05-15 09:15:00 | 1341.10 | 2025-05-21 09:15:00 | 1358.50 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-06-06 14:00:00 | 1518.00 | 2025-06-12 10:15:00 | 1513.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-07-01 11:00:00 | 1495.70 | 2025-07-08 14:15:00 | 1491.60 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-07-01 12:30:00 | 1494.50 | 2025-07-09 10:15:00 | 1493.70 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-07-02 12:00:00 | 1494.90 | 2025-07-09 10:15:00 | 1493.70 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-07-03 12:15:00 | 1493.00 | 2025-07-09 10:15:00 | 1493.70 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-07-08 10:15:00 | 1475.30 | 2025-07-09 10:15:00 | 1493.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1505.00 | 2025-07-21 14:15:00 | 1500.10 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-07-25 11:15:00 | 1455.00 | 2025-08-04 14:15:00 | 1428.40 | STOP_HIT | 1.00 | 1.83% |
| SELL | retest2 | 2025-07-25 12:15:00 | 1452.30 | 2025-08-04 14:15:00 | 1428.40 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1408.00 | 2025-08-12 09:15:00 | 1337.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1408.00 | 2025-08-12 14:15:00 | 1267.20 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-04 13:00:00 | 1414.00 | 2025-09-10 14:15:00 | 1438.60 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2025-09-19 10:30:00 | 1456.90 | 2025-09-26 14:15:00 | 1384.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:30:00 | 1456.90 | 2025-09-30 14:15:00 | 1366.10 | STOP_HIT | 0.50 | 6.23% |
| BUY | retest2 | 2025-10-13 12:45:00 | 1412.20 | 2025-10-24 14:15:00 | 1431.30 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2025-10-14 13:15:00 | 1412.30 | 2025-10-24 14:15:00 | 1431.30 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2025-10-28 09:15:00 | 1432.80 | 2025-10-28 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-06 09:15:00 | 1523.00 | 2025-11-13 13:15:00 | 1552.10 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2025-11-24 14:15:00 | 1448.60 | 2025-11-24 14:15:00 | 1474.30 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-12-12 12:30:00 | 1409.20 | 2025-12-12 14:15:00 | 1417.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-12-12 14:30:00 | 1411.30 | 2025-12-15 10:15:00 | 1418.90 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1407.30 | 2025-12-15 10:15:00 | 1418.90 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-12-22 11:30:00 | 1423.00 | 2025-12-30 10:15:00 | 1351.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 11:30:00 | 1423.00 | 2025-12-31 09:15:00 | 1379.50 | STOP_HIT | 0.50 | 3.06% |
| BUY | retest2 | 2026-02-13 11:45:00 | 1588.20 | 2026-02-20 09:15:00 | 1617.40 | STOP_HIT | 1.00 | 1.84% |
| BUY | retest2 | 2026-02-13 14:15:00 | 1587.10 | 2026-02-20 09:15:00 | 1617.40 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2026-02-24 11:00:00 | 1648.40 | 2026-03-02 09:15:00 | 1651.70 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2026-02-24 13:45:00 | 1647.30 | 2026-03-02 09:15:00 | 1651.70 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2026-03-17 11:30:00 | 1640.60 | 2026-03-17 14:15:00 | 1652.40 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-04-06 09:15:00 | 1491.20 | 2026-04-08 09:15:00 | 1568.90 | STOP_HIT | 1.00 | -5.21% |
| SELL | retest2 | 2026-04-23 09:30:00 | 1576.20 | 2026-05-04 09:15:00 | 1569.50 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2026-04-23 10:00:00 | 1573.90 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2026-04-23 14:00:00 | 1576.70 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2026-04-23 14:30:00 | 1576.10 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 1.42% |
| SELL | retest2 | 2026-04-24 10:30:00 | 1561.60 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1566.00 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2026-04-27 13:00:00 | 1561.80 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2026-04-29 11:00:00 | 1564.40 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1531.80 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | -1.43% |
