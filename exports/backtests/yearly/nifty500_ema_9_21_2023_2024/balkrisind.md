# Balkrishna Industries Ltd. (BALKRISIND)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 2265.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 242 |
| ALERT1 | 160 |
| ALERT2 | 157 |
| ALERT2_SKIP | 72 |
| ALERT3 | 409 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 10 |
| ENTRY2 | 171 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 175 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 200 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 137
- **Target hits / Stop hits / Partials:** 7 / 174 / 19
- **Avg / median % per leg:** 0.45% / -0.82%
- **Sum % (uncompounded):** 90.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 85 | 22 | 25.9% | 2 | 83 | 0 | -0.27% | -22.7% |
| BUY @ 2nd Alert (retest1) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.67% | -15.0% |
| BUY @ 3rd Alert (retest2) | 76 | 22 | 28.9% | 2 | 74 | 0 | -0.10% | -7.7% |
| SELL (all) | 115 | 41 | 35.7% | 5 | 91 | 19 | 0.99% | 113.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.21% | -1.2% |
| SELL @ 3rd Alert (retest2) | 114 | 41 | 36.0% | 5 | 90 | 19 | 1.01% | 114.8% |
| retest1 (combined) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.62% | -16.2% |
| retest2 (combined) | 190 | 63 | 33.2% | 7 | 164 | 19 | 0.56% | 107.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 13:15:00 | 2201.05 | 2209.78 | 2210.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 14:15:00 | 2193.70 | 2201.58 | 2205.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 10:15:00 | 2186.80 | 2172.63 | 2183.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 10:15:00 | 2186.80 | 2172.63 | 2183.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 2186.80 | 2172.63 | 2183.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 11:00:00 | 2186.80 | 2172.63 | 2183.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 11:15:00 | 2229.10 | 2183.92 | 2187.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 12:00:00 | 2229.10 | 2183.92 | 2187.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 12:15:00 | 2260.00 | 2199.14 | 2194.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 13:15:00 | 2273.85 | 2214.08 | 2201.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 15:15:00 | 2371.50 | 2376.29 | 2340.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 09:15:00 | 2344.00 | 2376.29 | 2340.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 2354.10 | 2371.85 | 2341.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:30:00 | 2349.05 | 2371.85 | 2341.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 2375.10 | 2368.67 | 2345.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:30:00 | 2359.00 | 2368.67 | 2345.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 2335.70 | 2413.57 | 2397.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 09:30:00 | 2255.00 | 2413.57 | 2397.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 11:15:00 | 2317.30 | 2379.10 | 2383.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 13:15:00 | 2293.75 | 2350.25 | 2368.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 11:15:00 | 2274.30 | 2258.85 | 2290.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-31 12:00:00 | 2274.30 | 2258.85 | 2290.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 2286.75 | 2272.01 | 2285.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 11:00:00 | 2277.70 | 2283.38 | 2285.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 09:15:00 | 2273.20 | 2281.46 | 2283.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 12:45:00 | 2280.50 | 2278.56 | 2281.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 14:30:00 | 2278.20 | 2279.20 | 2281.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 2280.95 | 2279.55 | 2281.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:15:00 | 2286.25 | 2279.55 | 2281.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-06 09:15:00 | 2293.00 | 2282.24 | 2282.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 09:15:00 | 2293.00 | 2282.24 | 2282.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 11:15:00 | 2306.20 | 2289.56 | 2285.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 11:15:00 | 2291.15 | 2296.13 | 2291.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 11:15:00 | 2291.15 | 2296.13 | 2291.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 11:15:00 | 2291.15 | 2296.13 | 2291.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 12:15:00 | 2291.15 | 2296.13 | 2291.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 12:15:00 | 2293.10 | 2295.52 | 2291.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 12:30:00 | 2290.10 | 2295.52 | 2291.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 13:15:00 | 2294.95 | 2295.41 | 2292.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 13:30:00 | 2288.00 | 2295.41 | 2292.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 2288.95 | 2295.40 | 2293.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 10:00:00 | 2288.95 | 2295.40 | 2293.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 2300.00 | 2296.32 | 2293.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:15:00 | 2288.00 | 2296.32 | 2293.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 2283.35 | 2293.73 | 2292.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:45:00 | 2280.25 | 2293.73 | 2292.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 2277.90 | 2290.56 | 2291.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 2274.00 | 2281.88 | 2286.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 2284.00 | 2273.39 | 2278.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 2284.00 | 2273.39 | 2278.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 2284.00 | 2273.39 | 2278.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:30:00 | 2289.80 | 2273.39 | 2278.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 2276.75 | 2274.06 | 2278.78 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 13:15:00 | 2297.85 | 2283.03 | 2282.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 2312.25 | 2293.77 | 2287.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 13:15:00 | 2311.65 | 2317.80 | 2308.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-14 14:00:00 | 2311.65 | 2317.80 | 2308.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 14:15:00 | 2322.00 | 2318.64 | 2309.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 09:15:00 | 2334.00 | 2318.91 | 2310.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 11:15:00 | 2409.85 | 2440.87 | 2441.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 11:15:00 | 2409.85 | 2440.87 | 2441.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-27 10:15:00 | 2378.00 | 2391.15 | 2406.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 11:15:00 | 2362.95 | 2362.78 | 2380.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-28 12:00:00 | 2362.95 | 2362.78 | 2380.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 2377.65 | 2364.86 | 2374.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 10:00:00 | 2377.65 | 2364.86 | 2374.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 2373.95 | 2366.68 | 2374.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 09:45:00 | 2349.20 | 2366.19 | 2371.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 10:15:00 | 2354.00 | 2357.20 | 2363.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 10:45:00 | 2354.90 | 2356.66 | 2362.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 09:15:00 | 2356.55 | 2355.17 | 2358.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-05 10:15:00 | 2393.60 | 2365.71 | 2363.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 2393.60 | 2365.71 | 2363.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 11:15:00 | 2414.30 | 2375.43 | 2367.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 2425.35 | 2433.07 | 2414.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-07 10:45:00 | 2427.00 | 2433.07 | 2414.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 2414.50 | 2428.04 | 2415.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:30:00 | 2410.00 | 2428.04 | 2415.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 13:15:00 | 2415.85 | 2425.60 | 2415.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 14:15:00 | 2417.00 | 2425.60 | 2415.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 2397.20 | 2419.92 | 2413.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 15:00:00 | 2397.20 | 2419.92 | 2413.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 2391.30 | 2414.20 | 2411.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 2325.50 | 2414.20 | 2411.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 09:15:00 | 2319.55 | 2395.27 | 2403.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 11:15:00 | 2286.00 | 2359.60 | 2385.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 13:15:00 | 2326.00 | 2317.11 | 2340.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-11 14:00:00 | 2326.00 | 2317.11 | 2340.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 2340.00 | 2321.69 | 2340.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:30:00 | 2340.00 | 2321.69 | 2340.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 2341.00 | 2325.55 | 2340.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 2341.70 | 2325.55 | 2340.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 2366.90 | 2333.82 | 2342.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:00:00 | 2366.90 | 2333.82 | 2342.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 2374.05 | 2341.87 | 2345.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 11:00:00 | 2374.05 | 2341.87 | 2345.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 11:15:00 | 2375.00 | 2348.49 | 2348.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 12:15:00 | 2393.00 | 2357.39 | 2352.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 2387.00 | 2402.72 | 2384.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 13:15:00 | 2387.00 | 2402.72 | 2384.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 2387.00 | 2402.72 | 2384.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 2387.00 | 2402.72 | 2384.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 2388.50 | 2399.87 | 2384.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:30:00 | 2376.65 | 2399.87 | 2384.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 2436.70 | 2424.19 | 2414.57 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 14:15:00 | 2401.90 | 2409.48 | 2409.97 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 2429.00 | 2413.56 | 2411.46 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 15:15:00 | 2408.00 | 2412.52 | 2413.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 10:15:00 | 2396.90 | 2408.45 | 2411.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 13:15:00 | 2404.25 | 2403.63 | 2407.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-21 14:00:00 | 2404.25 | 2403.63 | 2407.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 2420.00 | 2406.91 | 2408.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 14:45:00 | 2412.40 | 2406.91 | 2408.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 2405.00 | 2406.53 | 2408.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:15:00 | 2435.95 | 2406.53 | 2408.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2023-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 09:15:00 | 2426.35 | 2410.49 | 2410.22 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 11:15:00 | 2404.65 | 2409.64 | 2409.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 13:15:00 | 2397.15 | 2406.66 | 2408.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 12:15:00 | 2387.00 | 2386.68 | 2395.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-25 12:30:00 | 2385.00 | 2386.68 | 2395.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 2431.10 | 2394.54 | 2396.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 10:00:00 | 2431.10 | 2394.54 | 2396.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 10:15:00 | 2430.25 | 2401.68 | 2399.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 11:15:00 | 2465.80 | 2414.51 | 2405.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 11:15:00 | 2470.55 | 2471.58 | 2446.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-27 12:00:00 | 2470.55 | 2471.58 | 2446.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 2482.50 | 2477.22 | 2455.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 15:00:00 | 2482.50 | 2477.22 | 2455.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 2459.60 | 2473.75 | 2457.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-31 10:00:00 | 2483.85 | 2466.96 | 2460.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 11:15:00 | 2461.25 | 2498.92 | 2500.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 2461.25 | 2498.92 | 2500.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 2441.15 | 2480.68 | 2491.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 09:15:00 | 2488.00 | 2478.52 | 2487.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 09:15:00 | 2488.00 | 2478.52 | 2487.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 2488.00 | 2478.52 | 2487.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 10:00:00 | 2488.00 | 2478.52 | 2487.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 2476.20 | 2478.06 | 2486.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 12:15:00 | 2456.10 | 2474.35 | 2483.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-04 11:15:00 | 2510.00 | 2486.52 | 2484.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 11:15:00 | 2510.00 | 2486.52 | 2484.90 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 09:15:00 | 2461.05 | 2482.58 | 2484.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 10:15:00 | 2393.10 | 2464.68 | 2475.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 13:15:00 | 2361.25 | 2356.69 | 2381.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-09 14:00:00 | 2361.25 | 2356.69 | 2381.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 12:15:00 | 2367.75 | 2358.73 | 2371.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 13:30:00 | 2362.35 | 2359.58 | 2370.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 09:15:00 | 2349.40 | 2361.68 | 2369.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 15:00:00 | 2350.00 | 2360.40 | 2365.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 2348.75 | 2354.23 | 2356.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 09:15:00 | 2378.35 | 2359.05 | 2358.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 09:15:00 | 2378.35 | 2359.05 | 2358.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 12:15:00 | 2384.40 | 2368.23 | 2363.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-16 14:15:00 | 2369.00 | 2369.30 | 2364.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-16 14:45:00 | 2377.30 | 2369.30 | 2364.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 2362.00 | 2367.84 | 2364.49 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 11:15:00 | 2349.65 | 2362.33 | 2362.68 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 14:15:00 | 2378.30 | 2362.84 | 2360.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 11:15:00 | 2385.10 | 2373.70 | 2367.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 15:15:00 | 2380.05 | 2381.61 | 2373.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 09:45:00 | 2378.00 | 2381.11 | 2374.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 2386.15 | 2386.25 | 2380.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 09:15:00 | 2391.00 | 2386.25 | 2380.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-23 14:15:00 | 2376.55 | 2389.80 | 2385.33 | SL hit (close<static) qty=1.00 sl=2377.70 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 14:15:00 | 2377.90 | 2382.62 | 2383.02 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 15:15:00 | 2387.80 | 2383.65 | 2383.45 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 2380.55 | 2383.03 | 2383.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 2327.80 | 2355.47 | 2363.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 09:15:00 | 2354.85 | 2354.76 | 2361.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 09:30:00 | 2350.00 | 2354.76 | 2361.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 2362.85 | 2356.38 | 2361.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-01 12:15:00 | 2354.40 | 2357.58 | 2361.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-01 13:00:00 | 2353.15 | 2356.70 | 2361.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 11:15:00 | 2375.00 | 2364.50 | 2363.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 11:15:00 | 2375.00 | 2364.50 | 2363.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 12:15:00 | 2380.00 | 2367.60 | 2365.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 09:15:00 | 2408.70 | 2411.63 | 2402.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 09:15:00 | 2408.70 | 2411.63 | 2402.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 2408.70 | 2411.63 | 2402.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 11:00:00 | 2428.40 | 2410.37 | 2405.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 13:15:00 | 2431.75 | 2417.39 | 2410.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 11:15:00 | 2383.20 | 2408.82 | 2409.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 2383.20 | 2408.82 | 2409.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 2362.10 | 2399.47 | 2405.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 09:15:00 | 2399.95 | 2396.89 | 2402.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 09:15:00 | 2399.95 | 2396.89 | 2402.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 2399.95 | 2396.89 | 2402.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 09:30:00 | 2372.65 | 2396.89 | 2402.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 2410.00 | 2399.52 | 2402.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 11:00:00 | 2410.00 | 2399.52 | 2402.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 2417.00 | 2403.01 | 2404.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 11:45:00 | 2414.40 | 2403.01 | 2404.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 2403.00 | 2403.01 | 2403.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:30:00 | 2403.10 | 2403.01 | 2403.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 2402.00 | 2402.81 | 2403.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 14:30:00 | 2401.05 | 2402.81 | 2403.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 2403.20 | 2402.89 | 2403.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 2436.10 | 2402.89 | 2403.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 2442.00 | 2410.71 | 2407.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 10:15:00 | 2462.40 | 2421.05 | 2412.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 10:15:00 | 2515.30 | 2517.72 | 2490.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 10:30:00 | 2519.70 | 2517.72 | 2490.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 2526.45 | 2525.79 | 2507.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 12:00:00 | 2552.50 | 2531.77 | 2513.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 15:00:00 | 2550.10 | 2538.69 | 2521.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 11:15:00 | 2549.20 | 2528.33 | 2525.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 11:15:00 | 2541.00 | 2547.41 | 2547.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 11:15:00 | 2541.00 | 2547.41 | 2547.57 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 2552.45 | 2548.38 | 2547.94 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 10:15:00 | 2543.45 | 2547.51 | 2547.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 11:15:00 | 2526.70 | 2543.35 | 2545.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 13:15:00 | 2546.65 | 2541.02 | 2544.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 13:15:00 | 2546.65 | 2541.02 | 2544.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 2546.65 | 2541.02 | 2544.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:00:00 | 2546.65 | 2541.02 | 2544.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 2541.25 | 2541.07 | 2543.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:15:00 | 2522.00 | 2541.07 | 2543.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 2522.00 | 2537.25 | 2541.88 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 12:15:00 | 2548.45 | 2544.73 | 2544.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 14:15:00 | 2559.60 | 2548.40 | 2546.23 | Break + close above crossover candle high |

### Cycle 33 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 2501.10 | 2540.39 | 2543.05 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 10:15:00 | 2548.00 | 2530.71 | 2529.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 14:15:00 | 2581.15 | 2549.75 | 2539.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 2540.35 | 2557.97 | 2552.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 2540.35 | 2557.97 | 2552.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 2540.35 | 2557.97 | 2552.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 13:00:00 | 2548.95 | 2551.53 | 2550.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 13:15:00 | 2537.75 | 2548.78 | 2549.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 2537.75 | 2548.78 | 2549.22 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 2573.00 | 2551.77 | 2550.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 2594.30 | 2569.04 | 2561.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 11:15:00 | 2584.20 | 2584.38 | 2576.25 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 14:45:00 | 2590.70 | 2587.68 | 2579.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 2609.60 | 2593.24 | 2583.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-13 15:15:00 | 2584.00 | 2592.02 | 2587.55 | SL hit (close<ema400) qty=1.00 sl=2587.55 alert=retest1 |

### Cycle 37 — SELL (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 10:15:00 | 2572.70 | 2583.28 | 2584.05 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 14:15:00 | 2593.10 | 2585.66 | 2584.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 09:15:00 | 2627.40 | 2593.45 | 2588.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 15:15:00 | 2612.00 | 2612.59 | 2602.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:15:00 | 2626.30 | 2612.59 | 2602.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:45:00 | 2624.05 | 2614.28 | 2604.10 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 10:30:00 | 2620.90 | 2614.31 | 2605.04 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 11:15:00 | 2621.80 | 2614.31 | 2605.04 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 2608.70 | 2613.15 | 2606.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:30:00 | 2594.30 | 2613.15 | 2606.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 13:15:00 | 2606.80 | 2611.88 | 2606.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 13:30:00 | 2603.60 | 2611.88 | 2606.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 2596.00 | 2608.71 | 2605.26 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-18 14:15:00 | 2596.00 | 2608.71 | 2605.26 | SL hit (close<ema400) qty=1.00 sl=2605.26 alert=retest1 |

### Cycle 39 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 2575.00 | 2601.97 | 2602.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 2560.75 | 2593.72 | 2598.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 2578.15 | 2577.61 | 2586.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-20 10:00:00 | 2578.15 | 2577.61 | 2586.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 2545.15 | 2552.35 | 2567.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:30:00 | 2544.15 | 2552.35 | 2567.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 12:15:00 | 2562.05 | 2554.52 | 2564.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 13:00:00 | 2562.05 | 2554.52 | 2564.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 13:15:00 | 2566.35 | 2556.88 | 2564.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 14:00:00 | 2566.35 | 2556.88 | 2564.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 14:15:00 | 2562.80 | 2558.07 | 2564.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 14:45:00 | 2573.40 | 2558.07 | 2564.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 15:15:00 | 2549.80 | 2556.41 | 2563.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 09:15:00 | 2602.55 | 2556.41 | 2563.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-25 09:15:00 | 2615.00 | 2568.13 | 2567.90 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 10:15:00 | 2541.50 | 2569.36 | 2571.99 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 2596.70 | 2572.19 | 2569.69 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 14:15:00 | 2554.30 | 2577.56 | 2579.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 11:15:00 | 2534.55 | 2558.00 | 2566.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 2571.15 | 2549.87 | 2552.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 2571.15 | 2549.87 | 2552.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 2571.15 | 2549.87 | 2552.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 10:00:00 | 2571.15 | 2549.87 | 2552.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 2571.10 | 2554.12 | 2554.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 11:15:00 | 2573.30 | 2554.12 | 2554.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2023-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 11:15:00 | 2570.00 | 2557.29 | 2556.00 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 15:15:00 | 2550.00 | 2558.21 | 2558.41 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 2608.10 | 2568.19 | 2562.93 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 10:15:00 | 2563.35 | 2576.16 | 2577.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 11:15:00 | 2542.00 | 2562.46 | 2569.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 13:15:00 | 2571.20 | 2563.12 | 2568.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 13:15:00 | 2571.20 | 2563.12 | 2568.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 13:15:00 | 2571.20 | 2563.12 | 2568.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 13:30:00 | 2578.10 | 2563.12 | 2568.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 14:15:00 | 2574.05 | 2565.31 | 2569.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 15:00:00 | 2574.05 | 2565.31 | 2569.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 2561.00 | 2564.45 | 2568.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:15:00 | 2597.55 | 2564.45 | 2568.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 2604.50 | 2572.46 | 2571.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 10:15:00 | 2637.95 | 2585.56 | 2577.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 15:15:00 | 2640.20 | 2641.22 | 2623.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 09:15:00 | 2671.30 | 2641.22 | 2623.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 10:15:00 | 2660.75 | 2641.80 | 2625.12 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-20 09:15:00 | 2566.45 | 2625.66 | 2625.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 49 — SELL (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 09:15:00 | 2566.45 | 2625.66 | 2625.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 10:15:00 | 2545.60 | 2609.65 | 2618.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 2495.65 | 2489.64 | 2524.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-22 09:30:00 | 2495.90 | 2489.64 | 2524.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 2553.25 | 2504.72 | 2514.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:00:00 | 2553.25 | 2504.72 | 2514.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 2543.75 | 2512.52 | 2517.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:30:00 | 2550.85 | 2512.52 | 2517.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2023-11-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 12:15:00 | 2542.65 | 2523.00 | 2521.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 13:15:00 | 2548.05 | 2528.01 | 2523.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 12:15:00 | 2603.55 | 2617.40 | 2596.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 12:45:00 | 2603.55 | 2617.40 | 2596.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 2599.65 | 2613.85 | 2597.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 13:30:00 | 2595.60 | 2613.85 | 2597.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 2583.00 | 2607.68 | 2595.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 15:00:00 | 2583.00 | 2607.68 | 2595.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 2591.75 | 2604.50 | 2595.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 09:30:00 | 2592.75 | 2601.24 | 2594.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 11:00:00 | 2597.30 | 2600.45 | 2594.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 13:15:00 | 2574.20 | 2591.79 | 2592.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 13:15:00 | 2574.20 | 2591.79 | 2592.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-01 09:15:00 | 2565.05 | 2581.85 | 2587.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 11:15:00 | 2564.95 | 2558.90 | 2568.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-04 12:00:00 | 2564.95 | 2558.90 | 2568.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 13:15:00 | 2559.85 | 2559.49 | 2567.40 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 10:15:00 | 2597.45 | 2570.57 | 2570.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 10:15:00 | 2622.60 | 2595.01 | 2583.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 12:15:00 | 2598.05 | 2598.50 | 2587.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-06 13:00:00 | 2598.05 | 2598.50 | 2587.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 2582.00 | 2595.20 | 2587.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 15:00:00 | 2582.00 | 2595.20 | 2587.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 15:15:00 | 2580.00 | 2592.16 | 2587.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:15:00 | 2589.55 | 2592.16 | 2587.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 2647.00 | 2603.13 | 2592.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 10:15:00 | 2672.00 | 2603.13 | 2592.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 11:00:00 | 2662.45 | 2614.99 | 2598.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 12:15:00 | 2578.15 | 2613.94 | 2611.45 | SL hit (close<static) qty=1.00 sl=2578.85 alert=retest2 |

### Cycle 53 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 2569.10 | 2604.97 | 2607.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 11:15:00 | 2540.10 | 2576.53 | 2586.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 14:15:00 | 2588.30 | 2568.24 | 2578.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 14:15:00 | 2588.30 | 2568.24 | 2578.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 2588.30 | 2568.24 | 2578.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 15:00:00 | 2588.30 | 2568.24 | 2578.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 2594.00 | 2573.40 | 2580.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 2587.35 | 2573.40 | 2580.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 2582.55 | 2575.23 | 2580.52 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 14:15:00 | 2605.60 | 2586.94 | 2584.69 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 14:15:00 | 2569.95 | 2586.47 | 2587.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 09:15:00 | 2559.95 | 2578.21 | 2583.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 09:15:00 | 2533.75 | 2522.20 | 2537.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-20 10:15:00 | 2535.95 | 2522.20 | 2537.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 2539.55 | 2525.67 | 2537.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:00:00 | 2539.55 | 2525.67 | 2537.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 2514.80 | 2523.50 | 2535.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 12:15:00 | 2510.50 | 2523.50 | 2535.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 09:15:00 | 2512.00 | 2494.01 | 2492.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 2512.00 | 2494.01 | 2492.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 10:15:00 | 2519.75 | 2499.15 | 2494.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 2528.85 | 2530.68 | 2520.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 10:00:00 | 2528.85 | 2530.68 | 2520.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 2518.25 | 2528.19 | 2520.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 11:00:00 | 2518.25 | 2528.19 | 2520.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 2510.00 | 2524.56 | 2519.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 11:30:00 | 2507.50 | 2524.56 | 2519.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 2513.65 | 2518.41 | 2517.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 2534.00 | 2518.41 | 2517.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 10:00:00 | 2531.05 | 2520.94 | 2519.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 14:15:00 | 2521.40 | 2552.62 | 2553.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 14:15:00 | 2521.40 | 2552.62 | 2553.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 10:15:00 | 2497.45 | 2534.42 | 2544.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 15:15:00 | 2503.00 | 2502.82 | 2515.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-05 09:15:00 | 2511.30 | 2502.82 | 2515.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 2532.00 | 2508.66 | 2516.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:00:00 | 2532.00 | 2508.66 | 2516.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 2550.85 | 2517.10 | 2519.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 11:00:00 | 2550.85 | 2517.10 | 2519.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 11:15:00 | 2557.90 | 2525.26 | 2523.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 12:15:00 | 2584.00 | 2537.01 | 2528.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 11:15:00 | 2557.50 | 2559.84 | 2546.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 12:00:00 | 2557.50 | 2559.84 | 2546.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 2547.30 | 2556.33 | 2546.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:45:00 | 2546.20 | 2556.33 | 2546.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 2547.25 | 2554.51 | 2547.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 15:00:00 | 2547.25 | 2554.51 | 2547.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 2538.00 | 2551.21 | 2546.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 2592.05 | 2551.21 | 2546.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 11:15:00 | 2607.90 | 2627.03 | 2629.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 11:15:00 | 2607.90 | 2627.03 | 2629.03 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 15:15:00 | 2640.00 | 2631.23 | 2630.40 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 2619.10 | 2630.27 | 2630.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 14:15:00 | 2607.00 | 2623.20 | 2627.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 2581.75 | 2570.79 | 2590.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 11:15:00 | 2581.75 | 2570.79 | 2590.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 2581.75 | 2570.79 | 2590.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:00:00 | 2581.75 | 2570.79 | 2590.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 2602.00 | 2573.04 | 2583.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:00:00 | 2602.00 | 2573.04 | 2583.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 2611.65 | 2580.76 | 2585.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:45:00 | 2608.00 | 2580.76 | 2585.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 12:15:00 | 2613.55 | 2590.90 | 2589.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 2645.40 | 2605.88 | 2597.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 11:15:00 | 2683.45 | 2706.82 | 2670.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 12:00:00 | 2683.45 | 2706.82 | 2670.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 2669.50 | 2699.36 | 2670.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:00:00 | 2669.50 | 2699.36 | 2670.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 2639.00 | 2687.29 | 2667.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 14:00:00 | 2639.00 | 2687.29 | 2667.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 2625.00 | 2674.83 | 2663.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 15:00:00 | 2625.00 | 2674.83 | 2663.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 2673.85 | 2665.81 | 2661.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:30:00 | 2669.90 | 2665.81 | 2661.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 2659.55 | 2664.56 | 2661.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:30:00 | 2660.25 | 2664.56 | 2661.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 2683.30 | 2668.31 | 2663.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:45:00 | 2670.00 | 2668.31 | 2663.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 2635.00 | 2682.36 | 2674.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:00:00 | 2635.00 | 2682.36 | 2674.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 11:15:00 | 2540.00 | 2653.89 | 2662.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 2507.05 | 2624.52 | 2648.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 15:15:00 | 2462.00 | 2456.64 | 2485.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-01 09:15:00 | 2475.60 | 2456.64 | 2485.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 2488.95 | 2463.10 | 2485.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 10:00:00 | 2488.95 | 2463.10 | 2485.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 2462.00 | 2462.88 | 2483.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 11:15:00 | 2456.55 | 2462.88 | 2483.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 11:45:00 | 2440.00 | 2458.24 | 2479.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 11:45:00 | 2454.95 | 2453.97 | 2465.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 09:45:00 | 2456.45 | 2455.16 | 2461.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 2450.00 | 2454.13 | 2460.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 13:00:00 | 2436.10 | 2449.58 | 2457.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 09:15:00 | 2333.72 | 2360.95 | 2385.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 09:15:00 | 2332.20 | 2360.95 | 2385.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 09:15:00 | 2333.63 | 2360.95 | 2385.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 13:15:00 | 2318.00 | 2340.28 | 2367.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 13:15:00 | 2314.29 | 2340.28 | 2367.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-09 15:15:00 | 2305.10 | 2303.50 | 2327.35 | SL hit (close>ema200) qty=0.50 sl=2303.50 alert=retest2 |

### Cycle 64 — BUY (started 2024-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 09:15:00 | 2344.00 | 2330.36 | 2330.04 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 11:15:00 | 2303.10 | 2325.61 | 2327.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 12:15:00 | 2297.65 | 2320.02 | 2325.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 2320.00 | 2318.65 | 2322.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-14 10:00:00 | 2320.00 | 2318.65 | 2322.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 2309.50 | 2316.82 | 2321.54 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 2351.65 | 2326.54 | 2324.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 15:15:00 | 2358.00 | 2332.83 | 2327.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 11:15:00 | 2336.35 | 2338.66 | 2332.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-15 12:00:00 | 2336.35 | 2338.66 | 2332.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 12:15:00 | 2338.25 | 2338.57 | 2332.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 13:15:00 | 2334.00 | 2338.57 | 2332.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 13:15:00 | 2335.20 | 2337.90 | 2332.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 14:15:00 | 2331.55 | 2337.90 | 2332.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 2337.50 | 2337.82 | 2333.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 15:15:00 | 2345.75 | 2337.82 | 2333.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 14:45:00 | 2341.65 | 2341.67 | 2338.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 09:15:00 | 2318.85 | 2335.88 | 2336.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 09:15:00 | 2318.85 | 2335.88 | 2336.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 09:15:00 | 2293.00 | 2319.45 | 2326.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 10:15:00 | 2300.10 | 2298.16 | 2308.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-21 10:45:00 | 2295.40 | 2298.16 | 2308.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 2283.55 | 2276.33 | 2285.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 2308.00 | 2276.33 | 2285.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 2321.10 | 2285.29 | 2288.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 10:00:00 | 2321.10 | 2285.29 | 2288.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 2343.20 | 2296.87 | 2293.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 2347.60 | 2313.60 | 2302.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 2276.10 | 2315.65 | 2307.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 2276.10 | 2315.65 | 2307.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 2276.10 | 2315.65 | 2307.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:45:00 | 2283.70 | 2315.65 | 2307.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 2268.15 | 2306.15 | 2304.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:00:00 | 2268.15 | 2306.15 | 2304.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 11:15:00 | 2285.05 | 2301.93 | 2302.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 2249.55 | 2277.44 | 2287.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 2235.00 | 2224.27 | 2245.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 2235.00 | 2224.27 | 2245.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 2211.80 | 2222.69 | 2241.25 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 09:15:00 | 2263.10 | 2245.96 | 2245.16 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 14:15:00 | 2239.55 | 2247.91 | 2248.25 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 09:15:00 | 2304.00 | 2257.70 | 2252.56 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 09:15:00 | 2240.05 | 2265.86 | 2267.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 10:15:00 | 2222.25 | 2257.14 | 2263.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 15:15:00 | 2249.35 | 2249.21 | 2256.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-12 09:15:00 | 2260.60 | 2249.21 | 2256.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 2266.45 | 2252.66 | 2257.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:00:00 | 2266.45 | 2252.66 | 2257.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 10:15:00 | 2328.95 | 2267.91 | 2263.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 13:15:00 | 2333.00 | 2298.53 | 2280.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 10:15:00 | 2316.10 | 2320.13 | 2298.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-13 10:30:00 | 2312.30 | 2320.13 | 2298.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 13:15:00 | 2310.45 | 2315.12 | 2301.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 14:15:00 | 2299.60 | 2315.12 | 2301.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 2285.85 | 2309.26 | 2299.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 14:45:00 | 2288.65 | 2309.26 | 2299.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 15:15:00 | 2294.90 | 2306.39 | 2299.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:15:00 | 2257.35 | 2306.39 | 2299.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-03-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 09:15:00 | 2233.50 | 2291.81 | 2293.29 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 11:15:00 | 2287.25 | 2270.38 | 2268.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 12:15:00 | 2291.40 | 2274.59 | 2270.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 15:15:00 | 2278.05 | 2278.90 | 2273.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 09:15:00 | 2293.90 | 2278.90 | 2273.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 2268.75 | 2276.87 | 2273.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 2268.75 | 2276.87 | 2273.38 | SL hit (close<ema400) qty=1.00 sl=2273.38 alert=retest1 |

### Cycle 77 — SELL (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 13:15:00 | 2267.80 | 2279.88 | 2280.36 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 09:15:00 | 2292.70 | 2278.54 | 2278.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 11:15:00 | 2301.75 | 2285.93 | 2281.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 14:15:00 | 2290.00 | 2291.93 | 2286.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 14:15:00 | 2290.00 | 2291.93 | 2286.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 2290.00 | 2291.93 | 2286.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 14:45:00 | 2287.15 | 2291.93 | 2286.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 2303.50 | 2295.53 | 2288.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 13:45:00 | 2311.35 | 2303.16 | 2294.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 10:00:00 | 2317.85 | 2303.47 | 2301.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 12:15:00 | 2402.45 | 2414.63 | 2415.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 2402.45 | 2414.63 | 2415.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 2375.25 | 2399.53 | 2407.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 10:15:00 | 2389.30 | 2388.80 | 2396.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 11:00:00 | 2389.30 | 2388.80 | 2396.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 2381.80 | 2370.97 | 2381.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:00:00 | 2381.80 | 2370.97 | 2381.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 2369.45 | 2370.67 | 2380.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 2360.95 | 2371.09 | 2379.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 13:15:00 | 2370.35 | 2358.56 | 2357.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 2370.35 | 2358.56 | 2357.37 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 12:15:00 | 2340.35 | 2355.86 | 2357.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 09:15:00 | 2316.00 | 2342.82 | 2350.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 11:15:00 | 2350.80 | 2334.17 | 2340.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 11:15:00 | 2350.80 | 2334.17 | 2340.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 2350.80 | 2334.17 | 2340.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:00:00 | 2350.80 | 2334.17 | 2340.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 12:15:00 | 2395.00 | 2346.34 | 2345.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 09:15:00 | 2451.00 | 2397.04 | 2381.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 2425.85 | 2428.49 | 2406.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 15:00:00 | 2425.85 | 2428.49 | 2406.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 2436.00 | 2446.27 | 2431.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:00:00 | 2436.00 | 2446.27 | 2431.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 2435.05 | 2444.03 | 2431.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:30:00 | 2435.00 | 2444.03 | 2431.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 2436.55 | 2442.53 | 2432.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:45:00 | 2427.65 | 2442.53 | 2432.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 2437.55 | 2441.53 | 2432.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:45:00 | 2426.85 | 2441.53 | 2432.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 2458.25 | 2447.86 | 2438.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:30:00 | 2439.20 | 2447.86 | 2438.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 12:15:00 | 2457.80 | 2451.31 | 2442.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 12:30:00 | 2438.35 | 2451.31 | 2442.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 2448.45 | 2456.44 | 2447.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:00:00 | 2448.45 | 2456.44 | 2447.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 2450.25 | 2455.21 | 2448.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:30:00 | 2434.10 | 2455.21 | 2448.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 2430.05 | 2450.17 | 2446.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:30:00 | 2407.20 | 2450.17 | 2446.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 2440.90 | 2448.32 | 2446.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:45:00 | 2427.00 | 2448.32 | 2446.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 14:15:00 | 2438.00 | 2445.13 | 2444.91 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 15:15:00 | 2436.00 | 2443.31 | 2444.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 10:15:00 | 2433.05 | 2440.65 | 2442.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 11:15:00 | 2445.50 | 2441.62 | 2442.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 11:15:00 | 2445.50 | 2441.62 | 2442.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 2445.50 | 2441.62 | 2442.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 2445.50 | 2441.62 | 2442.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 2450.60 | 2443.42 | 2443.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:15:00 | 2457.55 | 2443.42 | 2443.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 13:15:00 | 2452.10 | 2445.15 | 2444.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 14:15:00 | 2464.05 | 2448.93 | 2446.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 2447.60 | 2454.25 | 2450.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 12:15:00 | 2447.60 | 2454.25 | 2450.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 2447.60 | 2454.25 | 2450.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 12:45:00 | 2441.60 | 2454.25 | 2450.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 2441.15 | 2451.63 | 2449.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:00:00 | 2441.15 | 2451.63 | 2449.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 14:15:00 | 2435.80 | 2448.47 | 2448.52 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 15:15:00 | 2451.00 | 2448.97 | 2448.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 11:15:00 | 2454.95 | 2450.87 | 2449.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-10 12:15:00 | 2449.55 | 2450.61 | 2449.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 12:15:00 | 2449.55 | 2450.61 | 2449.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 2449.55 | 2450.61 | 2449.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:00:00 | 2449.55 | 2450.61 | 2449.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 2455.15 | 2451.52 | 2450.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 14:15:00 | 2462.40 | 2451.52 | 2450.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 2439.80 | 2456.12 | 2453.23 | SL hit (close<static) qty=1.00 sl=2446.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 3058.00 | 3083.67 | 3084.42 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 3108.90 | 3086.56 | 3085.47 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 14:15:00 | 3053.40 | 3079.93 | 3082.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 15:15:00 | 3022.00 | 3068.34 | 3077.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 2998.10 | 2943.36 | 2986.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 2998.10 | 2943.36 | 2986.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 2998.10 | 2943.36 | 2986.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 2998.10 | 2943.36 | 2986.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 3030.35 | 2960.76 | 2990.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:30:00 | 3015.45 | 2960.76 | 2990.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 3093.55 | 3014.34 | 3009.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 3129.00 | 3037.27 | 3020.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 3211.65 | 3236.06 | 3205.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 3211.65 | 3236.06 | 3205.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 3231.20 | 3235.09 | 3207.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 3236.90 | 3235.09 | 3207.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 14:15:00 | 3205.95 | 3221.80 | 3212.45 | SL hit (close<static) qty=1.00 sl=3206.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 14:15:00 | 3187.50 | 3206.41 | 3208.40 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 12:15:00 | 3220.00 | 3208.87 | 3208.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 13:15:00 | 3248.40 | 3216.78 | 3212.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 3300.55 | 3311.54 | 3277.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 10:00:00 | 3300.55 | 3311.54 | 3277.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 3290.70 | 3306.85 | 3288.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 3290.70 | 3306.85 | 3288.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 3286.00 | 3302.68 | 3287.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 3284.10 | 3302.68 | 3287.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 3300.00 | 3302.14 | 3289.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 3309.00 | 3302.14 | 3289.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 13:15:00 | 3310.95 | 3298.86 | 3290.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 14:15:00 | 3308.85 | 3298.62 | 3291.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 3267.60 | 3286.54 | 3288.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 3267.60 | 3286.54 | 3288.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 3240.90 | 3277.41 | 3284.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 13:15:00 | 3227.90 | 3226.13 | 3247.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 13:45:00 | 3234.15 | 3226.13 | 3247.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 3188.45 | 3185.32 | 3200.84 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 12:15:00 | 3226.05 | 3203.93 | 3200.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 15:15:00 | 3230.00 | 3215.72 | 3207.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 09:15:00 | 3204.60 | 3213.50 | 3207.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 3204.60 | 3213.50 | 3207.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 3204.60 | 3213.50 | 3207.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 3201.35 | 3213.50 | 3207.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 3199.45 | 3210.69 | 3206.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:00:00 | 3199.45 | 3210.69 | 3206.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 3194.20 | 3207.39 | 3205.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:00:00 | 3194.20 | 3207.39 | 3205.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 12:15:00 | 3189.20 | 3203.75 | 3204.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 09:15:00 | 3163.75 | 3192.01 | 3198.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 3138.95 | 3130.23 | 3148.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-04 10:00:00 | 3138.95 | 3130.23 | 3148.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 3154.75 | 3137.80 | 3149.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 3154.75 | 3137.80 | 3149.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 3154.25 | 3141.09 | 3149.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:30:00 | 3144.00 | 3142.11 | 3149.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 10:15:00 | 3190.80 | 3156.88 | 3154.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 3190.80 | 3156.88 | 3154.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 10:15:00 | 3204.05 | 3180.57 | 3169.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 11:15:00 | 3120.70 | 3168.59 | 3165.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 11:15:00 | 3120.70 | 3168.59 | 3165.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 3120.70 | 3168.59 | 3165.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 3120.70 | 3168.59 | 3165.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 3150.95 | 3165.06 | 3163.74 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 3153.65 | 3162.78 | 3162.82 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 14:15:00 | 3166.00 | 3163.42 | 3163.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 15:15:00 | 3174.70 | 3165.68 | 3164.16 | Break + close above crossover candle high |

### Cycle 99 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 10:15:00 | 3148.80 | 3162.67 | 3163.08 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 3176.05 | 3165.34 | 3164.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 12:15:00 | 3190.40 | 3170.35 | 3166.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 13:15:00 | 3162.05 | 3168.69 | 3166.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 3162.05 | 3168.69 | 3166.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 3162.05 | 3168.69 | 3166.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 3162.05 | 3168.69 | 3166.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 3185.55 | 3172.07 | 3167.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:30:00 | 3194.65 | 3172.07 | 3167.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 3159.40 | 3172.77 | 3169.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 3159.40 | 3172.77 | 3169.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 3152.85 | 3168.79 | 3167.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 3136.95 | 3168.79 | 3167.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 3182.55 | 3170.33 | 3168.53 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 09:15:00 | 3151.40 | 3167.31 | 3167.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 14:15:00 | 3149.25 | 3158.38 | 3162.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 3156.85 | 3156.14 | 3160.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 3156.85 | 3156.14 | 3160.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 3156.85 | 3156.14 | 3160.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:15:00 | 3138.95 | 3154.76 | 3159.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:30:00 | 3142.40 | 3135.50 | 3145.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 10:30:00 | 3139.75 | 3136.29 | 3145.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 12:00:00 | 3141.20 | 3137.27 | 3144.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 3152.35 | 3140.29 | 3145.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:00:00 | 3152.35 | 3140.29 | 3145.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 3155.75 | 3143.38 | 3146.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:30:00 | 3165.10 | 3143.38 | 3146.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 3150.05 | 3146.71 | 3147.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 3150.05 | 3146.71 | 3147.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 3128.05 | 3142.98 | 3145.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:30:00 | 3147.85 | 3142.98 | 3145.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 3126.45 | 3124.98 | 3132.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:15:00 | 3136.00 | 3124.98 | 3132.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 3136.90 | 3127.36 | 3133.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:30:00 | 3135.35 | 3127.36 | 3133.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 3139.05 | 3129.70 | 3133.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 3139.05 | 3129.70 | 3133.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-18 14:15:00 | 3173.80 | 3138.52 | 3137.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 14:15:00 | 3173.80 | 3138.52 | 3137.45 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 3106.85 | 3133.72 | 3135.84 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 15:15:00 | 3145.00 | 3137.36 | 3136.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 09:15:00 | 3178.10 | 3145.51 | 3140.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 12:15:00 | 3144.10 | 3153.37 | 3146.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 12:15:00 | 3144.10 | 3153.37 | 3146.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 3144.10 | 3153.37 | 3146.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:30:00 | 3144.90 | 3153.37 | 3146.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 3147.65 | 3152.23 | 3146.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:15:00 | 3157.90 | 3152.23 | 3146.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 3169.90 | 3155.76 | 3148.37 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 13:15:00 | 3120.65 | 3142.94 | 3145.44 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 3175.60 | 3149.47 | 3148.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 15:15:00 | 3183.00 | 3156.18 | 3151.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 3151.35 | 3155.21 | 3151.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 3151.35 | 3155.21 | 3151.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 3151.35 | 3155.21 | 3151.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 3155.25 | 3155.21 | 3151.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 3154.40 | 3155.05 | 3151.62 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 13:15:00 | 3134.05 | 3148.76 | 3149.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 3115.00 | 3139.74 | 3144.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 13:15:00 | 3162.00 | 3139.65 | 3142.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 13:15:00 | 3162.00 | 3139.65 | 3142.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 3162.00 | 3139.65 | 3142.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:45:00 | 3165.00 | 3139.65 | 3142.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 3161.00 | 3143.92 | 3144.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 3161.00 | 3143.92 | 3144.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 3234.10 | 3162.13 | 3152.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 3255.00 | 3203.87 | 3177.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 12:15:00 | 3284.15 | 3285.46 | 3255.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 13:00:00 | 3284.15 | 3285.46 | 3255.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 3263.40 | 3282.12 | 3259.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 3263.40 | 3282.12 | 3259.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 3270.90 | 3279.88 | 3260.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 3303.10 | 3279.88 | 3260.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:30:00 | 3297.05 | 3290.08 | 3268.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 11:00:00 | 3317.05 | 3290.08 | 3268.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 3208.30 | 3297.83 | 3308.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 3208.30 | 3297.83 | 3308.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 3144.75 | 3267.21 | 3293.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3222.50 | 3197.74 | 3238.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 3222.50 | 3197.74 | 3238.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 3203.90 | 3175.31 | 3204.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:00:00 | 3203.90 | 3175.31 | 3204.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 3219.65 | 3184.18 | 3206.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 3216.80 | 3184.18 | 3206.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 3201.00 | 3187.54 | 3205.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:15:00 | 3190.00 | 3188.40 | 3204.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 3184.90 | 3191.74 | 3202.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 11:15:00 | 3189.85 | 3193.89 | 3201.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 15:15:00 | 3030.50 | 3106.06 | 3141.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 15:15:00 | 3025.65 | 3106.06 | 3141.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 15:15:00 | 3030.36 | 3106.06 | 3141.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-12 09:15:00 | 2871.00 | 3059.98 | 3117.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 110 — BUY (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 13:15:00 | 2849.05 | 2824.13 | 2823.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 10:15:00 | 2864.00 | 2846.81 | 2838.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 14:15:00 | 2855.45 | 2866.33 | 2858.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 14:15:00 | 2855.45 | 2866.33 | 2858.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 2855.45 | 2866.33 | 2858.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 2855.45 | 2866.33 | 2858.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 2858.85 | 2864.83 | 2858.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:30:00 | 2844.40 | 2861.28 | 2857.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 2844.50 | 2857.93 | 2856.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 2844.50 | 2857.93 | 2856.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 12:15:00 | 2842.10 | 2854.07 | 2854.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 10:15:00 | 2836.55 | 2848.76 | 2852.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 10:15:00 | 2829.00 | 2828.85 | 2837.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-28 10:30:00 | 2831.45 | 2828.85 | 2837.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 2827.35 | 2822.82 | 2828.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:00:00 | 2827.35 | 2822.82 | 2828.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 2826.95 | 2823.64 | 2828.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:45:00 | 2840.00 | 2823.64 | 2828.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 2842.85 | 2827.48 | 2829.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 2842.85 | 2827.48 | 2829.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 2860.00 | 2833.99 | 2832.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 2870.00 | 2848.54 | 2841.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 2945.90 | 2951.42 | 2937.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 2945.90 | 2951.42 | 2937.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 2945.90 | 2951.42 | 2937.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 2945.90 | 2951.42 | 2937.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 2971.75 | 2955.49 | 2940.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:30:00 | 2981.20 | 2959.54 | 2943.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:15:00 | 2980.00 | 2959.54 | 2943.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:45:00 | 2979.70 | 2964.76 | 2948.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 10:30:00 | 2979.95 | 2970.39 | 2956.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 3072.95 | 3098.48 | 3088.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 3072.95 | 3098.48 | 3088.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 3072.10 | 3093.21 | 3087.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 3072.10 | 3093.21 | 3087.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-16 13:15:00 | 3065.85 | 3082.01 | 3082.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 3065.85 | 3082.01 | 3082.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 3033.30 | 3068.83 | 3076.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 3078.35 | 3065.92 | 3071.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 13:15:00 | 3078.35 | 3065.92 | 3071.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 3078.35 | 3065.92 | 3071.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:00:00 | 3078.35 | 3065.92 | 3071.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 3075.15 | 3067.76 | 3072.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 3066.90 | 3069.21 | 3072.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:45:00 | 3066.95 | 3047.58 | 3054.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 14:15:00 | 3087.95 | 3058.62 | 3058.65 | SL hit (close>static) qty=1.00 sl=3083.15 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 15:15:00 | 3081.05 | 3063.10 | 3060.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 3098.80 | 3073.74 | 3066.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 13:15:00 | 3069.00 | 3077.12 | 3069.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 13:15:00 | 3069.00 | 3077.12 | 3069.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 3069.00 | 3077.12 | 3069.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:30:00 | 3061.95 | 3077.12 | 3069.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 3077.05 | 3077.10 | 3070.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 3077.05 | 3077.10 | 3070.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 3077.15 | 3080.78 | 3073.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:00:00 | 3077.15 | 3080.78 | 3073.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 3073.10 | 3079.24 | 3073.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:00:00 | 3073.10 | 3079.24 | 3073.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 3084.50 | 3080.29 | 3074.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 12:45:00 | 3090.50 | 3082.23 | 3075.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 13:30:00 | 3095.20 | 3084.82 | 3077.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 3042.00 | 3093.06 | 3095.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 3042.00 | 3093.06 | 3095.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 13:15:00 | 3031.40 | 3072.77 | 3085.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 15:15:00 | 3015.00 | 3011.37 | 3037.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:15:00 | 3025.05 | 3011.37 | 3037.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 3048.70 | 3018.83 | 3038.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 3048.70 | 3018.83 | 3038.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 3074.90 | 3030.05 | 3041.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 3077.60 | 3030.05 | 3041.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 3068.70 | 3047.87 | 3047.86 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 3004.80 | 3039.78 | 3044.23 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 3077.15 | 3046.01 | 3043.32 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 2952.50 | 3026.79 | 3036.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 2927.35 | 2974.80 | 3003.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2873.85 | 2846.78 | 2887.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 2873.85 | 2846.78 | 2887.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 2939.00 | 2865.22 | 2892.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 2939.00 | 2865.22 | 2892.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 2957.00 | 2883.58 | 2898.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:45:00 | 2957.00 | 2883.58 | 2898.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 2964.60 | 2912.42 | 2909.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 2997.70 | 2929.47 | 2917.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 3038.10 | 3046.69 | 3019.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:00:00 | 3038.10 | 3046.69 | 3019.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 2983.70 | 3034.09 | 3016.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 2983.70 | 3034.09 | 3016.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 3004.65 | 3028.21 | 3015.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:30:00 | 3019.15 | 3028.51 | 3016.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:45:00 | 3017.00 | 3023.06 | 3015.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 15:15:00 | 3039.00 | 3023.06 | 3015.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 13:15:00 | 2999.00 | 3014.05 | 3015.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 13:15:00 | 2999.00 | 3014.05 | 3015.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 14:15:00 | 2983.40 | 3007.92 | 3012.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 3005.85 | 3001.64 | 3007.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-15 11:00:00 | 3005.85 | 3001.64 | 3007.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 3000.35 | 3001.38 | 3007.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:00:00 | 2991.40 | 2999.39 | 3005.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 14:15:00 | 3030.45 | 3007.71 | 3008.49 | SL hit (close>static) qty=1.00 sl=3014.60 alert=retest2 |

### Cycle 122 — BUY (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 14:15:00 | 3028.45 | 3011.12 | 3009.25 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 2952.00 | 3002.31 | 3005.76 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 3010.65 | 2990.28 | 2989.12 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 2980.00 | 2988.23 | 2988.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 2979.65 | 2986.05 | 2987.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 2956.15 | 2955.15 | 2965.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 10:15:00 | 2956.15 | 2955.15 | 2965.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 2956.15 | 2955.15 | 2965.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 2956.15 | 2955.15 | 2965.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 2947.85 | 2953.69 | 2963.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 2962.45 | 2953.69 | 2963.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 2957.20 | 2954.39 | 2963.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 2957.20 | 2954.39 | 2963.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 2940.20 | 2951.55 | 2961.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 2963.60 | 2951.55 | 2961.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 2937.10 | 2948.76 | 2957.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:00:00 | 2909.05 | 2934.18 | 2945.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:30:00 | 2908.25 | 2929.13 | 2941.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:30:00 | 2905.00 | 2925.84 | 2939.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 12:15:00 | 2903.25 | 2925.84 | 2939.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 2932.65 | 2913.29 | 2926.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:45:00 | 2943.85 | 2913.29 | 2926.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 2964.20 | 2923.47 | 2930.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 2964.20 | 2923.47 | 2930.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 2975.50 | 2933.88 | 2934.14 | SL hit (close>static) qty=1.00 sl=2967.20 alert=retest2 |

### Cycle 126 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 2949.15 | 2936.93 | 2935.50 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 15:15:00 | 2920.05 | 2932.22 | 2933.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 2883.05 | 2922.39 | 2929.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 2892.00 | 2890.70 | 2903.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 11:15:00 | 2892.00 | 2890.70 | 2903.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 2892.00 | 2890.70 | 2903.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:45:00 | 2897.00 | 2890.70 | 2903.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 2895.45 | 2892.44 | 2902.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:45:00 | 2897.00 | 2892.44 | 2902.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 2875.20 | 2888.99 | 2899.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 09:15:00 | 2856.05 | 2887.68 | 2898.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 12:15:00 | 2871.15 | 2827.14 | 2821.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 2871.15 | 2827.14 | 2821.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 2876.25 | 2858.37 | 2843.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 2860.45 | 2860.97 | 2847.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 2860.45 | 2860.97 | 2847.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2847.85 | 2861.32 | 2850.20 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 2823.65 | 2842.03 | 2843.53 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 2867.00 | 2848.38 | 2845.84 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 2815.80 | 2841.33 | 2843.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 2777.05 | 2828.47 | 2837.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2750.65 | 2713.11 | 2744.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 2750.65 | 2713.11 | 2744.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 2750.65 | 2713.11 | 2744.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:30:00 | 2754.00 | 2713.11 | 2744.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 2729.00 | 2716.29 | 2742.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 2749.50 | 2716.29 | 2742.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 2740.40 | 2722.67 | 2739.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:30:00 | 2738.80 | 2722.67 | 2739.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 2737.65 | 2725.67 | 2739.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:30:00 | 2747.95 | 2725.67 | 2739.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 2744.90 | 2729.51 | 2739.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 2740.80 | 2729.51 | 2739.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 2746.70 | 2732.95 | 2740.38 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 15:15:00 | 2745.90 | 2744.22 | 2744.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 2760.55 | 2747.49 | 2745.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 2765.30 | 2774.13 | 2761.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 2765.30 | 2774.13 | 2761.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 2763.00 | 2771.90 | 2761.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 2744.85 | 2771.90 | 2761.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 2745.30 | 2766.58 | 2760.34 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 12:15:00 | 2748.10 | 2756.98 | 2757.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 14:15:00 | 2723.20 | 2747.55 | 2752.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 2754.65 | 2741.63 | 2747.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 11:15:00 | 2754.65 | 2741.63 | 2747.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 2754.65 | 2741.63 | 2747.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 2754.65 | 2741.63 | 2747.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 2763.30 | 2745.97 | 2748.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 2763.30 | 2745.97 | 2748.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 2768.55 | 2754.10 | 2752.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 2790.10 | 2762.82 | 2756.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 13:15:00 | 2769.95 | 2774.87 | 2765.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 13:15:00 | 2769.95 | 2774.87 | 2765.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 2769.95 | 2774.87 | 2765.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:00:00 | 2769.95 | 2774.87 | 2765.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 2746.20 | 2769.13 | 2763.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 2746.20 | 2769.13 | 2763.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 2750.00 | 2765.31 | 2762.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 2754.60 | 2765.31 | 2762.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:45:00 | 2755.80 | 2764.43 | 2762.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 09:15:00 | 2732.40 | 2757.79 | 2760.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 09:15:00 | 2732.40 | 2757.79 | 2760.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 2706.75 | 2731.98 | 2743.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 2732.25 | 2724.78 | 2732.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 2732.25 | 2724.78 | 2732.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 2732.25 | 2724.78 | 2732.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:30:00 | 2732.60 | 2724.78 | 2732.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 2742.45 | 2728.31 | 2733.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 2745.75 | 2728.31 | 2733.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 2747.95 | 2732.24 | 2734.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 2750.35 | 2732.24 | 2734.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 2766.50 | 2739.09 | 2737.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 14:15:00 | 2778.00 | 2746.87 | 2741.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 2857.80 | 2868.22 | 2845.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:00:00 | 2857.80 | 2868.22 | 2845.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2844.40 | 2863.45 | 2844.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 2844.40 | 2863.45 | 2844.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 2846.45 | 2860.05 | 2845.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:45:00 | 2829.95 | 2860.05 | 2845.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 2850.00 | 2858.04 | 2845.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:30:00 | 2850.00 | 2858.04 | 2845.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 2841.65 | 2854.76 | 2845.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:00:00 | 2841.65 | 2854.76 | 2845.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 2840.70 | 2851.95 | 2844.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 2824.45 | 2851.95 | 2844.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 2839.00 | 2849.36 | 2844.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 2837.35 | 2849.36 | 2844.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 2828.60 | 2842.60 | 2841.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 2828.60 | 2842.60 | 2841.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 11:15:00 | 2828.15 | 2839.71 | 2840.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 14:15:00 | 2810.95 | 2828.44 | 2834.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 10:15:00 | 2878.15 | 2834.26 | 2835.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 10:15:00 | 2878.15 | 2834.26 | 2835.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 2878.15 | 2834.26 | 2835.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:00:00 | 2878.15 | 2834.26 | 2835.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 2878.90 | 2843.18 | 2839.34 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 2818.50 | 2839.80 | 2840.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 2806.00 | 2833.04 | 2837.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 2828.00 | 2825.68 | 2831.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 2828.00 | 2825.68 | 2831.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 2828.00 | 2825.68 | 2831.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 15:15:00 | 2816.00 | 2826.77 | 2830.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:00:00 | 2805.60 | 2810.94 | 2818.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 12:15:00 | 2848.00 | 2824.75 | 2822.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 12:15:00 | 2848.00 | 2824.75 | 2822.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 2852.50 | 2833.56 | 2827.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 2848.05 | 2860.94 | 2850.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 2848.05 | 2860.94 | 2850.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 2848.05 | 2860.94 | 2850.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 2848.05 | 2860.94 | 2850.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 2824.35 | 2853.62 | 2848.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 2824.35 | 2853.62 | 2848.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 2818.25 | 2846.55 | 2845.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 2818.25 | 2846.55 | 2845.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 2812.20 | 2839.68 | 2842.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 2798.00 | 2824.09 | 2830.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 13:15:00 | 2815.25 | 2808.74 | 2819.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 13:15:00 | 2815.25 | 2808.74 | 2819.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 2815.25 | 2808.74 | 2819.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 2815.25 | 2808.74 | 2819.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 2820.05 | 2811.00 | 2819.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 2820.05 | 2811.00 | 2819.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2824.00 | 2813.60 | 2819.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 2837.50 | 2813.60 | 2819.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2812.55 | 2813.39 | 2819.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 2846.00 | 2813.39 | 2819.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 2802.05 | 2811.12 | 2817.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 11:15:00 | 2791.10 | 2811.12 | 2817.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 2779.95 | 2801.12 | 2810.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 2787.95 | 2801.89 | 2810.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 2824.35 | 2803.34 | 2809.19 | SL hit (close>static) qty=1.00 sl=2820.65 alert=retest2 |

### Cycle 142 — BUY (started 2024-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 15:15:00 | 2816.85 | 2811.38 | 2811.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 09:15:00 | 2821.10 | 2813.32 | 2812.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 10:15:00 | 2805.05 | 2811.67 | 2811.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 10:15:00 | 2805.05 | 2811.67 | 2811.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 2805.05 | 2811.67 | 2811.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:45:00 | 2807.55 | 2811.67 | 2811.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 11:15:00 | 2785.00 | 2806.33 | 2809.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 12:15:00 | 2777.95 | 2800.66 | 2806.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 13:15:00 | 2805.85 | 2801.70 | 2806.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 13:15:00 | 2805.85 | 2801.70 | 2806.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 2805.85 | 2801.70 | 2806.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 14:00:00 | 2805.85 | 2801.70 | 2806.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 2811.10 | 2803.58 | 2806.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 2811.10 | 2803.58 | 2806.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 2784.45 | 2799.75 | 2804.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:15:00 | 2820.55 | 2799.75 | 2804.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 2816.70 | 2803.14 | 2805.80 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 10:15:00 | 2826.40 | 2807.79 | 2807.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 2876.35 | 2830.66 | 2819.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 11:15:00 | 2862.70 | 2863.49 | 2847.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 12:00:00 | 2862.70 | 2863.49 | 2847.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 2847.30 | 2860.25 | 2847.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 2847.30 | 2860.25 | 2847.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 2839.10 | 2856.02 | 2846.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:45:00 | 2848.40 | 2856.02 | 2846.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 2850.20 | 2854.86 | 2847.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:15:00 | 2860.00 | 2854.86 | 2847.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 2860.00 | 2855.89 | 2848.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:45:00 | 2851.10 | 2854.10 | 2848.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 2863.40 | 2855.96 | 2849.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 2875.10 | 2862.10 | 2853.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:30:00 | 2876.00 | 2882.94 | 2869.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:30:00 | 2873.45 | 2876.20 | 2873.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:30:00 | 2870.50 | 2876.34 | 2873.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 2874.25 | 2875.92 | 2873.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 12:45:00 | 2866.35 | 2875.92 | 2873.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 2881.15 | 2876.97 | 2874.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:30:00 | 2871.10 | 2876.97 | 2874.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 2852.60 | 2877.18 | 2875.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:45:00 | 2856.50 | 2877.18 | 2875.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 2847.65 | 2871.27 | 2873.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 2847.65 | 2871.27 | 2873.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 2806.50 | 2837.85 | 2853.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 2747.35 | 2746.43 | 2780.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 14:00:00 | 2747.35 | 2746.43 | 2780.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 2759.10 | 2747.82 | 2765.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:45:00 | 2765.60 | 2747.82 | 2765.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 2790.60 | 2756.38 | 2767.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 2790.60 | 2756.38 | 2767.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 2780.75 | 2761.25 | 2769.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:45:00 | 2775.55 | 2761.25 | 2769.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 2765.45 | 2760.68 | 2767.33 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 13:15:00 | 2779.10 | 2771.07 | 2770.74 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 2734.75 | 2766.46 | 2768.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 2706.00 | 2740.57 | 2753.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 2689.15 | 2657.11 | 2695.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 2689.15 | 2657.11 | 2695.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 2689.15 | 2657.11 | 2695.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:45:00 | 2704.40 | 2657.11 | 2695.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 2712.20 | 2668.13 | 2697.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 2712.20 | 2668.13 | 2697.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 2700.00 | 2674.50 | 2697.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:30:00 | 2695.75 | 2684.87 | 2698.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 2721.65 | 2694.85 | 2700.78 | SL hit (close>static) qty=1.00 sl=2716.85 alert=retest2 |

### Cycle 148 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 2738.50 | 2707.08 | 2705.53 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 14:15:00 | 2676.10 | 2700.90 | 2703.51 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 2726.25 | 2700.98 | 2700.47 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 2669.95 | 2697.76 | 2700.51 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 2722.85 | 2698.69 | 2698.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 09:15:00 | 2780.05 | 2718.36 | 2707.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 2685.75 | 2729.36 | 2719.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 14:15:00 | 2685.75 | 2729.36 | 2719.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 2685.75 | 2729.36 | 2719.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 2685.75 | 2729.36 | 2719.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 2680.00 | 2719.49 | 2716.23 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 2675.40 | 2710.67 | 2712.52 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 2754.00 | 2711.87 | 2710.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 2756.40 | 2720.78 | 2714.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 14:15:00 | 2726.30 | 2729.40 | 2721.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-23 14:45:00 | 2732.85 | 2729.40 | 2721.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 2699.80 | 2722.46 | 2719.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 2699.80 | 2722.46 | 2719.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 2718.35 | 2721.63 | 2719.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:15:00 | 2702.95 | 2721.63 | 2719.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 2702.35 | 2717.78 | 2717.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 2672.45 | 2705.68 | 2712.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 2606.70 | 2603.64 | 2642.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 09:15:00 | 2606.70 | 2603.64 | 2642.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 2606.70 | 2603.64 | 2642.64 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 2735.40 | 2668.13 | 2663.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 2765.00 | 2746.08 | 2723.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 2751.05 | 2760.74 | 2742.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 2751.05 | 2760.74 | 2742.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 2761.70 | 2764.24 | 2748.90 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 2736.50 | 2750.41 | 2750.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 2723.05 | 2741.94 | 2746.75 | Break + close below crossover candle low |

### Cycle 158 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 2788.80 | 2748.51 | 2748.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 13:15:00 | 2829.70 | 2787.65 | 2772.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 2799.00 | 2801.05 | 2785.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 2799.00 | 2801.05 | 2785.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 2760.85 | 2793.13 | 2784.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 2760.85 | 2793.13 | 2784.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 2755.10 | 2785.52 | 2781.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 2755.10 | 2785.52 | 2781.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 15:15:00 | 2759.35 | 2776.80 | 2778.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 2741.00 | 2769.64 | 2774.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 2641.00 | 2638.20 | 2662.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 2639.45 | 2638.20 | 2662.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2647.50 | 2638.05 | 2654.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 2647.50 | 2638.05 | 2654.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 2673.65 | 2645.17 | 2656.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 2673.65 | 2645.17 | 2656.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 2656.65 | 2647.47 | 2656.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 2651.20 | 2649.17 | 2656.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 2672.95 | 2632.31 | 2632.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 2672.95 | 2632.31 | 2632.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 15:15:00 | 2676.50 | 2641.15 | 2636.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 10:15:00 | 2622.70 | 2637.75 | 2635.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 10:15:00 | 2622.70 | 2637.75 | 2635.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 2622.70 | 2637.75 | 2635.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 2622.70 | 2637.75 | 2635.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 2630.70 | 2636.34 | 2635.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:45:00 | 2625.50 | 2636.34 | 2635.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 12:15:00 | 2625.10 | 2634.09 | 2634.17 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 2644.30 | 2636.13 | 2635.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 2647.20 | 2638.35 | 2636.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 09:15:00 | 2638.25 | 2638.61 | 2636.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 2638.25 | 2638.61 | 2636.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 2638.25 | 2638.61 | 2636.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 11:45:00 | 2655.90 | 2639.26 | 2637.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 12:45:00 | 2654.95 | 2645.41 | 2640.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 2659.00 | 2684.46 | 2686.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 2659.00 | 2684.46 | 2686.76 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 2713.50 | 2690.62 | 2688.97 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 2680.15 | 2687.95 | 2688.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 13:15:00 | 2671.95 | 2684.68 | 2686.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 2687.40 | 2683.08 | 2685.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 2687.40 | 2683.08 | 2685.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 2687.40 | 2683.08 | 2685.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 11:45:00 | 2671.05 | 2680.37 | 2683.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 14:30:00 | 2664.25 | 2681.33 | 2683.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 15:15:00 | 2671.90 | 2681.33 | 2683.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 2537.50 | 2582.08 | 2613.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 2531.04 | 2582.08 | 2613.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 2538.30 | 2582.08 | 2613.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 2578.05 | 2539.80 | 2569.20 | SL hit (close>ema200) qty=0.50 sl=2539.80 alert=retest2 |

### Cycle 166 — BUY (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 09:15:00 | 2584.30 | 2552.93 | 2552.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-10 12:15:00 | 2594.95 | 2569.83 | 2562.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 2555.95 | 2567.05 | 2561.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 13:15:00 | 2555.95 | 2567.05 | 2561.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 2555.95 | 2567.05 | 2561.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 2555.95 | 2567.05 | 2561.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 2561.35 | 2565.91 | 2561.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 2547.25 | 2565.91 | 2561.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 2563.95 | 2565.52 | 2562.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 2525.00 | 2565.52 | 2562.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 2531.50 | 2558.72 | 2559.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 2507.50 | 2540.16 | 2550.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 2526.95 | 2525.39 | 2535.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 13:15:00 | 2526.95 | 2525.39 | 2535.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 2526.95 | 2525.39 | 2535.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:30:00 | 2511.55 | 2531.57 | 2535.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:15:00 | 2517.50 | 2531.57 | 2535.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 2515.00 | 2523.36 | 2531.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 10:30:00 | 2515.00 | 2517.39 | 2525.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 2533.00 | 2520.51 | 2525.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 2533.00 | 2520.51 | 2525.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 2525.35 | 2521.48 | 2525.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 2543.10 | 2528.66 | 2527.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 2543.10 | 2528.66 | 2527.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 2593.20 | 2552.19 | 2541.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 2587.90 | 2593.73 | 2571.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 2587.90 | 2593.73 | 2571.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 2609.00 | 2607.69 | 2594.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 2595.90 | 2607.69 | 2594.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2614.95 | 2609.14 | 2596.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:45:00 | 2600.30 | 2609.14 | 2596.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 11:15:00 | 2610.75 | 2618.06 | 2605.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 11:30:00 | 2601.15 | 2618.06 | 2605.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 2620.15 | 2618.63 | 2607.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:45:00 | 2613.15 | 2618.63 | 2607.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 2611.05 | 2617.12 | 2608.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 15:00:00 | 2611.05 | 2617.12 | 2608.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 2619.90 | 2617.67 | 2609.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 2602.15 | 2617.67 | 2609.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 2593.00 | 2612.74 | 2607.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 2593.00 | 2612.74 | 2607.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 2574.80 | 2605.15 | 2604.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 2577.90 | 2605.15 | 2604.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 2563.00 | 2596.72 | 2600.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 10:15:00 | 2536.00 | 2568.80 | 2579.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 2567.65 | 2559.09 | 2568.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 2567.65 | 2559.09 | 2568.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2567.65 | 2559.09 | 2568.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:45:00 | 2540.65 | 2554.06 | 2561.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:45:00 | 2539.85 | 2537.87 | 2548.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 2575.00 | 2555.93 | 2553.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 2575.00 | 2555.93 | 2553.79 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 2488.45 | 2542.44 | 2547.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 2397.55 | 2478.41 | 2508.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 2308.85 | 2303.33 | 2358.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 2308.85 | 2303.33 | 2358.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 2332.75 | 2295.99 | 2313.74 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 2337.35 | 2325.03 | 2323.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 2436.40 | 2349.52 | 2335.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2471.30 | 2478.31 | 2442.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 2471.30 | 2478.31 | 2442.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 2477.50 | 2493.07 | 2469.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 2513.10 | 2495.02 | 2472.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:00:00 | 2517.00 | 2499.41 | 2476.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:45:00 | 2517.60 | 2506.46 | 2484.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:30:00 | 2514.60 | 2551.17 | 2549.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 2517.60 | 2544.46 | 2546.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 2517.60 | 2544.46 | 2546.28 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 2554.20 | 2544.88 | 2543.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 13:15:00 | 2566.30 | 2549.17 | 2545.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 2661.20 | 2665.63 | 2635.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 12:00:00 | 2661.20 | 2665.63 | 2635.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2714.40 | 2721.28 | 2702.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:00:00 | 2755.70 | 2728.16 | 2707.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:45:00 | 2756.20 | 2736.40 | 2715.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:00:00 | 2755.60 | 2740.24 | 2718.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:45:00 | 2755.70 | 2748.08 | 2743.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 2742.10 | 2746.88 | 2743.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 2772.10 | 2745.22 | 2743.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 15:15:00 | 2755.00 | 2749.88 | 2747.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 10:15:00 | 2726.00 | 2745.30 | 2746.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 10:15:00 | 2726.00 | 2745.30 | 2746.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 11:15:00 | 2722.70 | 2740.78 | 2744.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 2729.40 | 2720.08 | 2730.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2729.40 | 2720.08 | 2730.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2729.40 | 2720.08 | 2730.53 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 2749.70 | 2736.66 | 2735.14 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 13:15:00 | 2708.60 | 2730.67 | 2732.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 15:15:00 | 2690.00 | 2717.90 | 2726.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 14:15:00 | 2712.80 | 2707.27 | 2715.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 14:15:00 | 2712.80 | 2707.27 | 2715.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 2712.80 | 2707.27 | 2715.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 2712.80 | 2707.27 | 2715.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2749.40 | 2716.93 | 2718.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 2749.40 | 2716.93 | 2718.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 2735.80 | 2720.71 | 2720.45 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 2694.20 | 2716.32 | 2718.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 2690.30 | 2711.11 | 2716.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2679.50 | 2664.57 | 2682.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 2679.50 | 2664.57 | 2682.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2660.30 | 2663.72 | 2680.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 2682.00 | 2663.72 | 2680.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 2653.00 | 2659.68 | 2675.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:30:00 | 2674.30 | 2659.68 | 2675.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 2655.90 | 2659.28 | 2671.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 2668.40 | 2659.28 | 2671.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2665.90 | 2660.61 | 2670.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:15:00 | 2634.00 | 2657.84 | 2667.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 2632.00 | 2648.75 | 2661.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 2641.10 | 2655.05 | 2660.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 2641.70 | 2652.38 | 2658.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 2653.80 | 2649.72 | 2656.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:45:00 | 2653.90 | 2649.72 | 2656.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 2656.70 | 2651.12 | 2656.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:30:00 | 2664.00 | 2651.12 | 2656.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 2671.00 | 2655.09 | 2657.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 2397.00 | 2655.09 | 2657.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:15:00 | 2502.30 | 2610.56 | 2637.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:15:00 | 2500.40 | 2610.56 | 2637.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:15:00 | 2509.04 | 2610.56 | 2637.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:15:00 | 2509.61 | 2610.56 | 2637.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 13:15:00 | 2476.30 | 2476.16 | 2507.67 | SL hit (close>ema200) qty=0.50 sl=2476.16 alert=retest2 |

### Cycle 180 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 2471.70 | 2460.27 | 2459.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 2489.70 | 2468.77 | 2464.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 11:15:00 | 2511.40 | 2515.95 | 2505.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 12:00:00 | 2511.40 | 2515.95 | 2505.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 2500.00 | 2512.76 | 2505.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 2500.00 | 2512.76 | 2505.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 2494.90 | 2509.19 | 2504.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 2497.00 | 2509.19 | 2504.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 2497.50 | 2504.95 | 2502.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 2494.90 | 2504.95 | 2502.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 2477.50 | 2497.56 | 2499.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 14:15:00 | 2469.50 | 2482.56 | 2491.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 2465.80 | 2464.18 | 2474.65 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 2440.00 | 2464.18 | 2474.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 2462.80 | 2455.32 | 2464.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 2462.80 | 2455.32 | 2464.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2460.90 | 2457.14 | 2463.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 2469.60 | 2459.63 | 2464.05 | SL hit (close>ema400) qty=1.00 sl=2464.05 alert=retest1 |

### Cycle 182 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 2432.00 | 2408.49 | 2406.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 2434.90 | 2423.33 | 2415.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 2460.00 | 2468.73 | 2451.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 13:15:00 | 2460.00 | 2468.73 | 2451.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 2460.00 | 2468.73 | 2451.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 2456.00 | 2468.73 | 2451.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 2449.60 | 2464.90 | 2450.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 2449.60 | 2464.90 | 2450.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 2440.00 | 2459.92 | 2449.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 2458.00 | 2459.54 | 2450.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-17 09:15:00 | 2703.80 | 2667.59 | 2659.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 2724.50 | 2742.45 | 2742.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 2688.80 | 2725.90 | 2733.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2705.00 | 2695.72 | 2710.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 2705.00 | 2695.72 | 2710.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 2701.20 | 2696.82 | 2710.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 2736.40 | 2696.82 | 2710.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 2707.10 | 2698.87 | 2709.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:45:00 | 2722.20 | 2698.87 | 2709.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 2700.10 | 2699.12 | 2708.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:30:00 | 2700.30 | 2699.12 | 2708.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 2726.40 | 2702.44 | 2707.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:45:00 | 2719.20 | 2702.44 | 2707.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 2724.70 | 2706.89 | 2708.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 2732.00 | 2706.89 | 2708.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 11:15:00 | 2728.30 | 2711.17 | 2710.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 2741.10 | 2717.16 | 2713.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 2704.40 | 2738.94 | 2732.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 2704.40 | 2738.94 | 2732.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2704.40 | 2738.94 | 2732.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:15:00 | 2704.70 | 2738.94 | 2732.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 2696.60 | 2724.47 | 2726.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 12:15:00 | 2688.60 | 2717.29 | 2722.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 2585.00 | 2583.97 | 2615.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:15:00 | 2575.20 | 2583.97 | 2615.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2412.70 | 2424.66 | 2440.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 10:30:00 | 2396.20 | 2419.99 | 2436.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:15:00 | 2400.10 | 2419.99 | 2436.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:00:00 | 2400.10 | 2416.01 | 2433.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 2390.40 | 2413.69 | 2426.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 2402.20 | 2411.39 | 2424.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 2440.00 | 2412.69 | 2411.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 2440.00 | 2412.69 | 2411.79 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 2413.60 | 2420.34 | 2420.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 2402.50 | 2416.77 | 2419.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 2420.40 | 2415.60 | 2418.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 13:15:00 | 2420.40 | 2415.60 | 2418.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 2420.40 | 2415.60 | 2418.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 2420.40 | 2415.60 | 2418.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 2405.20 | 2413.52 | 2417.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 2401.10 | 2412.87 | 2416.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 2281.04 | 2315.75 | 2337.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 2291.90 | 2287.46 | 2305.94 | SL hit (close>ema200) qty=0.50 sl=2287.46 alert=retest2 |

### Cycle 188 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 2351.80 | 2314.54 | 2313.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 2374.90 | 2335.27 | 2323.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 2350.40 | 2352.15 | 2337.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 2350.40 | 2352.15 | 2337.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 2366.90 | 2354.43 | 2341.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:30:00 | 2350.00 | 2354.43 | 2341.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 2349.40 | 2361.50 | 2352.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 2349.40 | 2361.50 | 2352.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 2345.00 | 2358.20 | 2351.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 2345.00 | 2358.20 | 2351.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 2338.40 | 2354.24 | 2350.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 2338.40 | 2354.24 | 2350.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 2334.90 | 2347.76 | 2348.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 2310.00 | 2337.45 | 2343.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 2350.90 | 2328.23 | 2335.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 2350.90 | 2328.23 | 2335.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2350.90 | 2328.23 | 2335.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 2350.90 | 2328.23 | 2335.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 2328.40 | 2328.27 | 2334.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 2320.40 | 2328.27 | 2334.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:15:00 | 2323.40 | 2328.19 | 2333.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 2320.30 | 2327.65 | 2333.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:30:00 | 2322.70 | 2325.88 | 2331.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2311.60 | 2322.66 | 2328.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:30:00 | 2306.80 | 2319.83 | 2327.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:00:00 | 2308.50 | 2319.83 | 2327.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:30:00 | 2307.60 | 2315.29 | 2323.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 2361.80 | 2336.01 | 2329.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 13:15:00 | 2394.90 | 2409.89 | 2386.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:00:00 | 2394.90 | 2409.89 | 2386.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 2388.50 | 2405.61 | 2387.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:15:00 | 2387.00 | 2405.61 | 2387.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 2387.00 | 2401.89 | 2387.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 2406.00 | 2401.89 | 2387.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 15:15:00 | 2489.00 | 2503.97 | 2504.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 2489.00 | 2503.97 | 2504.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 2455.40 | 2494.26 | 2500.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 11:15:00 | 2440.50 | 2433.22 | 2455.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 12:00:00 | 2440.50 | 2433.22 | 2455.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 2307.30 | 2300.30 | 2312.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 2292.10 | 2301.29 | 2306.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 12:30:00 | 2289.40 | 2298.10 | 2304.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 11:30:00 | 2289.50 | 2295.94 | 2300.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 11:15:00 | 2320.10 | 2296.36 | 2296.79 | SL hit (close>static) qty=1.00 sl=2317.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 2302.00 | 2297.49 | 2297.26 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 2295.00 | 2296.99 | 2297.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 2281.80 | 2293.95 | 2295.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 2297.90 | 2293.25 | 2294.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 2297.90 | 2293.25 | 2294.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2297.90 | 2293.25 | 2294.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 2297.90 | 2293.25 | 2294.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2270.30 | 2288.66 | 2292.71 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 2295.80 | 2290.23 | 2290.18 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 2287.10 | 2289.71 | 2289.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 2247.30 | 2280.83 | 2285.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 10:15:00 | 2233.00 | 2232.41 | 2252.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 11:00:00 | 2233.00 | 2232.41 | 2252.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 2220.00 | 2228.22 | 2244.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 2242.00 | 2228.22 | 2244.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2235.30 | 2222.71 | 2231.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 2238.10 | 2222.71 | 2231.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 2233.40 | 2224.85 | 2231.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 2231.00 | 2224.85 | 2231.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 2238.00 | 2227.48 | 2231.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 2230.10 | 2227.48 | 2231.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 2243.70 | 2230.72 | 2232.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 2245.80 | 2230.72 | 2232.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 2263.00 | 2237.18 | 2235.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 2268.30 | 2243.40 | 2238.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 15:15:00 | 2265.00 | 2265.85 | 2256.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:15:00 | 2272.30 | 2265.85 | 2256.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 2278.00 | 2286.97 | 2275.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 2301.50 | 2289.77 | 2277.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 12:15:00 | 2322.10 | 2329.00 | 2329.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 2322.10 | 2329.00 | 2329.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 2310.20 | 2325.24 | 2327.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 15:15:00 | 2330.00 | 2326.04 | 2327.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 15:15:00 | 2330.00 | 2326.04 | 2327.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 2330.00 | 2326.04 | 2327.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 2330.00 | 2326.04 | 2327.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 2335.00 | 2327.83 | 2328.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 2335.00 | 2327.83 | 2328.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 2327.10 | 2327.69 | 2328.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 2330.90 | 2327.69 | 2328.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 2328.70 | 2327.89 | 2328.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 2328.70 | 2327.89 | 2328.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 2303.30 | 2322.97 | 2326.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:15:00 | 2290.10 | 2318.40 | 2323.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 2175.59 | 2293.18 | 2309.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 2290.40 | 2283.07 | 2297.14 | SL hit (close>ema200) qty=0.50 sl=2283.07 alert=retest2 |

### Cycle 198 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 2347.90 | 2308.14 | 2305.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 2355.10 | 2317.53 | 2310.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 2299.90 | 2334.45 | 2323.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 2299.90 | 2334.45 | 2323.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2299.90 | 2334.45 | 2323.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 2299.90 | 2334.45 | 2323.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 2330.50 | 2333.66 | 2324.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 2337.20 | 2333.66 | 2324.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 2315.60 | 2325.13 | 2325.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 2315.60 | 2325.13 | 2325.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 2301.00 | 2315.11 | 2318.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 11:15:00 | 2315.80 | 2311.65 | 2315.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 12:00:00 | 2315.80 | 2311.65 | 2315.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 2331.90 | 2315.70 | 2317.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:45:00 | 2329.90 | 2315.70 | 2317.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 2342.90 | 2321.14 | 2319.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 2347.10 | 2326.33 | 2322.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 2363.50 | 2368.71 | 2355.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:30:00 | 2360.30 | 2368.71 | 2355.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 2331.80 | 2360.28 | 2354.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 2331.80 | 2360.28 | 2354.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 2340.30 | 2356.29 | 2352.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 2332.90 | 2356.29 | 2352.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 2323.20 | 2349.67 | 2350.28 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 2361.70 | 2347.22 | 2346.19 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2316.30 | 2347.67 | 2347.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 2308.10 | 2339.75 | 2344.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 2317.90 | 2316.91 | 2327.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 2317.90 | 2316.91 | 2327.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2317.90 | 2316.91 | 2327.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 2319.20 | 2316.91 | 2327.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 2322.20 | 2317.16 | 2324.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 2322.20 | 2317.16 | 2324.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2321.00 | 2317.92 | 2324.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2317.80 | 2317.92 | 2324.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2287.10 | 2311.76 | 2320.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:30:00 | 2269.90 | 2303.37 | 2316.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:00:00 | 2269.10 | 2291.14 | 2307.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 2339.80 | 2302.13 | 2307.73 | SL hit (close>static) qty=1.00 sl=2336.00 alert=retest2 |

### Cycle 204 — BUY (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 11:15:00 | 2341.00 | 2313.82 | 2312.31 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 2281.60 | 2311.23 | 2313.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 2263.10 | 2297.34 | 2306.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 2287.30 | 2278.81 | 2291.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 2287.30 | 2278.81 | 2291.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2309.40 | 2284.93 | 2292.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 2309.40 | 2284.93 | 2292.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 2303.00 | 2288.54 | 2293.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 2274.90 | 2288.54 | 2293.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 2312.40 | 2297.12 | 2296.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 2324.20 | 2305.25 | 2300.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 2299.30 | 2307.31 | 2303.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 2299.30 | 2307.31 | 2303.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 2299.30 | 2307.31 | 2303.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 2299.30 | 2307.31 | 2303.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 2295.20 | 2304.89 | 2303.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 2296.10 | 2304.89 | 2303.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 2298.20 | 2301.97 | 2302.01 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 14:15:00 | 2303.80 | 2302.34 | 2302.17 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 2300.00 | 2301.87 | 2301.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 2268.00 | 2295.09 | 2298.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 2294.70 | 2288.81 | 2294.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 12:15:00 | 2294.70 | 2288.81 | 2294.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 2294.70 | 2288.81 | 2294.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:30:00 | 2298.10 | 2288.81 | 2294.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 2296.70 | 2290.39 | 2294.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:30:00 | 2296.40 | 2290.39 | 2294.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 2315.10 | 2295.33 | 2296.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 2315.10 | 2295.33 | 2296.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 2310.00 | 2298.27 | 2297.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 2368.20 | 2314.76 | 2306.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 14:15:00 | 2409.90 | 2411.98 | 2383.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 2409.90 | 2411.98 | 2383.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 2395.00 | 2405.39 | 2385.44 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 2379.80 | 2388.31 | 2389.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 2356.00 | 2382.76 | 2386.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 15:15:00 | 2324.90 | 2323.69 | 2336.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:15:00 | 2313.50 | 2323.69 | 2336.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 2315.80 | 2313.10 | 2324.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 2315.80 | 2313.10 | 2324.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2317.60 | 2313.60 | 2322.85 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 2350.10 | 2329.30 | 2327.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 2357.00 | 2339.88 | 2333.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 2369.30 | 2370.36 | 2357.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:45:00 | 2370.00 | 2370.36 | 2357.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 2350.30 | 2385.66 | 2378.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:30:00 | 2348.00 | 2385.66 | 2378.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 2350.50 | 2378.63 | 2375.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 2350.50 | 2378.63 | 2375.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 11:15:00 | 2350.40 | 2372.98 | 2373.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 09:15:00 | 2346.00 | 2357.97 | 2363.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 2303.00 | 2299.16 | 2306.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 2303.00 | 2299.16 | 2306.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 2303.00 | 2299.16 | 2306.24 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 2309.50 | 2305.22 | 2305.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 2322.70 | 2308.72 | 2306.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 2306.60 | 2308.29 | 2306.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 2306.60 | 2308.29 | 2306.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 2306.60 | 2308.29 | 2306.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 2323.90 | 2315.79 | 2311.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:45:00 | 2324.90 | 2316.73 | 2312.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 15:15:00 | 2306.00 | 2314.58 | 2311.62 | SL hit (close<static) qty=1.00 sl=2306.60 alert=retest2 |

### Cycle 215 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 2371.70 | 2380.91 | 2381.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 2349.20 | 2371.92 | 2376.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 2368.00 | 2365.56 | 2372.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 15:15:00 | 2368.00 | 2365.56 | 2372.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 2368.00 | 2365.56 | 2372.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 2359.40 | 2365.56 | 2372.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2383.70 | 2369.19 | 2373.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:30:00 | 2385.80 | 2369.19 | 2373.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 2386.70 | 2372.69 | 2374.49 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 2394.40 | 2377.03 | 2376.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 12:15:00 | 2400.00 | 2381.62 | 2378.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 09:15:00 | 2395.80 | 2396.92 | 2387.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:30:00 | 2390.30 | 2396.92 | 2387.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 2399.60 | 2400.70 | 2393.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 2388.00 | 2400.70 | 2393.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2411.00 | 2402.00 | 2395.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 11:30:00 | 2421.00 | 2405.19 | 2397.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 12:30:00 | 2421.20 | 2404.15 | 2398.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 14:45:00 | 2432.90 | 2408.79 | 2401.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:30:00 | 2420.70 | 2413.79 | 2405.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 2439.60 | 2432.10 | 2422.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 2407.10 | 2418.01 | 2418.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 2407.10 | 2418.01 | 2418.65 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 2425.00 | 2418.21 | 2417.31 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 2412.00 | 2417.78 | 2418.00 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 2450.90 | 2420.27 | 2418.34 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 2412.30 | 2418.29 | 2418.56 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 2433.00 | 2421.46 | 2419.97 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 14:15:00 | 2411.80 | 2419.62 | 2419.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 2385.00 | 2410.68 | 2415.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 2309.40 | 2305.41 | 2326.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 2309.40 | 2305.41 | 2326.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2309.40 | 2305.41 | 2326.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 2309.40 | 2305.41 | 2326.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 2294.90 | 2278.63 | 2302.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 2294.90 | 2278.63 | 2302.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2304.00 | 2283.70 | 2302.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2474.20 | 2283.70 | 2302.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2424.90 | 2311.94 | 2313.37 | EMA400 retest candle locked (from downside) |

### Cycle 224 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 2497.50 | 2349.05 | 2330.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 2529.00 | 2385.04 | 2348.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 2600.10 | 2601.02 | 2530.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:45:00 | 2602.50 | 2601.02 | 2530.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2605.00 | 2642.05 | 2594.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 2592.60 | 2642.05 | 2594.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2681.20 | 2706.40 | 2687.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 2681.20 | 2706.40 | 2687.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 2684.40 | 2702.00 | 2687.43 | EMA400 retest candle locked (from upside) |

### Cycle 225 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 2661.80 | 2678.25 | 2680.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 2637.50 | 2670.10 | 2676.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 2579.10 | 2577.70 | 2601.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 2579.10 | 2577.70 | 2601.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2591.40 | 2549.80 | 2557.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 2591.40 | 2549.80 | 2557.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 2595.10 | 2558.86 | 2560.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 2595.10 | 2558.86 | 2560.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — BUY (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 11:15:00 | 2589.50 | 2564.99 | 2563.25 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 2536.00 | 2561.56 | 2562.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 2514.60 | 2547.12 | 2555.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 13:15:00 | 2507.90 | 2507.58 | 2522.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 14:00:00 | 2507.90 | 2507.58 | 2522.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 2531.20 | 2502.10 | 2510.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 2531.20 | 2502.10 | 2510.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 2508.00 | 2503.28 | 2510.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:00:00 | 2489.10 | 2503.21 | 2507.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 2489.00 | 2506.01 | 2508.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2364.64 | 2391.52 | 2430.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2364.55 | 2391.52 | 2430.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-05 10:15:00 | 2240.19 | 2261.26 | 2304.65 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 228 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 2253.50 | 2246.80 | 2246.78 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 2233.50 | 2244.14 | 2245.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 2225.00 | 2240.31 | 2243.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 2225.00 | 2203.70 | 2216.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 12:15:00 | 2225.00 | 2203.70 | 2216.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 2225.00 | 2203.70 | 2216.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:45:00 | 2226.70 | 2203.70 | 2216.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 2241.30 | 2211.22 | 2218.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 2240.00 | 2211.22 | 2218.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 2286.20 | 2226.22 | 2225.10 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 2132.40 | 2212.57 | 2223.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 2120.20 | 2146.52 | 2157.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 2059.70 | 2050.91 | 2075.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 2057.00 | 2050.91 | 2075.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 2063.00 | 2053.33 | 2074.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 2079.30 | 2053.33 | 2074.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 2059.80 | 2054.62 | 2072.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 2050.00 | 2054.62 | 2072.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 2143.50 | 2071.66 | 2077.42 | SL hit (close>static) qty=1.00 sl=2073.90 alert=retest2 |

### Cycle 232 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2153.90 | 2088.11 | 2084.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 2154.60 | 2101.40 | 2090.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2146.50 | 2150.50 | 2122.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 2146.50 | 2150.50 | 2122.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 2130.40 | 2143.55 | 2124.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 2145.90 | 2141.56 | 2124.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 2074.60 | 2133.41 | 2127.14 | SL hit (close<static) qty=1.00 sl=2118.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 2088.40 | 2117.14 | 2120.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 2078.40 | 2109.39 | 2116.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2108.80 | 2097.29 | 2107.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2108.80 | 2097.29 | 2107.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2108.80 | 2097.29 | 2107.26 | EMA400 retest candle locked (from downside) |

### Cycle 234 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 2130.00 | 2112.87 | 2111.17 | EMA200 above EMA400 |

### Cycle 235 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 2027.00 | 2095.70 | 2103.52 | EMA200 below EMA400 |

### Cycle 236 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 2118.10 | 2086.19 | 2085.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2221.40 | 2138.37 | 2115.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 2194.30 | 2198.54 | 2164.14 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 2274.40 | 2217.40 | 2190.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2228.30 | 2251.95 | 2228.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 2228.30 | 2251.95 | 2228.60 | SL hit (close<ema400) qty=1.00 sl=2228.60 alert=retest1 |

### Cycle 237 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 2279.80 | 2303.61 | 2304.10 | EMA200 below EMA400 |

### Cycle 238 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 2304.70 | 2300.45 | 2300.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 2311.30 | 2304.10 | 2301.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 2283.00 | 2302.79 | 2302.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 2283.00 | 2302.79 | 2302.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2283.00 | 2302.79 | 2302.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 2288.00 | 2302.79 | 2302.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 239 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 2282.00 | 2298.63 | 2300.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 2270.10 | 2292.93 | 2297.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 2236.20 | 2231.55 | 2256.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 2236.20 | 2231.55 | 2256.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 2244.00 | 2235.43 | 2253.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 2246.90 | 2235.43 | 2253.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 2199.70 | 2222.61 | 2237.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 2195.00 | 2213.47 | 2230.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 2175.90 | 2212.49 | 2218.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:30:00 | 2195.20 | 2180.85 | 2189.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 2187.80 | 2183.20 | 2189.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 2195.20 | 2185.60 | 2190.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:45:00 | 2196.00 | 2185.60 | 2190.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 2204.80 | 2195.01 | 2193.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 240 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 2204.80 | 2195.01 | 2193.69 | EMA200 above EMA400 |

### Cycle 241 — SELL (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 14:15:00 | 2177.90 | 2191.35 | 2192.91 | EMA200 below EMA400 |

### Cycle 242 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 2203.80 | 2193.74 | 2193.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 2218.00 | 2204.10 | 2198.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 2256.80 | 2272.50 | 2252.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 14:15:00 | 2256.80 | 2272.50 | 2252.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 2256.80 | 2272.50 | 2252.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 2256.80 | 2272.50 | 2252.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 2265.10 | 2271.02 | 2253.77 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-06-02 11:00:00 | 2277.70 | 2023-06-06 09:15:00 | 2293.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2023-06-05 09:15:00 | 2273.20 | 2023-06-06 09:15:00 | 2293.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-06-05 12:45:00 | 2280.50 | 2023-06-06 09:15:00 | 2293.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-06-05 14:30:00 | 2278.20 | 2023-06-06 09:15:00 | 2293.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-06-15 09:15:00 | 2334.00 | 2023-06-23 11:15:00 | 2409.85 | STOP_HIT | 1.00 | 3.25% |
| SELL | retest2 | 2023-07-03 09:45:00 | 2349.20 | 2023-07-05 10:15:00 | 2393.60 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2023-07-04 10:15:00 | 2354.00 | 2023-07-05 10:15:00 | 2393.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2023-07-04 10:45:00 | 2354.90 | 2023-07-05 10:15:00 | 2393.60 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-07-05 09:15:00 | 2356.55 | 2023-07-05 10:15:00 | 2393.60 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2023-07-31 10:00:00 | 2483.85 | 2023-08-02 11:15:00 | 2461.25 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-08-03 12:15:00 | 2456.10 | 2023-08-04 11:15:00 | 2510.00 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2023-08-10 13:30:00 | 2362.35 | 2023-08-16 09:15:00 | 2378.35 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2023-08-11 09:15:00 | 2349.40 | 2023-08-16 09:15:00 | 2378.35 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-08-11 15:00:00 | 2350.00 | 2023-08-16 09:15:00 | 2378.35 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2023-08-16 09:15:00 | 2348.75 | 2023-08-16 09:15:00 | 2378.35 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-08-23 09:15:00 | 2391.00 | 2023-08-23 14:15:00 | 2376.55 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-08-24 09:15:00 | 2392.50 | 2023-08-24 10:15:00 | 2377.65 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-09-01 12:15:00 | 2354.40 | 2023-09-04 11:15:00 | 2375.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-09-01 13:00:00 | 2353.15 | 2023-09-04 11:15:00 | 2375.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-09-11 11:00:00 | 2428.40 | 2023-09-12 11:15:00 | 2383.20 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2023-09-11 13:15:00 | 2431.75 | 2023-09-12 11:15:00 | 2383.20 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-09-20 12:00:00 | 2552.50 | 2023-09-27 11:15:00 | 2541.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2023-09-20 15:00:00 | 2550.10 | 2023-09-27 11:15:00 | 2541.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2023-09-22 11:15:00 | 2549.20 | 2023-09-27 11:15:00 | 2541.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2023-10-09 13:00:00 | 2548.95 | 2023-10-09 13:15:00 | 2537.75 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-10-12 14:45:00 | 2590.70 | 2023-10-13 15:15:00 | 2584.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-10-18 09:15:00 | 2626.30 | 2023-10-18 14:15:00 | 2596.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest1 | 2023-10-18 09:45:00 | 2624.05 | 2023-10-18 14:15:00 | 2596.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest1 | 2023-10-18 10:30:00 | 2620.90 | 2023-10-18 14:15:00 | 2596.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2023-10-18 11:15:00 | 2621.80 | 2023-10-18 14:15:00 | 2596.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest1 | 2023-11-17 09:15:00 | 2671.30 | 2023-11-20 09:15:00 | 2566.45 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest1 | 2023-11-17 10:15:00 | 2660.75 | 2023-11-20 09:15:00 | 2566.45 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2023-11-30 09:30:00 | 2592.75 | 2023-11-30 13:15:00 | 2574.20 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-11-30 11:00:00 | 2597.30 | 2023-11-30 13:15:00 | 2574.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-12-07 10:15:00 | 2672.00 | 2023-12-08 12:15:00 | 2578.15 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2023-12-07 11:00:00 | 2662.45 | 2023-12-08 12:15:00 | 2578.15 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2023-12-20 12:15:00 | 2510.50 | 2023-12-26 09:15:00 | 2512.00 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2023-12-29 09:15:00 | 2534.00 | 2024-01-02 14:15:00 | 2521.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-12-29 10:00:00 | 2531.05 | 2024-01-02 14:15:00 | 2521.40 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-01-09 09:15:00 | 2592.05 | 2024-01-15 11:15:00 | 2607.90 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2024-02-01 11:15:00 | 2456.55 | 2024-02-08 09:15:00 | 2333.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-01 11:45:00 | 2440.00 | 2024-02-08 09:15:00 | 2332.20 | PARTIAL | 0.50 | 4.42% |
| SELL | retest2 | 2024-02-02 11:45:00 | 2454.95 | 2024-02-08 09:15:00 | 2333.63 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-02-05 09:45:00 | 2456.45 | 2024-02-08 13:15:00 | 2318.00 | PARTIAL | 0.50 | 5.64% |
| SELL | retest2 | 2024-02-05 13:00:00 | 2436.10 | 2024-02-08 13:15:00 | 2314.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-01 11:15:00 | 2456.55 | 2024-02-09 15:15:00 | 2305.10 | STOP_HIT | 0.50 | 6.17% |
| SELL | retest2 | 2024-02-01 11:45:00 | 2440.00 | 2024-02-09 15:15:00 | 2305.10 | STOP_HIT | 0.50 | 5.53% |
| SELL | retest2 | 2024-02-02 11:45:00 | 2454.95 | 2024-02-09 15:15:00 | 2305.10 | STOP_HIT | 0.50 | 6.10% |
| SELL | retest2 | 2024-02-05 09:45:00 | 2456.45 | 2024-02-09 15:15:00 | 2305.10 | STOP_HIT | 0.50 | 6.16% |
| SELL | retest2 | 2024-02-05 13:00:00 | 2436.10 | 2024-02-09 15:15:00 | 2305.10 | STOP_HIT | 0.50 | 5.38% |
| BUY | retest2 | 2024-02-15 15:15:00 | 2345.75 | 2024-02-19 09:15:00 | 2318.85 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-02-16 14:45:00 | 2341.65 | 2024-02-19 09:15:00 | 2318.85 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2024-03-20 09:15:00 | 2293.90 | 2024-03-20 09:15:00 | 2268.75 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-03-20 13:15:00 | 2286.05 | 2024-03-22 13:15:00 | 2267.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-03-21 09:45:00 | 2280.10 | 2024-03-22 13:15:00 | 2267.80 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-03-22 10:00:00 | 2281.20 | 2024-03-22 13:15:00 | 2267.80 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-03-28 13:45:00 | 2311.35 | 2024-04-12 12:15:00 | 2402.45 | STOP_HIT | 1.00 | 3.94% |
| BUY | retest2 | 2024-04-02 10:00:00 | 2317.85 | 2024-04-12 12:15:00 | 2402.45 | STOP_HIT | 1.00 | 3.65% |
| SELL | retest2 | 2024-04-18 13:15:00 | 2360.95 | 2024-04-22 13:15:00 | 2370.35 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-05-10 14:15:00 | 2462.40 | 2024-05-13 09:15:00 | 2439.80 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-05-13 13:30:00 | 2463.55 | 2024-05-18 09:15:00 | 2709.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-12 09:15:00 | 3236.90 | 2024-06-12 14:15:00 | 3205.95 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-06-20 10:15:00 | 3309.00 | 2024-06-21 11:15:00 | 3267.60 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-06-20 13:15:00 | 3310.95 | 2024-06-21 11:15:00 | 3267.60 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-06-20 14:15:00 | 3308.85 | 2024-06-21 11:15:00 | 3267.60 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-07-04 13:30:00 | 3144.00 | 2024-07-05 10:15:00 | 3190.80 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-07-12 11:15:00 | 3138.95 | 2024-07-18 14:15:00 | 3173.80 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-07-15 09:30:00 | 3142.40 | 2024-07-18 14:15:00 | 3173.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-07-15 10:30:00 | 3139.75 | 2024-07-18 14:15:00 | 3173.80 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-07-15 12:00:00 | 3141.20 | 2024-07-18 14:15:00 | 3173.80 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-07-31 09:15:00 | 3303.10 | 2024-08-05 09:15:00 | 3208.30 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-07-31 10:30:00 | 3297.05 | 2024-08-05 09:15:00 | 3208.30 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-07-31 11:00:00 | 3317.05 | 2024-08-05 09:15:00 | 3208.30 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2024-08-07 13:15:00 | 3190.00 | 2024-08-09 15:15:00 | 3030.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 09:15:00 | 3184.90 | 2024-08-09 15:15:00 | 3025.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 11:15:00 | 3189.85 | 2024-08-09 15:15:00 | 3030.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-07 13:15:00 | 3190.00 | 2024-08-12 09:15:00 | 2871.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-08 09:15:00 | 3184.90 | 2024-08-12 09:15:00 | 2866.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-08 11:15:00 | 3189.85 | 2024-08-12 09:15:00 | 2870.86 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-09-06 11:30:00 | 2981.20 | 2024-09-16 13:15:00 | 3065.85 | STOP_HIT | 1.00 | 2.84% |
| BUY | retest2 | 2024-09-06 12:15:00 | 2980.00 | 2024-09-16 13:15:00 | 3065.85 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2024-09-06 13:45:00 | 2979.70 | 2024-09-16 13:15:00 | 3065.85 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2024-09-09 10:30:00 | 2979.95 | 2024-09-16 13:15:00 | 3065.85 | STOP_HIT | 1.00 | 2.88% |
| SELL | retest2 | 2024-09-18 09:15:00 | 3066.90 | 2024-09-19 14:15:00 | 3087.95 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-09-19 12:45:00 | 3066.95 | 2024-09-19 14:15:00 | 3087.95 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-09-23 12:45:00 | 3090.50 | 2024-09-25 11:15:00 | 3042.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-09-23 13:30:00 | 3095.20 | 2024-09-25 11:15:00 | 3042.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-10-11 12:30:00 | 3019.15 | 2024-10-14 13:15:00 | 2999.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-10-11 14:45:00 | 3017.00 | 2024-10-14 13:15:00 | 2999.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-10-11 15:15:00 | 3039.00 | 2024-10-14 13:15:00 | 2999.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-10-15 13:00:00 | 2991.40 | 2024-10-15 14:15:00 | 3030.45 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-10-16 10:00:00 | 2991.45 | 2024-10-16 10:15:00 | 3020.65 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-10-16 12:00:00 | 2987.90 | 2024-10-16 13:15:00 | 3020.35 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-16 12:45:00 | 2991.10 | 2024-10-16 13:15:00 | 3020.35 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-10-25 10:00:00 | 2909.05 | 2024-10-28 11:15:00 | 2975.50 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-10-25 10:30:00 | 2908.25 | 2024-10-28 11:15:00 | 2975.50 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-10-25 11:30:00 | 2905.00 | 2024-10-28 11:15:00 | 2975.50 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-10-25 12:15:00 | 2903.25 | 2024-10-28 11:15:00 | 2975.50 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-10-31 09:15:00 | 2856.05 | 2024-11-06 12:15:00 | 2871.15 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-11-26 09:15:00 | 2754.60 | 2024-11-27 09:15:00 | 2732.40 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-11-26 09:45:00 | 2755.80 | 2024-11-27 09:15:00 | 2732.40 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-11 15:15:00 | 2816.00 | 2024-12-13 12:15:00 | 2848.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-12-12 15:00:00 | 2805.60 | 2024-12-13 12:15:00 | 2848.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-12-20 11:15:00 | 2791.10 | 2024-12-23 10:15:00 | 2824.35 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-12-20 15:00:00 | 2779.95 | 2024-12-23 10:15:00 | 2824.35 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-12-23 09:15:00 | 2787.95 | 2024-12-23 10:15:00 | 2824.35 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-12-31 12:30:00 | 2875.10 | 2025-01-03 10:15:00 | 2847.65 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-01-01 10:30:00 | 2876.00 | 2025-01-03 10:15:00 | 2847.65 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-01-02 10:30:00 | 2873.45 | 2025-01-03 10:15:00 | 2847.65 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-01-02 11:30:00 | 2870.50 | 2025-01-03 10:15:00 | 2847.65 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-01-14 13:30:00 | 2695.75 | 2025-01-14 15:15:00 | 2721.65 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-02-13 13:15:00 | 2651.20 | 2025-02-17 14:15:00 | 2672.95 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-02-19 11:45:00 | 2655.90 | 2025-02-24 13:15:00 | 2659.00 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-02-19 12:45:00 | 2654.95 | 2025-02-24 13:15:00 | 2659.00 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-02-27 11:45:00 | 2671.05 | 2025-03-04 09:15:00 | 2537.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 14:30:00 | 2664.25 | 2025-03-04 09:15:00 | 2531.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 15:15:00 | 2671.90 | 2025-03-04 09:15:00 | 2538.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 11:45:00 | 2671.05 | 2025-03-05 09:15:00 | 2578.05 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-02-27 14:30:00 | 2664.25 | 2025-03-05 09:15:00 | 2578.05 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2025-02-27 15:15:00 | 2671.90 | 2025-03-05 09:15:00 | 2578.05 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-03-13 11:30:00 | 2511.55 | 2025-03-18 09:15:00 | 2543.10 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-03-13 12:15:00 | 2517.50 | 2025-03-18 09:15:00 | 2543.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-03-13 14:00:00 | 2515.00 | 2025-03-18 09:15:00 | 2543.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-03-17 10:30:00 | 2515.00 | 2025-03-18 09:15:00 | 2543.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-04-01 10:45:00 | 2540.65 | 2025-04-02 15:15:00 | 2575.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-04-02 09:45:00 | 2539.85 | 2025-04-02 15:15:00 | 2575.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-04-21 11:15:00 | 2513.10 | 2025-04-25 11:15:00 | 2517.60 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-04-21 12:00:00 | 2517.00 | 2025-04-25 11:15:00 | 2517.60 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-04-21 13:45:00 | 2517.60 | 2025-04-25 11:15:00 | 2517.60 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-04-25 10:30:00 | 2514.60 | 2025-04-25 11:15:00 | 2517.60 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-05-07 11:00:00 | 2755.70 | 2025-05-13 10:15:00 | 2726.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-05-07 12:45:00 | 2756.20 | 2025-05-13 10:15:00 | 2726.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-05-07 14:00:00 | 2755.60 | 2025-05-13 10:15:00 | 2726.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-09 11:45:00 | 2755.70 | 2025-05-13 10:15:00 | 2726.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-05-12 09:15:00 | 2772.10 | 2025-05-13 10:15:00 | 2726.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-12 15:15:00 | 2755.00 | 2025-05-13 10:15:00 | 2726.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-22 12:15:00 | 2634.00 | 2025-05-26 09:15:00 | 2502.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-22 13:30:00 | 2632.00 | 2025-05-26 09:15:00 | 2500.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 11:15:00 | 2641.10 | 2025-05-26 09:15:00 | 2509.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 12:00:00 | 2641.70 | 2025-05-26 09:15:00 | 2509.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-22 12:15:00 | 2634.00 | 2025-05-28 13:15:00 | 2476.30 | STOP_HIT | 0.50 | 5.99% |
| SELL | retest2 | 2025-05-22 13:30:00 | 2632.00 | 2025-05-28 13:15:00 | 2476.30 | STOP_HIT | 0.50 | 5.92% |
| SELL | retest2 | 2025-05-23 11:15:00 | 2641.10 | 2025-05-28 13:15:00 | 2476.30 | STOP_HIT | 0.50 | 6.24% |
| SELL | retest2 | 2025-05-23 12:00:00 | 2641.70 | 2025-05-28 13:15:00 | 2476.30 | STOP_HIT | 0.50 | 6.26% |
| SELL | retest2 | 2025-05-26 09:15:00 | 2397.00 | 2025-06-05 09:15:00 | 2471.70 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest1 | 2025-06-16 09:15:00 | 2440.00 | 2025-06-17 10:15:00 | 2469.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-06-18 09:30:00 | 2442.80 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-06-18 10:15:00 | 2440.30 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-06-18 12:15:00 | 2442.00 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-06-19 12:45:00 | 2437.60 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-06-24 13:15:00 | 2400.00 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-25 09:30:00 | 2400.50 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-25 10:45:00 | 2402.90 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-01 10:00:00 | 2458.00 | 2025-07-17 09:15:00 | 2703.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-11 10:30:00 | 2396.20 | 2025-08-18 10:15:00 | 2440.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-08-11 11:15:00 | 2400.10 | 2025-08-18 10:15:00 | 2440.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-08-11 12:00:00 | 2400.10 | 2025-08-18 10:15:00 | 2440.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-08-12 09:15:00 | 2390.40 | 2025-08-18 10:15:00 | 2440.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-08-22 09:45:00 | 2401.10 | 2025-08-29 09:15:00 | 2281.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 09:45:00 | 2401.10 | 2025-09-01 11:15:00 | 2291.90 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2025-09-08 11:15:00 | 2320.40 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-08 12:15:00 | 2323.40 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-08 13:15:00 | 2320.30 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-09-08 14:30:00 | 2322.70 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-09 10:30:00 | 2306.80 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-09-09 11:00:00 | 2308.50 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-09-09 12:30:00 | 2307.60 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-09-15 09:15:00 | 2406.00 | 2025-09-23 15:15:00 | 2489.00 | STOP_HIT | 1.00 | 3.45% |
| SELL | retest2 | 2025-10-06 11:15:00 | 2292.10 | 2025-10-08 11:15:00 | 2320.10 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-06 12:30:00 | 2289.40 | 2025-10-08 11:15:00 | 2320.10 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-10-07 11:30:00 | 2289.50 | 2025-10-08 11:15:00 | 2320.10 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-21 13:45:00 | 2301.50 | 2025-10-30 12:15:00 | 2322.10 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-10-31 14:15:00 | 2290.10 | 2025-11-03 09:15:00 | 2175.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 14:15:00 | 2290.10 | 2025-11-03 14:15:00 | 2290.40 | STOP_HIT | 0.50 | -0.01% |
| SELL | retest2 | 2025-11-03 15:00:00 | 2290.40 | 2025-11-04 09:15:00 | 2347.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-11-06 11:15:00 | 2337.20 | 2025-11-07 11:15:00 | 2315.60 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-20 10:30:00 | 2269.90 | 2025-11-21 09:15:00 | 2339.80 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-20 13:00:00 | 2269.10 | 2025-11-21 09:15:00 | 2339.80 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2026-01-01 14:15:00 | 2323.90 | 2026-01-01 15:15:00 | 2306.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-01-01 14:45:00 | 2324.90 | 2026-01-01 15:15:00 | 2306.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-01-02 09:15:00 | 2327.00 | 2026-01-09 09:15:00 | 2371.70 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2026-01-14 11:30:00 | 2421.00 | 2026-01-20 10:15:00 | 2407.10 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-14 12:30:00 | 2421.20 | 2026-01-20 10:15:00 | 2407.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-01-14 14:45:00 | 2432.90 | 2026-01-20 10:15:00 | 2407.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-01-16 10:30:00 | 2420.70 | 2026-01-20 10:15:00 | 2407.10 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-02-25 14:00:00 | 2489.10 | 2026-03-02 09:15:00 | 2364.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 2489.00 | 2026-03-02 09:15:00 | 2364.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:00:00 | 2489.10 | 2026-03-05 10:15:00 | 2240.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 2489.00 | 2026-03-05 10:15:00 | 2240.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 15:15:00 | 2050.00 | 2026-03-25 09:15:00 | 2143.50 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2026-03-27 13:15:00 | 2145.90 | 2026-03-30 09:15:00 | 2074.60 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest1 | 2026-04-10 09:15:00 | 2274.40 | 2026-04-13 09:15:00 | 2228.30 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-04-13 11:00:00 | 2258.40 | 2026-04-20 15:15:00 | 2279.80 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2026-04-13 12:00:00 | 2249.50 | 2026-04-20 15:15:00 | 2279.80 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2026-04-13 14:15:00 | 2249.90 | 2026-04-20 15:15:00 | 2279.80 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2026-04-15 09:15:00 | 2257.30 | 2026-04-20 15:15:00 | 2279.80 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2026-04-28 12:00:00 | 2195.00 | 2026-05-05 09:15:00 | 2204.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-04-30 09:15:00 | 2175.90 | 2026-05-05 09:15:00 | 2204.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-05-04 11:30:00 | 2195.20 | 2026-05-05 09:15:00 | 2204.80 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-05-04 12:30:00 | 2187.80 | 2026-05-05 09:15:00 | 2204.80 | STOP_HIT | 1.00 | -0.78% |
