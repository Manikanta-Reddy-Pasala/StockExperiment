# ASIANPAINT (ASIANPAINT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2600.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 24 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 51 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 37
- **Target hits / Stop hits / Partials:** 1 / 50 / 2
- **Avg / median % per leg:** -0.04% / -0.71%
- **Sum % (uncompounded):** -2.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 7 | 29.2% | 1 | 23 | 0 | 0.15% | 3.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 7 | 29.2% | 1 | 23 | 0 | 0.15% | 3.5% |
| SELL (all) | 29 | 9 | 31.0% | 0 | 27 | 2 | -0.19% | -5.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 9 | 31.0% | 0 | 27 | 2 | -0.19% | -5.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 16 | 30.2% | 1 | 50 | 2 | -0.04% | -2.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 2364.10 | 2345.37 | 2344.24 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 10:15:00 | 2334.10 | 2343.22 | 2343.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 09:15:00 | 2292.90 | 2322.83 | 2332.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 10:15:00 | 2293.00 | 2291.26 | 2307.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 2293.00 | 2291.26 | 2307.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 2314.50 | 2296.84 | 2307.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 2314.50 | 2296.84 | 2307.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 2321.40 | 2301.76 | 2308.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 2319.10 | 2301.76 | 2308.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 2315.90 | 2312.64 | 2312.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 2346.80 | 2322.30 | 2317.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 2341.30 | 2342.05 | 2331.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:00:00 | 2341.30 | 2342.05 | 2331.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 2331.20 | 2339.38 | 2332.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 2331.20 | 2339.38 | 2332.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 2331.00 | 2337.71 | 2332.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 2338.00 | 2337.71 | 2332.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:00:00 | 2334.10 | 2336.99 | 2332.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 11:15:00 | 2320.80 | 2332.71 | 2331.51 | SL hit (close<static) qty=1.00 sl=2326.60 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 2311.00 | 2328.37 | 2329.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 2304.40 | 2323.58 | 2327.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 2311.30 | 2309.68 | 2315.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 2311.30 | 2309.68 | 2315.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 2311.30 | 2309.68 | 2315.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 2311.30 | 2309.68 | 2315.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2315.00 | 2302.55 | 2306.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 2315.00 | 2302.55 | 2306.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2322.80 | 2306.60 | 2308.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 2322.50 | 2306.60 | 2308.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 2322.10 | 2309.70 | 2309.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 2336.00 | 2318.08 | 2314.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 2308.60 | 2316.40 | 2313.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 11:15:00 | 2308.60 | 2316.40 | 2313.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 2308.60 | 2316.40 | 2313.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 2308.60 | 2316.40 | 2313.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 2307.90 | 2314.70 | 2313.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 2305.80 | 2314.70 | 2313.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2324.70 | 2321.65 | 2317.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 2324.90 | 2321.65 | 2317.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 2326.50 | 2330.88 | 2324.85 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 2304.20 | 2318.61 | 2320.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 2299.00 | 2307.92 | 2313.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 2268.60 | 2267.77 | 2279.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:45:00 | 2265.40 | 2267.77 | 2279.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 2264.90 | 2266.96 | 2275.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 2260.00 | 2264.72 | 2273.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 2244.10 | 2230.54 | 2230.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 2244.10 | 2230.54 | 2230.04 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 2215.20 | 2227.29 | 2228.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 2199.90 | 2220.71 | 2225.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 2213.00 | 2212.09 | 2218.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 2214.10 | 2212.09 | 2218.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 2224.70 | 2215.33 | 2218.80 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 2237.20 | 2224.03 | 2222.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 2244.30 | 2228.08 | 2224.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 2272.40 | 2274.49 | 2265.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 2272.40 | 2274.49 | 2265.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 2264.00 | 2271.76 | 2266.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 2278.50 | 2271.76 | 2266.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 2273.50 | 2272.11 | 2266.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 2285.90 | 2276.68 | 2271.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 2262.60 | 2269.43 | 2269.42 | SL hit (close<static) qty=1.00 sl=2263.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 2264.30 | 2268.40 | 2268.95 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 2295.00 | 2272.58 | 2270.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 2317.60 | 2291.90 | 2285.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 2334.90 | 2336.50 | 2318.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 2334.90 | 2336.50 | 2318.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2353.90 | 2358.49 | 2345.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:45:00 | 2370.30 | 2360.55 | 2347.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 15:15:00 | 2447.20 | 2469.58 | 2470.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 2447.20 | 2469.58 | 2470.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 2444.90 | 2459.46 | 2465.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 11:15:00 | 2418.60 | 2396.67 | 2409.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 2418.60 | 2396.67 | 2409.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 2418.60 | 2396.67 | 2409.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 2418.60 | 2396.67 | 2409.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 2411.70 | 2399.67 | 2409.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 2415.80 | 2399.67 | 2409.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 2411.20 | 2404.73 | 2410.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 2416.90 | 2404.73 | 2410.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2403.00 | 2404.39 | 2409.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 2406.00 | 2404.39 | 2409.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 2398.20 | 2403.15 | 2408.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 2395.30 | 2402.15 | 2406.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 15:15:00 | 2359.50 | 2352.22 | 2352.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 15:15:00 | 2359.50 | 2352.22 | 2352.02 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 2349.20 | 2351.62 | 2351.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 2342.30 | 2349.75 | 2350.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 2350.50 | 2349.51 | 2350.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 2350.50 | 2349.51 | 2350.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2350.50 | 2349.51 | 2350.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 2346.60 | 2349.51 | 2350.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 2398.10 | 2359.23 | 2354.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 2406.10 | 2375.09 | 2363.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 10:15:00 | 2397.40 | 2406.61 | 2392.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 10:45:00 | 2399.00 | 2406.61 | 2392.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 2386.50 | 2402.59 | 2391.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:00:00 | 2386.50 | 2402.59 | 2391.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 2403.90 | 2402.85 | 2393.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 2406.00 | 2402.85 | 2393.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 2411.20 | 2399.34 | 2393.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 2479.00 | 2485.44 | 2485.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 2479.00 | 2485.44 | 2485.70 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 2490.00 | 2486.35 | 2486.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 2500.00 | 2491.94 | 2489.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 2575.40 | 2576.51 | 2561.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 2562.40 | 2570.36 | 2565.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 2562.40 | 2570.36 | 2565.07 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 2538.50 | 2562.22 | 2563.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 2521.20 | 2554.01 | 2559.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 2505.10 | 2496.57 | 2511.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 12:00:00 | 2505.10 | 2496.57 | 2511.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 2510.50 | 2499.36 | 2511.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:45:00 | 2519.10 | 2499.36 | 2511.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 2511.70 | 2501.83 | 2511.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 2511.70 | 2501.83 | 2511.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 2473.80 | 2496.22 | 2508.11 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 2521.30 | 2507.64 | 2506.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 15:15:00 | 2523.50 | 2511.76 | 2508.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 11:15:00 | 2554.10 | 2557.75 | 2543.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:00:00 | 2554.10 | 2557.75 | 2543.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 2537.90 | 2553.82 | 2543.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 2537.90 | 2553.82 | 2543.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 2541.50 | 2551.36 | 2543.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 2538.50 | 2551.36 | 2543.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 2538.00 | 2548.68 | 2543.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 2544.60 | 2548.68 | 2543.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 2550.60 | 2549.74 | 2544.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 2550.60 | 2549.74 | 2544.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 2553.00 | 2551.70 | 2546.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 2553.00 | 2551.70 | 2546.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 2560.00 | 2553.61 | 2548.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 2566.00 | 2557.43 | 2550.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 2566.00 | 2558.54 | 2551.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:00:00 | 2568.00 | 2572.98 | 2568.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 13:15:00 | 2542.00 | 2563.23 | 2564.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 2542.00 | 2563.23 | 2564.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 2530.20 | 2556.62 | 2561.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 2541.30 | 2540.58 | 2548.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 15:15:00 | 2541.30 | 2540.58 | 2548.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 2541.30 | 2540.58 | 2548.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 2546.00 | 2540.58 | 2548.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2531.50 | 2538.76 | 2547.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 2527.00 | 2538.76 | 2547.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:00:00 | 2530.00 | 2536.50 | 2544.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 2553.60 | 2546.60 | 2545.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 2553.60 | 2546.60 | 2545.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 2556.00 | 2548.48 | 2546.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 2539.20 | 2550.76 | 2549.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 12:15:00 | 2539.20 | 2550.76 | 2549.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 2539.20 | 2550.76 | 2549.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 2539.20 | 2550.76 | 2549.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 2551.80 | 2550.97 | 2549.32 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 2540.50 | 2548.27 | 2548.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 2514.80 | 2541.58 | 2545.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 2492.80 | 2489.21 | 2499.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 15:00:00 | 2492.80 | 2489.21 | 2499.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2495.10 | 2491.24 | 2498.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:15:00 | 2491.00 | 2492.60 | 2498.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 2507.00 | 2496.32 | 2496.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 2507.00 | 2496.32 | 2496.05 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 2477.90 | 2494.11 | 2495.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 2434.40 | 2477.67 | 2486.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 2460.80 | 2458.38 | 2470.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 2460.80 | 2458.38 | 2470.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 2469.20 | 2460.55 | 2470.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 2469.20 | 2460.55 | 2470.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 2470.70 | 2462.58 | 2470.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 2470.70 | 2462.58 | 2470.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 2467.00 | 2463.46 | 2469.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:15:00 | 2459.20 | 2464.29 | 2469.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2336.24 | 2363.74 | 2393.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 2350.10 | 2349.94 | 2363.16 | SL hit (close>ema200) qty=0.50 sl=2349.94 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 2354.40 | 2349.59 | 2349.16 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 2338.10 | 2347.29 | 2348.16 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 2353.90 | 2349.52 | 2349.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 2369.90 | 2356.50 | 2352.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 2348.80 | 2356.91 | 2353.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 2348.80 | 2356.91 | 2353.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 2348.80 | 2356.91 | 2353.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 2348.80 | 2356.91 | 2353.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 2354.70 | 2356.47 | 2353.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 2359.20 | 2356.47 | 2353.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 2342.40 | 2353.66 | 2352.83 | SL hit (close<static) qty=1.00 sl=2348.80 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 2337.10 | 2350.35 | 2351.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 12:15:00 | 2332.10 | 2344.40 | 2348.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 2334.10 | 2332.47 | 2338.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 13:15:00 | 2334.10 | 2332.47 | 2338.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 2334.10 | 2332.47 | 2338.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:45:00 | 2334.50 | 2332.47 | 2338.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 2340.00 | 2334.46 | 2338.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 2353.00 | 2334.46 | 2338.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 2348.20 | 2337.21 | 2339.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 2349.30 | 2337.21 | 2339.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 2348.50 | 2339.47 | 2339.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 2348.50 | 2339.47 | 2339.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 2347.40 | 2341.05 | 2340.60 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 2326.50 | 2339.61 | 2341.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 2321.00 | 2335.89 | 2339.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 2353.50 | 2331.30 | 2334.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 2353.50 | 2331.30 | 2334.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2353.50 | 2331.30 | 2334.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 2347.60 | 2331.30 | 2334.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 2366.50 | 2338.34 | 2337.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 2379.00 | 2355.65 | 2346.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 13:15:00 | 2509.00 | 2510.86 | 2481.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 2506.90 | 2510.86 | 2481.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2498.70 | 2516.18 | 2499.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 2498.70 | 2516.18 | 2499.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 2499.70 | 2512.89 | 2499.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 2501.90 | 2512.89 | 2499.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 2494.60 | 2509.23 | 2499.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:15:00 | 2488.30 | 2509.23 | 2499.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 2490.70 | 2505.52 | 2498.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 2489.40 | 2505.52 | 2498.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 2497.00 | 2503.82 | 2498.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 2499.90 | 2503.82 | 2498.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 2499.40 | 2501.83 | 2498.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 2500.90 | 2501.64 | 2498.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 2506.30 | 2509.52 | 2507.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 2506.90 | 2508.99 | 2507.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 2501.20 | 2506.63 | 2506.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 2501.20 | 2506.63 | 2506.82 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 2524.60 | 2510.22 | 2508.44 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 2509.50 | 2519.87 | 2520.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 2499.10 | 2513.83 | 2517.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 2514.50 | 2508.80 | 2513.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 2514.50 | 2508.80 | 2513.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 2514.50 | 2508.80 | 2513.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 2514.50 | 2508.80 | 2513.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 2506.50 | 2508.34 | 2512.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 2503.10 | 2508.34 | 2512.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2509.90 | 2508.65 | 2512.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:15:00 | 2494.10 | 2507.40 | 2510.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 2618.00 | 2521.76 | 2515.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 2618.00 | 2521.76 | 2515.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 2652.50 | 2611.13 | 2584.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 2639.00 | 2641.78 | 2617.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:30:00 | 2638.30 | 2641.78 | 2617.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 2789.10 | 2684.67 | 2656.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 2660.50 | 2684.67 | 2656.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2890.00 | 2891.05 | 2865.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 2904.30 | 2896.44 | 2872.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:00:00 | 2906.20 | 2901.23 | 2880.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 2909.80 | 2900.74 | 2884.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 2902.50 | 2901.17 | 2885.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2892.00 | 2899.54 | 2891.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2891.40 | 2899.54 | 2891.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2874.90 | 2894.61 | 2889.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 2874.90 | 2894.61 | 2889.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 2872.60 | 2890.21 | 2888.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 2870.00 | 2890.21 | 2888.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 2858.80 | 2883.93 | 2885.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 2858.80 | 2883.93 | 2885.46 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 09:15:00 | 2894.50 | 2878.09 | 2877.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 09:15:00 | 2898.50 | 2884.37 | 2880.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 12:15:00 | 2886.30 | 2886.46 | 2882.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 13:00:00 | 2886.30 | 2886.46 | 2882.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 2888.90 | 2886.94 | 2883.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 2885.10 | 2886.94 | 2883.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2876.90 | 2884.94 | 2882.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 2876.90 | 2884.94 | 2882.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 2877.20 | 2883.39 | 2882.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 2886.10 | 2883.39 | 2882.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 13:00:00 | 2879.20 | 2883.64 | 2882.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 2864.40 | 2879.79 | 2881.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 13:15:00 | 2864.40 | 2879.79 | 2881.26 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 2899.70 | 2881.76 | 2881.61 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 2873.00 | 2881.08 | 2881.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 2856.20 | 2871.63 | 2875.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 2867.90 | 2864.22 | 2869.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 2867.90 | 2864.22 | 2869.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 2861.50 | 2863.67 | 2869.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 2907.70 | 2863.67 | 2869.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 2890.00 | 2868.94 | 2870.99 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 2904.90 | 2876.13 | 2874.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 12:15:00 | 2919.00 | 2889.80 | 2880.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 12:15:00 | 2952.90 | 2956.90 | 2937.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 13:00:00 | 2952.90 | 2956.90 | 2937.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 2950.00 | 2953.36 | 2944.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 2959.90 | 2953.36 | 2944.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 2935.40 | 2952.64 | 2948.70 | SL hit (close<static) qty=1.00 sl=2941.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 2920.80 | 2943.06 | 2944.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 2811.30 | 2911.62 | 2929.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 2782.30 | 2770.84 | 2787.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 2782.30 | 2770.84 | 2787.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2782.30 | 2770.84 | 2787.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 2781.40 | 2770.84 | 2787.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2793.90 | 2775.45 | 2788.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 2793.90 | 2775.45 | 2788.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 2796.80 | 2779.72 | 2788.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 2794.10 | 2779.72 | 2788.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 2782.30 | 2780.24 | 2788.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 14:15:00 | 2777.00 | 2780.43 | 2787.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 2777.80 | 2780.72 | 2787.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 2804.60 | 2785.63 | 2787.80 | SL hit (close>static) qty=1.00 sl=2802.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 2801.10 | 2790.48 | 2789.49 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 2778.90 | 2788.10 | 2788.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 2770.30 | 2782.50 | 2785.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 2781.40 | 2770.49 | 2776.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 2781.40 | 2770.49 | 2776.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2781.40 | 2770.49 | 2776.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 2781.40 | 2770.49 | 2776.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2784.90 | 2773.37 | 2777.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 2788.90 | 2773.37 | 2777.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 2788.00 | 2780.24 | 2779.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2818.50 | 2793.01 | 2786.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 2792.40 | 2795.28 | 2789.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:30:00 | 2792.60 | 2795.28 | 2789.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2796.10 | 2799.34 | 2793.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 2786.00 | 2799.34 | 2793.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2789.90 | 2797.45 | 2793.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 2788.50 | 2797.45 | 2793.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2802.40 | 2798.44 | 2794.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 12:30:00 | 2804.60 | 2799.71 | 2795.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 14:15:00 | 2787.60 | 2797.46 | 2797.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 2787.60 | 2797.46 | 2797.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 2762.70 | 2788.67 | 2793.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 12:15:00 | 2760.00 | 2754.72 | 2767.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 13:00:00 | 2760.00 | 2754.72 | 2767.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2787.90 | 2761.36 | 2769.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 2787.90 | 2761.36 | 2769.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2774.50 | 2763.98 | 2769.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 2764.10 | 2766.19 | 2770.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 2788.30 | 2773.99 | 2773.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 11:15:00 | 2788.30 | 2773.99 | 2773.06 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 2758.50 | 2770.73 | 2771.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 15:15:00 | 2753.50 | 2765.18 | 2768.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 2781.90 | 2768.53 | 2770.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 2781.90 | 2768.53 | 2770.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2781.90 | 2768.53 | 2770.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 2780.80 | 2768.53 | 2770.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2775.00 | 2769.82 | 2770.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 2773.60 | 2769.82 | 2770.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 2778.40 | 2771.54 | 2771.25 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 2749.70 | 2768.67 | 2770.89 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 2802.10 | 2776.17 | 2773.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 2823.20 | 2791.31 | 2780.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 2821.80 | 2833.25 | 2819.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 12:00:00 | 2821.80 | 2833.25 | 2819.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 2813.80 | 2829.36 | 2819.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 2814.70 | 2829.36 | 2819.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 2812.20 | 2825.93 | 2818.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:45:00 | 2809.20 | 2825.93 | 2818.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2780.80 | 2811.93 | 2813.51 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 2824.80 | 2808.33 | 2806.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 12:15:00 | 2833.60 | 2813.38 | 2809.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 2840.20 | 2870.67 | 2859.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 2840.20 | 2870.67 | 2859.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2840.20 | 2870.67 | 2859.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 2840.20 | 2870.67 | 2859.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2838.60 | 2864.26 | 2857.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 2843.40 | 2864.26 | 2857.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 13:15:00 | 2820.10 | 2847.43 | 2851.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 14:15:00 | 2810.90 | 2840.12 | 2847.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2706.50 | 2680.27 | 2705.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2706.50 | 2680.27 | 2705.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2706.50 | 2680.27 | 2705.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 2724.00 | 2680.27 | 2705.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2696.40 | 2683.49 | 2704.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 2691.90 | 2683.49 | 2704.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 2691.20 | 2683.91 | 2701.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2737.90 | 2702.25 | 2705.13 | SL hit (close>static) qty=1.00 sl=2710.30 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 2782.00 | 2718.20 | 2712.12 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 2680.00 | 2712.11 | 2714.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 2629.70 | 2686.33 | 2701.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 2438.40 | 2436.20 | 2481.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 14:00:00 | 2438.40 | 2436.20 | 2481.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2407.50 | 2392.30 | 2426.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 2417.10 | 2392.30 | 2426.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2417.10 | 2401.29 | 2415.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2402.70 | 2401.29 | 2415.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 2445.70 | 2423.29 | 2420.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 2445.70 | 2423.29 | 2420.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 2460.60 | 2440.16 | 2430.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2421.50 | 2439.85 | 2433.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 2421.50 | 2439.85 | 2433.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 2421.50 | 2439.85 | 2433.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 2418.00 | 2439.85 | 2433.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 2409.70 | 2433.82 | 2431.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 2409.70 | 2433.82 | 2431.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 2433.50 | 2430.70 | 2429.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 2404.50 | 2430.70 | 2429.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2401.00 | 2424.76 | 2427.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 2390.50 | 2410.42 | 2419.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2417.00 | 2408.01 | 2415.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 2417.00 | 2408.01 | 2415.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2417.00 | 2408.01 | 2415.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 2417.00 | 2408.01 | 2415.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 2417.70 | 2409.95 | 2415.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 2422.60 | 2409.95 | 2415.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 2416.50 | 2411.51 | 2414.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 2416.50 | 2411.51 | 2414.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 2417.00 | 2412.61 | 2414.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 2411.00 | 2412.61 | 2414.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2383.90 | 2396.46 | 2403.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 2380.40 | 2394.28 | 2402.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 2379.70 | 2387.77 | 2394.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 2410.40 | 2395.86 | 2396.55 | SL hit (close>static) qty=1.00 sl=2405.70 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 2410.90 | 2398.87 | 2397.86 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 2384.60 | 2395.74 | 2396.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 2365.00 | 2387.95 | 2393.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 2380.90 | 2379.70 | 2386.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 2380.90 | 2379.70 | 2386.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 2390.40 | 2381.84 | 2387.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 2390.40 | 2381.84 | 2387.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 2397.60 | 2384.99 | 2388.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 2396.20 | 2384.99 | 2388.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 2397.50 | 2387.49 | 2388.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 2397.50 | 2387.49 | 2388.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 2400.00 | 2390.00 | 2389.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 2433.60 | 2398.72 | 2393.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 11:15:00 | 2419.10 | 2425.90 | 2415.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 12:00:00 | 2419.10 | 2425.90 | 2415.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2407.20 | 2425.02 | 2419.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 2403.30 | 2425.02 | 2419.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 2400.00 | 2420.01 | 2417.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 2400.00 | 2420.01 | 2417.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 2409.10 | 2415.21 | 2415.79 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 2421.80 | 2414.00 | 2413.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 2436.90 | 2427.85 | 2421.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 2424.10 | 2427.10 | 2422.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 14:00:00 | 2424.10 | 2427.10 | 2422.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 2425.60 | 2426.80 | 2422.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 2423.50 | 2426.80 | 2422.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 2433.30 | 2429.06 | 2424.31 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 2412.70 | 2421.87 | 2421.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 10:15:00 | 2398.20 | 2410.32 | 2414.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 2388.10 | 2384.01 | 2394.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 14:00:00 | 2388.10 | 2384.01 | 2394.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 2375.90 | 2382.39 | 2392.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 2394.00 | 2382.39 | 2392.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2284.80 | 2273.61 | 2293.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 2297.10 | 2273.61 | 2293.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2286.00 | 2277.96 | 2290.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 2286.00 | 2277.96 | 2290.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 2304.60 | 2283.29 | 2292.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 2304.60 | 2283.29 | 2292.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 2300.00 | 2286.63 | 2292.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 2295.10 | 2286.46 | 2292.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 2180.34 | 2261.00 | 2278.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 2255.80 | 2233.45 | 2251.41 | SL hit (close>ema200) qty=0.50 sl=2233.45 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 2279.00 | 2258.79 | 2258.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 2295.00 | 2266.03 | 2261.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 2268.70 | 2269.09 | 2264.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 2268.70 | 2269.09 | 2264.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 2262.10 | 2267.69 | 2263.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 2265.80 | 2267.69 | 2263.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 2244.80 | 2263.11 | 2262.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 2244.80 | 2263.11 | 2262.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 2242.40 | 2258.97 | 2260.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 2230.90 | 2253.36 | 2257.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 2248.20 | 2244.55 | 2251.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 11:00:00 | 2248.20 | 2244.55 | 2251.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 2221.80 | 2203.28 | 2212.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 2225.60 | 2203.28 | 2212.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 2215.00 | 2205.62 | 2212.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 2232.90 | 2205.62 | 2212.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2253.60 | 2215.22 | 2216.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 2253.60 | 2215.22 | 2216.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 2236.90 | 2219.56 | 2218.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 2263.90 | 2243.86 | 2234.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2202.60 | 2238.19 | 2233.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2202.60 | 2238.19 | 2233.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2202.60 | 2238.19 | 2233.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 2202.20 | 2238.19 | 2233.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 2195.70 | 2229.69 | 2230.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 2186.00 | 2207.48 | 2218.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2210.90 | 2206.15 | 2215.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2210.90 | 2206.15 | 2215.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2210.90 | 2206.15 | 2215.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:00:00 | 2199.80 | 2206.55 | 2213.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 2159.70 | 2204.29 | 2210.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:45:00 | 2203.50 | 2167.26 | 2174.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 2220.30 | 2177.87 | 2178.50 | SL hit (close>static) qty=1.00 sl=2219.60 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 2219.20 | 2186.13 | 2182.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2248.40 | 2203.96 | 2191.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2250.50 | 2255.43 | 2230.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 2230.20 | 2246.54 | 2232.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 2230.20 | 2246.54 | 2232.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 2230.20 | 2246.54 | 2232.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 2213.90 | 2240.01 | 2230.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 2213.90 | 2240.01 | 2230.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 2208.60 | 2233.73 | 2228.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 2208.60 | 2233.73 | 2228.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 2181.80 | 2219.42 | 2222.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 2170.50 | 2209.64 | 2217.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2231.20 | 2191.14 | 2202.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2231.20 | 2191.14 | 2202.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2231.20 | 2191.14 | 2202.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 2231.20 | 2191.14 | 2202.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 2211.90 | 2195.29 | 2203.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 2207.00 | 2200.27 | 2204.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 2240.00 | 2208.22 | 2207.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 2240.00 | 2208.22 | 2207.82 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 2170.00 | 2207.69 | 2208.63 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 2197.00 | 2181.58 | 2181.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2280.80 | 2201.43 | 2190.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 2265.00 | 2266.39 | 2243.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 2265.00 | 2266.39 | 2243.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2328.10 | 2338.90 | 2305.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2343.50 | 2338.90 | 2305.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 2577.85 | 2512.77 | 2479.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 2509.40 | 2528.21 | 2528.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 2484.30 | 2519.42 | 2524.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 2507.20 | 2497.27 | 2508.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 2507.20 | 2497.27 | 2508.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 2507.20 | 2497.27 | 2508.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 2518.00 | 2497.27 | 2508.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 2496.60 | 2497.14 | 2507.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 2491.40 | 2497.14 | 2507.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 2482.00 | 2494.11 | 2504.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:45:00 | 2491.40 | 2493.40 | 2502.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:00:00 | 2487.70 | 2492.26 | 2501.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 2488.00 | 2487.30 | 2496.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:30:00 | 2494.20 | 2487.30 | 2496.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 2477.00 | 2482.49 | 2491.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:45:00 | 2484.30 | 2482.49 | 2491.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 2454.60 | 2465.03 | 2476.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 2449.20 | 2465.03 | 2476.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 2447.00 | 2461.52 | 2473.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:45:00 | 2448.90 | 2435.55 | 2450.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 2486.60 | 2447.88 | 2453.82 | SL hit (close>static) qty=1.00 sl=2477.90 alert=retest2 |

### Cycle 75 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 2458.00 | 2457.54 | 2457.51 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 14:15:00 | 2450.00 | 2456.03 | 2456.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 2435.90 | 2450.72 | 2454.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 2449.40 | 2437.89 | 2443.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 2449.40 | 2437.89 | 2443.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 2449.40 | 2437.89 | 2443.83 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 2478.90 | 2451.26 | 2449.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 2494.90 | 2463.76 | 2455.47 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 09:15:00 | 2338.00 | 2025-05-20 11:15:00 | 2320.80 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-05-20 10:00:00 | 2334.10 | 2025-05-20 11:15:00 | 2320.80 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-06-03 10:45:00 | 2260.00 | 2025-06-12 11:15:00 | 2244.10 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2025-06-20 15:00:00 | 2285.90 | 2025-06-23 12:15:00 | 2262.60 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-02 10:45:00 | 2370.30 | 2025-07-10 15:15:00 | 2447.20 | STOP_HIT | 1.00 | 3.24% |
| SELL | retest2 | 2025-07-18 09:15:00 | 2395.30 | 2025-07-28 15:15:00 | 2359.50 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2025-07-31 13:15:00 | 2406.00 | 2025-08-12 15:15:00 | 2479.00 | STOP_HIT | 1.00 | 3.03% |
| BUY | retest2 | 2025-08-01 09:15:00 | 2411.20 | 2025-08-12 15:15:00 | 2479.00 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2025-09-04 09:30:00 | 2566.00 | 2025-09-08 13:15:00 | 2542.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-04 11:15:00 | 2566.00 | 2025-09-08 13:15:00 | 2542.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-08 11:00:00 | 2568.00 | 2025-09-08 13:15:00 | 2542.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-10 10:15:00 | 2527.00 | 2025-09-11 13:15:00 | 2553.60 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-10 12:00:00 | 2530.00 | 2025-09-11 13:15:00 | 2553.60 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-18 12:15:00 | 2491.00 | 2025-09-19 12:15:00 | 2507.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-24 14:15:00 | 2459.20 | 2025-09-29 09:15:00 | 2336.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:15:00 | 2459.20 | 2025-09-30 14:15:00 | 2350.10 | STOP_HIT | 0.50 | 4.44% |
| BUY | retest2 | 2025-10-08 09:15:00 | 2359.20 | 2025-10-08 09:15:00 | 2342.40 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-24 12:15:00 | 2499.90 | 2025-10-28 15:15:00 | 2501.20 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-10-24 14:00:00 | 2499.40 | 2025-10-28 15:15:00 | 2501.20 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-10-24 15:00:00 | 2500.90 | 2025-10-28 15:15:00 | 2501.20 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-10-28 12:15:00 | 2506.30 | 2025-10-28 15:15:00 | 2501.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-04 13:15:00 | 2494.10 | 2025-11-06 09:15:00 | 2618.00 | STOP_HIT | 1.00 | -4.97% |
| BUY | retest2 | 2025-11-18 11:30:00 | 2904.30 | 2025-11-20 11:15:00 | 2858.80 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-18 15:00:00 | 2906.20 | 2025-11-20 11:15:00 | 2858.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-19 09:30:00 | 2909.80 | 2025-11-20 11:15:00 | 2858.80 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-11-19 10:30:00 | 2902.50 | 2025-11-20 11:15:00 | 2858.80 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-11-26 09:15:00 | 2886.10 | 2025-11-26 13:15:00 | 2864.40 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-11-26 13:00:00 | 2879.20 | 2025-11-26 13:15:00 | 2864.40 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-05 12:15:00 | 2959.90 | 2025-12-08 10:15:00 | 2935.40 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-08 10:30:00 | 2950.20 | 2025-12-08 11:15:00 | 2937.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-12-15 14:15:00 | 2777.00 | 2025-12-16 10:15:00 | 2804.60 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-15 15:15:00 | 2777.80 | 2025-12-16 10:15:00 | 2804.60 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-23 12:30:00 | 2804.60 | 2025-12-24 14:15:00 | 2787.60 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-12-30 09:15:00 | 2764.10 | 2025-12-30 11:15:00 | 2788.30 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-01-22 11:15:00 | 2691.90 | 2026-01-23 09:15:00 | 2737.90 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-01-22 12:30:00 | 2691.20 | 2026-01-23 09:15:00 | 2737.90 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-02-03 10:15:00 | 2402.70 | 2026-02-04 09:15:00 | 2445.70 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-02-11 11:15:00 | 2380.40 | 2026-02-12 13:15:00 | 2410.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-12 09:45:00 | 2379.70 | 2026-02-12 13:15:00 | 2410.40 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2295.10 | 2026-03-09 09:15:00 | 2180.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2295.10 | 2026-03-10 09:15:00 | 2255.80 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2026-03-20 14:00:00 | 2199.80 | 2026-03-24 13:15:00 | 2220.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2159.70 | 2026-03-24 13:15:00 | 2220.30 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-03-24 12:45:00 | 2203.50 | 2026-03-24 13:15:00 | 2220.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-04-01 11:30:00 | 2207.00 | 2026-04-01 12:15:00 | 2240.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2343.50 | 2026-04-21 09:15:00 | 2577.85 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 2491.40 | 2026-05-04 09:15:00 | 2486.60 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2026-04-27 12:00:00 | 2482.00 | 2026-05-04 09:15:00 | 2486.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-04-27 13:45:00 | 2491.40 | 2026-05-04 09:15:00 | 2486.60 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2026-04-27 15:00:00 | 2487.70 | 2026-05-04 13:15:00 | 2458.00 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2026-04-29 14:15:00 | 2449.20 | 2026-05-04 13:15:00 | 2458.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-04-29 14:45:00 | 2447.00 | 2026-05-04 13:15:00 | 2458.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-04-30 14:45:00 | 2448.90 | 2026-05-04 13:15:00 | 2458.00 | STOP_HIT | 1.00 | -0.37% |
