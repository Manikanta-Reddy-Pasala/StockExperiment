# Ajanta Pharmaceuticals Ltd. (AJANTPHARM)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 3033.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 146 |
| ALERT1 | 103 |
| ALERT2 | 103 |
| ALERT2_SKIP | 64 |
| ALERT3 | 253 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 147 |
| PARTIAL | 14 |
| TARGET_HIT | 11 |
| STOP_HIT | 139 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 164 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 62 / 102
- **Target hits / Stop hits / Partials:** 11 / 139 / 14
- **Avg / median % per leg:** 0.63% / -0.72%
- **Sum % (uncompounded):** 103.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 28 | 37.8% | 10 | 64 | 0 | 1.10% | 81.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.74% | -2.2% |
| BUY @ 3rd Alert (retest2) | 71 | 28 | 39.4% | 10 | 61 | 0 | 1.17% | 83.3% |
| SELL (all) | 90 | 34 | 37.8% | 1 | 75 | 14 | 0.25% | 22.2% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.17% | 0.2% |
| SELL @ 3rd Alert (retest2) | 89 | 33 | 37.1% | 1 | 74 | 14 | 0.25% | 22.1% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.51% | -2.1% |
| retest2 (combined) | 160 | 61 | 38.1% | 11 | 135 | 14 | 0.66% | 105.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 2358.00 | 2379.19 | 2381.45 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 2396.15 | 2383.14 | 2381.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 2416.20 | 2389.76 | 2384.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 2403.20 | 2409.93 | 2398.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 2403.20 | 2409.93 | 2398.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 2409.50 | 2409.84 | 2399.44 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 2382.00 | 2394.65 | 2396.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 12:15:00 | 2375.05 | 2387.52 | 2391.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 2398.00 | 2385.82 | 2389.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 2398.00 | 2385.82 | 2389.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 2398.00 | 2385.82 | 2389.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 2406.50 | 2386.64 | 2389.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 2392.00 | 2387.71 | 2389.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 2384.55 | 2387.71 | 2389.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 2393.65 | 2388.90 | 2389.87 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 14:15:00 | 2399.00 | 2390.78 | 2390.38 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 2381.40 | 2388.83 | 2389.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 11:15:00 | 2380.00 | 2387.06 | 2388.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 2383.95 | 2382.54 | 2385.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 10:00:00 | 2383.95 | 2382.54 | 2385.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 6 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 2415.00 | 2388.49 | 2386.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 12:15:00 | 2418.35 | 2403.87 | 2395.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 11:15:00 | 2420.00 | 2421.12 | 2409.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 12:00:00 | 2420.00 | 2421.12 | 2409.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 2412.85 | 2419.47 | 2409.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:45:00 | 2400.50 | 2419.47 | 2409.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 2420.15 | 2419.61 | 2410.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 14:15:00 | 2424.05 | 2419.61 | 2410.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 10:15:00 | 2427.65 | 2424.61 | 2415.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 2389.65 | 2414.60 | 2414.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 2389.65 | 2414.60 | 2414.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 2350.80 | 2388.39 | 2399.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 11:15:00 | 2399.90 | 2387.60 | 2397.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 11:15:00 | 2399.90 | 2387.60 | 2397.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 2399.90 | 2387.60 | 2397.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:00:00 | 2399.90 | 2387.60 | 2397.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 2412.95 | 2392.67 | 2398.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:30:00 | 2415.00 | 2392.67 | 2398.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 2410.65 | 2396.48 | 2399.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 15:00:00 | 2410.65 | 2396.48 | 2399.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 2400.00 | 2397.18 | 2399.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 2372.55 | 2397.18 | 2399.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 2253.92 | 2312.82 | 2334.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 2135.30 | 2251.54 | 2297.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 2329.80 | 2309.27 | 2306.54 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-06 11:15:00 | 2294.55 | 2305.73 | 2305.97 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 2375.95 | 2318.92 | 2311.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 2407.45 | 2346.56 | 2326.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 2394.70 | 2408.64 | 2386.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 09:15:00 | 2394.70 | 2408.64 | 2386.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 2394.70 | 2408.64 | 2386.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:00:00 | 2394.70 | 2408.64 | 2386.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 2419.00 | 2410.71 | 2389.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:30:00 | 2419.55 | 2411.74 | 2393.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:15:00 | 2419.70 | 2411.74 | 2393.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 14:15:00 | 2379.55 | 2406.71 | 2394.80 | SL hit (close<static) qty=1.00 sl=2385.05 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 14:15:00 | 2380.50 | 2399.04 | 2401.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 2369.00 | 2393.03 | 2398.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 14:15:00 | 2416.65 | 2385.63 | 2389.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 14:15:00 | 2416.65 | 2385.63 | 2389.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 2416.65 | 2385.63 | 2389.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 15:00:00 | 2416.65 | 2385.63 | 2389.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 2410.95 | 2390.70 | 2391.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 2375.60 | 2390.70 | 2391.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 14:15:00 | 2256.82 | 2280.38 | 2295.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-03 09:15:00 | 2259.15 | 2254.48 | 2269.78 | SL hit (close>ema200) qty=0.50 sl=2254.48 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 2288.35 | 2249.40 | 2244.56 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 2245.00 | 2258.03 | 2258.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 12:15:00 | 2215.75 | 2246.40 | 2252.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 2255.05 | 2247.56 | 2252.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 2255.05 | 2247.56 | 2252.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 2255.05 | 2247.56 | 2252.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 2255.05 | 2247.56 | 2252.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 2258.00 | 2249.65 | 2252.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 2256.20 | 2249.65 | 2252.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 2246.65 | 2249.05 | 2252.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 2241.10 | 2249.05 | 2252.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:00:00 | 2243.15 | 2247.87 | 2251.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:45:00 | 2242.65 | 2223.12 | 2226.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 12:30:00 | 2242.40 | 2227.56 | 2228.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 13:15:00 | 2241.30 | 2230.31 | 2229.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 2241.30 | 2230.31 | 2229.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 14:15:00 | 2265.00 | 2237.25 | 2232.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 2230.00 | 2239.31 | 2235.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 11:15:00 | 2230.00 | 2239.31 | 2235.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 2230.00 | 2239.31 | 2235.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:30:00 | 2230.00 | 2239.31 | 2235.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 2227.35 | 2236.92 | 2234.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:45:00 | 2226.95 | 2236.92 | 2234.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 2228.80 | 2233.39 | 2233.40 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 09:15:00 | 2250.00 | 2235.52 | 2234.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 13:15:00 | 2269.00 | 2245.77 | 2239.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 15:15:00 | 2268.00 | 2286.21 | 2271.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 15:15:00 | 2268.00 | 2286.21 | 2271.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 2268.00 | 2286.21 | 2271.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 11:15:00 | 2301.00 | 2288.55 | 2274.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 11:45:00 | 2300.60 | 2291.41 | 2277.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 12:30:00 | 2316.00 | 2295.35 | 2280.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 15:00:00 | 2301.85 | 2298.06 | 2284.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 2314.00 | 2301.25 | 2287.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 09:45:00 | 2316.80 | 2306.40 | 2290.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-30 14:15:00 | 2531.10 | 2454.27 | 2427.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 10:15:00 | 3036.80 | 3095.03 | 3096.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 11:15:00 | 3024.20 | 3080.86 | 3089.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 3046.00 | 3036.95 | 3061.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 3046.00 | 3036.95 | 3061.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 3046.00 | 3036.95 | 3061.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:30:00 | 3058.95 | 3036.95 | 3061.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 2990.00 | 2978.22 | 3014.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 14:45:00 | 2940.85 | 2978.27 | 3001.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 15:15:00 | 3080.00 | 3015.88 | 3007.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 15:15:00 | 3080.00 | 3015.88 | 3007.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 3136.45 | 3039.99 | 3018.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 3081.55 | 3096.99 | 3066.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 3081.55 | 3096.99 | 3066.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 3068.80 | 3093.75 | 3080.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:30:00 | 3068.05 | 3093.75 | 3080.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 3036.30 | 3082.26 | 3076.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:45:00 | 3030.70 | 3082.26 | 3076.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 2977.40 | 3061.29 | 3067.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 11:15:00 | 2968.15 | 3001.72 | 3029.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 2983.90 | 2982.99 | 3008.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 2983.90 | 2982.99 | 3008.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 2983.90 | 2982.99 | 3008.12 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 15:15:00 | 3058.00 | 3014.20 | 3012.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 09:15:00 | 3212.10 | 3053.78 | 3030.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 13:15:00 | 3199.00 | 3201.79 | 3165.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:30:00 | 3200.30 | 3201.79 | 3165.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 3151.50 | 3190.63 | 3175.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:00:00 | 3151.50 | 3190.63 | 3175.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 3170.55 | 3186.61 | 3175.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 3242.00 | 3176.94 | 3172.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 12:15:00 | 3297.00 | 3382.31 | 3391.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 12:15:00 | 3297.00 | 3382.31 | 3391.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 13:15:00 | 3275.00 | 3360.84 | 3380.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 3153.20 | 3132.36 | 3171.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 3153.20 | 3132.36 | 3171.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 3153.20 | 3132.36 | 3171.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 3166.40 | 3132.36 | 3171.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 3201.35 | 3146.15 | 3174.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 3201.35 | 3146.15 | 3174.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 3188.70 | 3154.66 | 3175.92 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 3199.45 | 3185.70 | 3184.78 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 10:15:00 | 3171.25 | 3182.81 | 3183.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 12:15:00 | 3149.90 | 3175.64 | 3180.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 3174.75 | 3161.24 | 3169.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 10:15:00 | 3174.75 | 3161.24 | 3169.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 3174.75 | 3161.24 | 3169.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:00:00 | 3174.75 | 3161.24 | 3169.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 3149.35 | 3158.86 | 3167.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 12:45:00 | 3139.80 | 3155.09 | 3165.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 13:15:00 | 3139.00 | 3155.09 | 3165.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 09:30:00 | 3130.15 | 3137.29 | 3152.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 10:15:00 | 3138.05 | 3137.29 | 3152.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 3122.95 | 3134.42 | 3149.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:15:00 | 3106.45 | 3134.42 | 3149.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 3235.70 | 3135.47 | 3143.04 | SL hit (close>static) qty=1.00 sl=3177.90 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 15:15:00 | 3225.00 | 3153.38 | 3150.49 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 3133.75 | 3177.98 | 3179.23 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 3206.95 | 3180.81 | 3180.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 13:15:00 | 3214.75 | 3195.99 | 3188.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 13:15:00 | 3222.00 | 3247.54 | 3227.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 13:15:00 | 3222.00 | 3247.54 | 3227.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 3222.00 | 3247.54 | 3227.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:45:00 | 3220.10 | 3247.54 | 3227.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 3196.25 | 3237.28 | 3224.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:30:00 | 3197.10 | 3237.28 | 3224.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 10:15:00 | 3202.00 | 3215.91 | 3216.94 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 13:15:00 | 3230.85 | 3218.61 | 3217.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 15:15:00 | 3244.20 | 3225.92 | 3221.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 3291.45 | 3299.28 | 3269.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 3291.45 | 3299.28 | 3269.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 3291.45 | 3299.28 | 3269.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 3271.35 | 3299.28 | 3269.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 3275.55 | 3294.53 | 3270.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:30:00 | 3304.90 | 3296.74 | 3273.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 15:00:00 | 3315.95 | 3310.65 | 3286.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 09:30:00 | 3358.60 | 3337.36 | 3317.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 15:15:00 | 3335.00 | 3353.60 | 3355.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 15:15:00 | 3335.00 | 3353.60 | 3355.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 3315.40 | 3342.79 | 3349.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 3298.95 | 3282.13 | 3307.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 3298.95 | 3282.13 | 3307.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 3298.95 | 3282.13 | 3307.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:15:00 | 3254.75 | 3283.66 | 3304.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:45:00 | 3255.65 | 3275.65 | 3296.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 14:30:00 | 3257.50 | 3279.37 | 3296.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 3374.50 | 3299.78 | 3302.75 | SL hit (close>static) qty=1.00 sl=3345.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 3340.00 | 3307.82 | 3306.13 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 3254.50 | 3303.79 | 3309.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 3237.40 | 3290.52 | 3303.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 2997.55 | 2980.03 | 3032.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 12:15:00 | 3020.15 | 2992.88 | 3026.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 3020.15 | 2992.88 | 3026.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:00:00 | 3020.15 | 2992.88 | 3026.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 2990.25 | 2989.20 | 3013.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 2938.10 | 2983.73 | 3001.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:15:00 | 2947.85 | 2935.84 | 2956.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 12:15:00 | 3018.05 | 2960.71 | 2957.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 3018.05 | 2960.71 | 2957.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 3060.00 | 2992.03 | 2974.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 2994.80 | 3044.28 | 3021.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 2994.80 | 3044.28 | 3021.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 2994.80 | 3044.28 | 3021.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 2994.80 | 3044.28 | 3021.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 3012.00 | 3037.82 | 3020.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 3019.60 | 3037.82 | 3020.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:45:00 | 3020.80 | 3030.05 | 3018.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 12:15:00 | 2987.85 | 3021.61 | 3015.60 | SL hit (close<static) qty=1.00 sl=2994.80 alert=retest2 |

### Cycle 33 — SELL (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 14:15:00 | 2976.60 | 3010.76 | 3011.60 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 15:15:00 | 3050.00 | 3018.61 | 3015.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 09:15:00 | 3077.85 | 3030.46 | 3020.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 11:15:00 | 3076.00 | 3083.79 | 3061.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-06 12:00:00 | 3076.00 | 3083.79 | 3061.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 3040.40 | 3082.18 | 3070.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 3040.40 | 3082.18 | 3070.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 3037.80 | 3073.30 | 3067.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 3038.90 | 3073.30 | 3067.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 2982.05 | 3055.05 | 3059.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 2963.25 | 3002.89 | 3028.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2824.45 | 2803.19 | 2831.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 2824.45 | 2803.19 | 2831.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 2824.45 | 2803.19 | 2831.10 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 09:15:00 | 2870.00 | 2846.12 | 2842.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 2943.40 | 2899.38 | 2878.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 12:15:00 | 2923.90 | 2930.39 | 2907.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 13:00:00 | 2923.90 | 2930.39 | 2907.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 2940.25 | 2932.36 | 2910.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:30:00 | 2910.40 | 2932.36 | 2910.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 2968.65 | 3004.24 | 2979.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 2968.65 | 3004.24 | 2979.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 2980.00 | 2999.39 | 2979.12 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 2960.15 | 2984.20 | 2985.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 2917.15 | 2970.79 | 2979.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 2977.85 | 2954.33 | 2965.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 2977.85 | 2954.33 | 2965.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 2977.85 | 2954.33 | 2965.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 2977.85 | 2954.33 | 2965.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 3028.00 | 2969.06 | 2971.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 3028.00 | 2969.06 | 2971.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 12:15:00 | 3048.00 | 2984.85 | 2978.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 3052.70 | 3019.83 | 3000.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 13:15:00 | 3020.00 | 3021.60 | 3004.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 13:30:00 | 3022.25 | 3021.60 | 3004.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 3030.00 | 3023.28 | 3006.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 3051.95 | 3024.63 | 3008.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 11:15:00 | 2980.05 | 3013.12 | 3007.52 | SL hit (close<static) qty=1.00 sl=3000.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 14:15:00 | 2960.40 | 2999.55 | 3002.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 2948.00 | 2981.53 | 2990.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 15:15:00 | 2798.00 | 2796.59 | 2821.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 09:15:00 | 2843.25 | 2796.59 | 2821.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 2838.15 | 2804.90 | 2823.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:30:00 | 2843.60 | 2804.90 | 2823.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 2810.50 | 2806.02 | 2822.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:30:00 | 2807.20 | 2806.40 | 2820.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:15:00 | 2804.90 | 2806.40 | 2820.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:45:00 | 2806.65 | 2805.35 | 2819.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 14:00:00 | 2805.30 | 2805.34 | 2817.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 2841.00 | 2812.47 | 2819.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 2841.00 | 2812.47 | 2819.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 2825.35 | 2815.05 | 2820.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 2849.15 | 2815.05 | 2820.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 2810.00 | 2803.11 | 2810.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 2810.00 | 2803.11 | 2810.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 2799.75 | 2802.44 | 2809.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 2813.95 | 2802.44 | 2809.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 2817.65 | 2805.48 | 2810.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 2823.40 | 2805.48 | 2810.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 2810.65 | 2806.52 | 2810.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-16 11:15:00 | 2841.00 | 2813.41 | 2813.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 2841.00 | 2813.41 | 2813.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 12:15:00 | 2879.40 | 2826.61 | 2819.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 11:15:00 | 2865.15 | 2866.15 | 2846.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 12:00:00 | 2865.15 | 2866.15 | 2846.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 2854.00 | 2864.72 | 2849.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:45:00 | 2859.15 | 2864.72 | 2849.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 2865.90 | 2864.96 | 2850.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 2879.95 | 2862.97 | 2851.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:45:00 | 2891.25 | 2886.33 | 2872.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 2882.55 | 2884.87 | 2873.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:00:00 | 2885.05 | 2883.25 | 2874.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 2870.95 | 2880.79 | 2874.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 2870.95 | 2880.79 | 2874.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 2861.95 | 2877.02 | 2873.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:45:00 | 2863.40 | 2877.02 | 2873.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2865.55 | 2874.73 | 2872.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:45:00 | 2869.50 | 2877.70 | 2873.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 2884.10 | 2885.59 | 2879.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:30:00 | 2902.20 | 2885.59 | 2879.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 2858.05 | 2880.09 | 2877.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 2858.05 | 2880.09 | 2877.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-20 15:15:00 | 2819.90 | 2868.05 | 2872.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 2819.90 | 2868.05 | 2872.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 12:15:00 | 2800.00 | 2833.31 | 2852.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 13:15:00 | 2807.85 | 2806.51 | 2825.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 13:30:00 | 2804.30 | 2806.51 | 2825.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 2785.00 | 2781.95 | 2798.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 2818.55 | 2781.95 | 2798.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 2829.90 | 2791.54 | 2801.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 2829.90 | 2791.54 | 2801.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 2865.80 | 2806.39 | 2807.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 2865.80 | 2806.39 | 2807.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 11:15:00 | 2924.85 | 2830.08 | 2817.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 2937.55 | 2851.57 | 2828.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 10:15:00 | 2931.35 | 2931.95 | 2884.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 12:15:00 | 2951.80 | 2935.53 | 2890.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 09:15:00 | 2954.00 | 2941.22 | 2907.73 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 09:45:00 | 2951.50 | 2938.87 | 2909.71 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 2936.95 | 2935.04 | 2918.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 2924.10 | 2935.04 | 2918.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 2943.00 | 2936.63 | 2920.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 2921.40 | 2936.63 | 2920.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 2939.30 | 2937.17 | 2922.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:30:00 | 2942.65 | 2940.95 | 2925.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 2930.55 | 2956.09 | 2941.61 | SL hit (close<ema400) qty=1.00 sl=2941.61 alert=retest1 |

### Cycle 43 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 2899.00 | 2935.86 | 2938.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 11:15:00 | 2879.10 | 2924.51 | 2932.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 2938.00 | 2905.84 | 2918.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 2938.00 | 2905.84 | 2918.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2938.00 | 2905.84 | 2918.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2960.30 | 2905.84 | 2918.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 2918.00 | 2908.28 | 2918.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:15:00 | 2943.95 | 2908.28 | 2918.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 2950.75 | 2916.77 | 2921.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:30:00 | 2958.75 | 2916.77 | 2921.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 12:15:00 | 2956.00 | 2924.62 | 2924.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 10:15:00 | 3047.95 | 2964.68 | 2944.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 14:15:00 | 2995.75 | 3001.23 | 2972.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-07 15:00:00 | 2995.75 | 3001.23 | 2972.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2966.75 | 2993.98 | 2973.76 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 09:15:00 | 2953.00 | 2963.43 | 2964.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 15:15:00 | 2925.00 | 2948.15 | 2955.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 15:15:00 | 2820.00 | 2783.79 | 2832.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 15:15:00 | 2820.00 | 2783.79 | 2832.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 15:15:00 | 2820.00 | 2783.79 | 2832.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 09:15:00 | 2767.00 | 2783.79 | 2832.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 10:00:00 | 2760.45 | 2779.12 | 2825.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 2851.05 | 2812.79 | 2821.57 | SL hit (close>static) qty=1.00 sl=2849.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 2843.15 | 2827.91 | 2826.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 2876.05 | 2837.54 | 2831.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 2830.20 | 2855.16 | 2848.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 2830.20 | 2855.16 | 2848.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 2830.20 | 2855.16 | 2848.06 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 2834.50 | 2843.56 | 2843.80 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 13:15:00 | 2865.05 | 2847.86 | 2845.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 10:15:00 | 2878.60 | 2859.25 | 2852.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 2825.95 | 2861.81 | 2858.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 2825.95 | 2861.81 | 2858.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 2825.95 | 2861.81 | 2858.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 2825.95 | 2861.81 | 2858.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 2797.25 | 2848.90 | 2852.62 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 2874.50 | 2844.49 | 2844.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 2913.40 | 2863.26 | 2853.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 2845.05 | 2873.89 | 2864.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 2845.05 | 2873.89 | 2864.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 2845.05 | 2873.89 | 2864.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 2845.05 | 2873.89 | 2864.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 2837.40 | 2866.60 | 2862.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 2837.40 | 2866.60 | 2862.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 2823.55 | 2853.33 | 2856.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 2809.20 | 2844.50 | 2852.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 10:15:00 | 2634.85 | 2609.70 | 2662.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:45:00 | 2641.50 | 2609.70 | 2662.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 2695.50 | 2636.92 | 2653.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 2698.50 | 2636.92 | 2653.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 2673.00 | 2644.13 | 2655.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:45:00 | 2658.15 | 2646.33 | 2655.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:00:00 | 2659.45 | 2648.95 | 2655.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 15:15:00 | 2674.45 | 2660.98 | 2660.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 15:15:00 | 2674.45 | 2660.98 | 2660.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 2722.70 | 2673.32 | 2666.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 13:15:00 | 2714.75 | 2716.27 | 2697.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 14:00:00 | 2714.75 | 2716.27 | 2697.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 2724.95 | 2718.01 | 2700.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:30:00 | 2691.35 | 2718.01 | 2700.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 2701.05 | 2714.14 | 2701.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:15:00 | 2672.45 | 2714.14 | 2701.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 2696.85 | 2710.68 | 2701.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 12:00:00 | 2752.00 | 2718.94 | 2705.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-03 14:15:00 | 3027.20 | 2818.21 | 2756.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 10:15:00 | 2726.55 | 2773.30 | 2779.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 12:15:00 | 2715.00 | 2753.75 | 2768.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 2533.25 | 2523.96 | 2556.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:30:00 | 2522.15 | 2523.96 | 2556.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 2529.45 | 2516.70 | 2531.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:45:00 | 2542.75 | 2516.70 | 2531.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 2526.00 | 2518.56 | 2531.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:30:00 | 2535.80 | 2518.56 | 2531.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 13:15:00 | 2611.85 | 2537.22 | 2538.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:00:00 | 2611.85 | 2537.22 | 2538.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 14:15:00 | 2562.70 | 2542.32 | 2540.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 2786.85 | 2709.12 | 2662.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 2681.00 | 2732.45 | 2691.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 2681.00 | 2732.45 | 2691.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 2681.00 | 2732.45 | 2691.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:00:00 | 2681.00 | 2732.45 | 2691.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 2672.75 | 2720.51 | 2689.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:45:00 | 2649.10 | 2720.51 | 2689.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 2606.10 | 2686.32 | 2678.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 2606.10 | 2686.32 | 2678.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 13:15:00 | 2619.25 | 2672.91 | 2673.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 2593.05 | 2654.55 | 2664.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 15:15:00 | 2624.80 | 2619.55 | 2638.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 2652.30 | 2626.10 | 2640.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 2652.30 | 2626.10 | 2640.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 2652.30 | 2626.10 | 2640.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 2601.85 | 2621.25 | 2636.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 12:30:00 | 2586.90 | 2615.15 | 2631.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 13:30:00 | 2587.00 | 2609.73 | 2627.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 2553.10 | 2547.76 | 2558.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 2457.55 | 2488.44 | 2519.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 2457.65 | 2488.44 | 2519.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 2425.44 | 2488.44 | 2519.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 2467.10 | 2459.49 | 2485.65 | SL hit (close>ema200) qty=0.50 sl=2459.49 alert=retest2 |

### Cycle 56 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 2553.45 | 2486.61 | 2481.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 2584.45 | 2534.14 | 2508.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 2567.05 | 2574.92 | 2555.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 2567.05 | 2574.92 | 2555.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 2560.00 | 2571.94 | 2555.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 2563.00 | 2571.94 | 2555.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2594.85 | 2576.52 | 2559.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:15:00 | 2605.50 | 2576.52 | 2559.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 12:15:00 | 2521.90 | 2560.49 | 2555.97 | SL hit (close<static) qty=1.00 sl=2542.75 alert=retest2 |

### Cycle 57 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 2518.20 | 2552.04 | 2552.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 2462.10 | 2525.27 | 2539.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 2535.75 | 2511.00 | 2525.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 13:15:00 | 2535.75 | 2511.00 | 2525.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 2535.75 | 2511.00 | 2525.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:00:00 | 2535.75 | 2511.00 | 2525.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 2496.00 | 2508.00 | 2523.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:45:00 | 2569.35 | 2508.00 | 2523.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2486.50 | 2501.62 | 2517.51 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 2554.80 | 2526.48 | 2522.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 10:15:00 | 2587.40 | 2538.66 | 2528.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 2534.25 | 2555.62 | 2541.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 14:15:00 | 2534.25 | 2555.62 | 2541.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 2534.25 | 2555.62 | 2541.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 2534.25 | 2555.62 | 2541.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 2529.95 | 2550.49 | 2540.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 2582.50 | 2550.49 | 2540.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 2609.00 | 2562.19 | 2546.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 14:30:00 | 2614.55 | 2580.08 | 2571.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 15:15:00 | 2623.00 | 2580.08 | 2571.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 12:15:00 | 2614.30 | 2593.65 | 2581.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 12:45:00 | 2613.85 | 2600.12 | 2585.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 13:15:00 | 2876.01 | 2726.08 | 2674.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 2662.60 | 2716.39 | 2721.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 10:15:00 | 2648.00 | 2702.71 | 2714.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 2646.80 | 2626.15 | 2648.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 2646.80 | 2626.15 | 2648.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 2646.80 | 2626.15 | 2648.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 2657.85 | 2626.15 | 2648.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 2617.05 | 2624.33 | 2645.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:30:00 | 2636.70 | 2624.33 | 2645.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 2623.90 | 2615.63 | 2635.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 2623.90 | 2615.63 | 2635.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 2609.70 | 2613.97 | 2631.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 2600.45 | 2613.97 | 2631.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 2663.35 | 2627.40 | 2630.10 | SL hit (close>static) qty=1.00 sl=2643.10 alert=retest2 |

### Cycle 60 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 2527.70 | 2473.46 | 2468.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 2569.40 | 2523.74 | 2498.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 2674.20 | 2677.40 | 2644.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 09:30:00 | 2671.30 | 2677.40 | 2644.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 2679.10 | 2681.75 | 2664.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:45:00 | 2693.50 | 2681.40 | 2665.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 13:15:00 | 2695.00 | 2682.90 | 2668.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 14:00:00 | 2691.50 | 2684.62 | 2670.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:30:00 | 2696.20 | 2682.12 | 2673.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 2677.00 | 2681.09 | 2673.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 2672.40 | 2681.09 | 2673.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 2677.30 | 2680.33 | 2673.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:45:00 | 2669.10 | 2680.33 | 2673.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 2676.80 | 2679.63 | 2674.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 2680.00 | 2679.63 | 2674.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 2681.90 | 2683.71 | 2677.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:15:00 | 2708.40 | 2683.71 | 2677.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 2721.60 | 2691.29 | 2681.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:45:00 | 2732.90 | 2703.03 | 2687.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 2667.90 | 2707.06 | 2711.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 2667.90 | 2707.06 | 2711.21 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 2766.00 | 2722.77 | 2717.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 2775.10 | 2750.25 | 2733.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 12:15:00 | 2762.30 | 2762.49 | 2747.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 13:00:00 | 2762.30 | 2762.49 | 2747.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 2770.00 | 2765.17 | 2752.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 2739.30 | 2765.17 | 2752.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 2758.60 | 2763.86 | 2752.83 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 2705.80 | 2746.06 | 2748.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 2680.00 | 2732.85 | 2741.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 11:15:00 | 2529.60 | 2506.57 | 2529.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 11:15:00 | 2529.60 | 2506.57 | 2529.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 2529.60 | 2506.57 | 2529.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:45:00 | 2527.30 | 2506.57 | 2529.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 2530.00 | 2511.26 | 2529.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:45:00 | 2529.90 | 2511.26 | 2529.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 2513.00 | 2511.61 | 2528.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 15:15:00 | 2486.20 | 2513.69 | 2527.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:45:00 | 2512.20 | 2510.93 | 2523.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 10:15:00 | 2512.10 | 2510.93 | 2523.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 14:15:00 | 2538.20 | 2519.93 | 2523.19 | SL hit (close>static) qty=1.00 sl=2530.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2540.20 | 2525.77 | 2525.39 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 11:15:00 | 2498.20 | 2523.83 | 2524.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-12 12:15:00 | 2478.40 | 2514.75 | 2520.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 2523.00 | 2504.75 | 2512.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 2523.00 | 2504.75 | 2512.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 2523.00 | 2504.75 | 2512.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:15:00 | 2530.00 | 2504.75 | 2512.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 2559.50 | 2515.70 | 2517.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:00:00 | 2559.50 | 2515.70 | 2517.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 11:15:00 | 2570.70 | 2526.70 | 2521.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 2590.70 | 2552.81 | 2536.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 2555.00 | 2558.40 | 2541.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2555.00 | 2558.40 | 2541.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2555.00 | 2558.40 | 2541.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 14:45:00 | 2585.80 | 2571.42 | 2554.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 11:45:00 | 2597.80 | 2577.61 | 2563.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 2590.70 | 2580.23 | 2565.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 2589.00 | 2588.52 | 2575.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2659.00 | 2610.23 | 2592.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 2695.50 | 2624.27 | 2618.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 2591.60 | 2620.75 | 2622.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 2591.60 | 2620.75 | 2622.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 2565.70 | 2609.74 | 2617.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 2590.40 | 2583.51 | 2594.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 2590.40 | 2583.51 | 2594.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2590.40 | 2583.51 | 2594.37 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 2604.20 | 2593.38 | 2592.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 13:15:00 | 2610.00 | 2598.61 | 2595.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 2584.90 | 2595.87 | 2594.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 14:15:00 | 2584.90 | 2595.87 | 2594.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 2584.90 | 2595.87 | 2594.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 2584.90 | 2595.87 | 2594.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 2580.00 | 2592.69 | 2592.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 2575.50 | 2584.48 | 2588.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 2550.10 | 2530.12 | 2546.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 2550.10 | 2530.12 | 2546.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 2550.10 | 2530.12 | 2546.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 2550.10 | 2530.12 | 2546.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 2550.90 | 2534.27 | 2546.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:30:00 | 2560.80 | 2534.27 | 2546.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 2561.50 | 2542.86 | 2547.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 2561.50 | 2542.86 | 2547.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 2560.60 | 2546.41 | 2548.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:30:00 | 2569.80 | 2546.41 | 2548.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 2560.10 | 2551.62 | 2551.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 2580.40 | 2560.28 | 2555.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 11:15:00 | 2585.80 | 2590.16 | 2576.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 11:30:00 | 2582.70 | 2590.16 | 2576.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 2564.30 | 2584.99 | 2575.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:45:00 | 2565.00 | 2584.99 | 2575.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 2573.00 | 2582.59 | 2575.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:15:00 | 2575.10 | 2582.59 | 2575.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:45:00 | 2574.90 | 2581.43 | 2575.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 12:15:00 | 2575.50 | 2581.98 | 2578.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:00:00 | 2575.50 | 2580.68 | 2577.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 15:15:00 | 2572.00 | 2576.14 | 2576.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 2572.00 | 2576.14 | 2576.23 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 2594.00 | 2578.78 | 2577.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 13:15:00 | 2597.10 | 2586.75 | 2581.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 2585.10 | 2588.96 | 2583.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 15:15:00 | 2585.10 | 2588.96 | 2583.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 2585.10 | 2588.96 | 2583.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 2582.80 | 2587.85 | 2583.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 2576.60 | 2585.60 | 2583.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:30:00 | 2573.50 | 2585.60 | 2583.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 2574.50 | 2583.38 | 2582.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:30:00 | 2571.40 | 2583.38 | 2582.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 2565.20 | 2578.63 | 2580.25 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 2639.30 | 2591.10 | 2585.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 14:15:00 | 2647.90 | 2621.99 | 2603.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 2678.00 | 2702.66 | 2668.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 2678.00 | 2702.66 | 2668.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2678.00 | 2702.66 | 2668.53 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 15:15:00 | 2632.00 | 2655.05 | 2655.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 2616.80 | 2647.40 | 2652.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 10:15:00 | 2570.70 | 2563.15 | 2588.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 11:00:00 | 2570.70 | 2563.15 | 2588.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 2584.80 | 2568.96 | 2583.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 2584.80 | 2568.96 | 2583.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 2575.50 | 2570.27 | 2582.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 2584.70 | 2570.27 | 2582.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 2558.50 | 2567.91 | 2580.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 2547.80 | 2567.91 | 2580.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 2551.50 | 2562.94 | 2575.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:00:00 | 2550.80 | 2552.80 | 2563.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 2590.00 | 2568.29 | 2567.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 2590.00 | 2568.29 | 2567.34 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 2565.10 | 2566.54 | 2566.64 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 2575.00 | 2566.47 | 2566.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 2587.80 | 2572.45 | 2569.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 2579.60 | 2582.93 | 2576.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 2579.60 | 2582.93 | 2576.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 2579.60 | 2582.93 | 2576.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 2579.60 | 2582.93 | 2576.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 2582.20 | 2582.78 | 2577.02 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 15:15:00 | 2573.00 | 2574.36 | 2574.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 10:15:00 | 2561.50 | 2570.55 | 2572.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 11:15:00 | 2573.70 | 2571.18 | 2572.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 11:15:00 | 2573.70 | 2571.18 | 2572.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2573.70 | 2571.18 | 2572.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:00:00 | 2573.70 | 2571.18 | 2572.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 2553.80 | 2567.70 | 2571.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 2547.50 | 2567.70 | 2571.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:45:00 | 2535.70 | 2556.31 | 2565.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:00:00 | 2547.20 | 2556.39 | 2562.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 15:15:00 | 2575.00 | 2565.31 | 2565.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 15:15:00 | 2575.00 | 2565.31 | 2565.03 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 2529.50 | 2559.79 | 2562.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 2508.60 | 2549.55 | 2557.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 2519.50 | 2516.38 | 2532.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 12:00:00 | 2519.50 | 2516.38 | 2532.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 2555.00 | 2524.10 | 2534.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 2550.70 | 2524.10 | 2534.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2560.00 | 2531.28 | 2536.99 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 2565.30 | 2544.08 | 2541.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 2621.00 | 2563.32 | 2551.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 2665.90 | 2668.21 | 2634.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 10:45:00 | 2664.50 | 2668.21 | 2634.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2633.40 | 2654.40 | 2641.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 2633.40 | 2654.40 | 2641.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 2607.10 | 2644.94 | 2638.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 2607.10 | 2644.94 | 2638.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2616.50 | 2639.25 | 2636.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 2609.10 | 2639.25 | 2636.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 2622.00 | 2633.12 | 2634.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 14:15:00 | 2610.50 | 2628.60 | 2631.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 2624.70 | 2623.73 | 2628.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 2624.70 | 2623.73 | 2628.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 2624.70 | 2623.73 | 2628.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 2618.50 | 2623.73 | 2628.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 2616.00 | 2619.26 | 2623.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:45:00 | 2613.50 | 2605.87 | 2611.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 2634.80 | 2611.66 | 2613.56 | SL hit (close>static) qty=1.00 sl=2634.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 12:15:00 | 2634.00 | 2616.12 | 2615.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 2671.10 | 2634.14 | 2625.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 14:15:00 | 2781.40 | 2785.64 | 2748.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 15:00:00 | 2781.40 | 2785.64 | 2748.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 2776.90 | 2783.93 | 2762.16 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 2745.00 | 2763.85 | 2764.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 15:15:00 | 2735.00 | 2751.38 | 2757.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 2769.10 | 2736.30 | 2743.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 2769.10 | 2736.30 | 2743.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2769.10 | 2736.30 | 2743.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 2769.10 | 2736.30 | 2743.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 2804.70 | 2749.98 | 2749.51 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 2739.00 | 2772.96 | 2774.10 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 2812.40 | 2776.41 | 2773.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 2824.50 | 2786.03 | 2778.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 2799.20 | 2799.75 | 2787.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 2799.20 | 2799.75 | 2787.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 2799.20 | 2799.75 | 2787.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:45:00 | 2789.00 | 2799.75 | 2787.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2792.40 | 2800.91 | 2794.72 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 2746.70 | 2786.61 | 2789.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 2742.00 | 2763.67 | 2776.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 12:15:00 | 2574.50 | 2566.68 | 2597.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 13:00:00 | 2574.50 | 2566.68 | 2597.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 2599.30 | 2573.21 | 2597.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 2599.30 | 2573.21 | 2597.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2619.70 | 2582.50 | 2599.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2619.70 | 2582.50 | 2599.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2612.00 | 2588.40 | 2600.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 2589.20 | 2588.40 | 2600.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 2599.40 | 2594.55 | 2601.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 2594.40 | 2594.55 | 2601.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 2592.60 | 2598.09 | 2602.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:45:00 | 2591.80 | 2597.18 | 2601.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 2618.00 | 2605.35 | 2604.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 2618.00 | 2605.35 | 2604.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 2640.40 | 2612.36 | 2607.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 2673.60 | 2674.66 | 2653.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:45:00 | 2677.00 | 2674.66 | 2653.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2666.10 | 2674.72 | 2657.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 2661.20 | 2674.72 | 2657.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2700.30 | 2679.81 | 2666.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:45:00 | 2716.50 | 2682.84 | 2673.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 12:15:00 | 2655.00 | 2682.32 | 2680.00 | SL hit (close<static) qty=1.00 sl=2660.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 2637.30 | 2673.32 | 2676.12 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 2702.30 | 2679.11 | 2678.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 2737.00 | 2690.69 | 2683.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 2671.80 | 2686.91 | 2682.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 2671.80 | 2686.91 | 2682.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2671.80 | 2686.91 | 2682.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 2671.80 | 2686.91 | 2682.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2691.50 | 2687.83 | 2683.52 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 2671.80 | 2681.14 | 2681.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 2654.70 | 2669.71 | 2675.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 2499.80 | 2491.79 | 2524.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 15:15:00 | 2499.80 | 2491.79 | 2524.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 2499.80 | 2491.79 | 2524.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:15:00 | 2449.60 | 2485.71 | 2518.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:30:00 | 2450.10 | 2470.91 | 2502.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:45:00 | 2450.50 | 2465.46 | 2494.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:15:00 | 2448.70 | 2465.46 | 2494.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 2490.00 | 2470.77 | 2483.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 2490.00 | 2470.77 | 2483.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 2480.00 | 2472.61 | 2483.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:15:00 | 2475.00 | 2472.61 | 2483.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 2451.00 | 2474.61 | 2482.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 2529.30 | 2487.48 | 2487.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 2529.30 | 2487.48 | 2487.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 2559.80 | 2508.85 | 2497.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 09:15:00 | 2602.10 | 2605.54 | 2580.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 2602.10 | 2605.54 | 2580.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2602.10 | 2605.54 | 2580.51 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 2564.60 | 2578.27 | 2579.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 15:15:00 | 2555.00 | 2570.70 | 2576.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2581.30 | 2572.82 | 2576.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 2581.30 | 2572.82 | 2576.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2581.30 | 2572.82 | 2576.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 2581.30 | 2572.82 | 2576.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 2579.50 | 2574.15 | 2576.76 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 2592.30 | 2579.77 | 2578.99 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 10:15:00 | 2567.20 | 2578.26 | 2579.21 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 2586.80 | 2579.48 | 2578.94 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 2572.20 | 2578.02 | 2578.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 2568.00 | 2574.85 | 2576.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 2548.00 | 2546.48 | 2556.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 2548.00 | 2546.48 | 2556.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 2553.40 | 2547.86 | 2556.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 2553.70 | 2547.86 | 2556.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 2549.80 | 2548.25 | 2555.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:30:00 | 2551.40 | 2549.88 | 2555.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 2548.90 | 2549.68 | 2555.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 2547.10 | 2549.68 | 2555.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 2562.10 | 2553.74 | 2555.50 | SL hit (close>static) qty=1.00 sl=2559.20 alert=retest2 |

### Cycle 100 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 2430.00 | 2415.52 | 2414.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 2453.20 | 2428.81 | 2422.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 2424.00 | 2438.61 | 2430.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 2424.00 | 2438.61 | 2430.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 2424.00 | 2438.61 | 2430.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 2424.00 | 2438.61 | 2430.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 2424.20 | 2435.73 | 2429.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 2425.00 | 2435.73 | 2429.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 2400.20 | 2422.91 | 2424.41 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 2448.30 | 2424.16 | 2422.21 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 2402.80 | 2423.19 | 2423.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 2402.30 | 2419.01 | 2421.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 2419.00 | 2415.32 | 2418.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 2419.00 | 2415.32 | 2418.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 2419.00 | 2415.32 | 2418.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 2420.00 | 2415.32 | 2418.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 2405.00 | 2413.26 | 2417.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 2397.90 | 2413.26 | 2417.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 2429.80 | 2403.23 | 2407.63 | SL hit (close>static) qty=1.00 sl=2421.90 alert=retest2 |

### Cycle 104 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 2460.10 | 2416.76 | 2412.17 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 2409.30 | 2427.77 | 2429.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 2398.20 | 2412.05 | 2419.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 2414.00 | 2410.67 | 2417.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 2414.00 | 2410.67 | 2417.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 2414.00 | 2410.67 | 2417.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 2414.00 | 2410.67 | 2417.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 2414.90 | 2411.52 | 2417.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 2432.80 | 2411.52 | 2417.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 2419.00 | 2413.01 | 2417.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 2430.00 | 2413.01 | 2417.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 2412.00 | 2412.81 | 2417.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 2398.40 | 2411.39 | 2416.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 2397.50 | 2403.35 | 2408.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 2435.10 | 2406.28 | 2406.42 | SL hit (close>static) qty=1.00 sl=2421.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 2432.70 | 2411.57 | 2408.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 2449.80 | 2419.21 | 2412.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 2460.20 | 2461.72 | 2447.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:00:00 | 2460.20 | 2461.72 | 2447.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 2530.20 | 2481.58 | 2466.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 15:00:00 | 2542.90 | 2493.84 | 2473.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 2543.70 | 2515.29 | 2487.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 13:15:00 | 2523.10 | 2539.65 | 2541.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 2523.10 | 2539.65 | 2541.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 2509.60 | 2528.12 | 2535.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 12:15:00 | 2525.80 | 2525.26 | 2531.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:00:00 | 2525.80 | 2525.26 | 2531.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 2524.10 | 2524.89 | 2530.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 2524.10 | 2524.89 | 2530.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 2524.10 | 2524.73 | 2530.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 2516.80 | 2524.73 | 2530.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 2562.20 | 2535.61 | 2534.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 2562.20 | 2535.61 | 2534.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 2575.80 | 2550.58 | 2542.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 2560.00 | 2568.63 | 2559.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 15:15:00 | 2560.00 | 2568.63 | 2559.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 2560.00 | 2568.63 | 2559.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 2553.20 | 2568.63 | 2559.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 2554.10 | 2565.72 | 2559.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 2556.00 | 2565.72 | 2559.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 2566.00 | 2565.78 | 2559.90 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 2550.10 | 2556.51 | 2556.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 2530.00 | 2549.84 | 2553.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 15:15:00 | 2530.00 | 2527.59 | 2538.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 09:15:00 | 2500.00 | 2527.59 | 2538.46 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 2504.40 | 2506.07 | 2521.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 2512.10 | 2506.07 | 2521.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2484.30 | 2477.19 | 2491.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 2493.00 | 2477.19 | 2491.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 2495.70 | 2480.89 | 2492.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 2495.70 | 2480.89 | 2492.33 | SL hit (close>ema400) qty=1.00 sl=2492.33 alert=retest1 |

### Cycle 110 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 2511.00 | 2499.24 | 2498.46 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 2473.20 | 2502.56 | 2502.69 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 13:15:00 | 2500.90 | 2495.47 | 2495.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 2521.20 | 2500.62 | 2497.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 15:15:00 | 2500.00 | 2500.49 | 2497.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:15:00 | 2491.60 | 2500.49 | 2497.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 2500.00 | 2500.39 | 2498.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 2517.60 | 2500.71 | 2498.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:45:00 | 2517.30 | 2506.99 | 2502.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 2513.30 | 2506.99 | 2502.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 2519.70 | 2506.84 | 2503.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2517.30 | 2508.93 | 2504.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 2533.30 | 2508.93 | 2504.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 2540.10 | 2516.64 | 2508.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 2530.00 | 2518.53 | 2510.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 2530.70 | 2519.18 | 2511.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 2508.50 | 2521.78 | 2514.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 2508.40 | 2521.78 | 2514.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 2508.20 | 2519.06 | 2514.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 2525.20 | 2517.87 | 2514.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 2621.40 | 2635.23 | 2635.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 2621.40 | 2635.23 | 2635.46 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 2640.20 | 2634.93 | 2634.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 12:15:00 | 2646.00 | 2637.14 | 2635.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 2625.90 | 2641.97 | 2638.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 2625.90 | 2641.97 | 2638.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 2625.90 | 2641.97 | 2638.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 2625.90 | 2641.97 | 2638.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 2632.40 | 2640.06 | 2638.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 2624.90 | 2640.06 | 2638.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 2639.00 | 2639.85 | 2638.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 2623.40 | 2639.85 | 2638.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 2643.30 | 2640.54 | 2638.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 2643.30 | 2640.54 | 2638.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 2639.20 | 2640.27 | 2638.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 2639.20 | 2640.27 | 2638.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 2646.40 | 2641.50 | 2639.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 2638.00 | 2641.50 | 2639.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 2670.00 | 2647.20 | 2642.34 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 2636.20 | 2643.80 | 2644.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 2619.30 | 2638.28 | 2641.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 2601.80 | 2597.47 | 2612.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 12:30:00 | 2599.00 | 2597.47 | 2612.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 2570.80 | 2591.18 | 2604.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 2565.00 | 2586.92 | 2601.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:30:00 | 2564.50 | 2575.86 | 2593.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 2670.00 | 2602.14 | 2600.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 2670.00 | 2602.14 | 2600.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 2685.10 | 2647.10 | 2627.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 2675.00 | 2678.03 | 2654.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 2675.00 | 2678.03 | 2654.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2651.00 | 2672.62 | 2654.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 2655.00 | 2672.62 | 2654.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2663.40 | 2670.78 | 2655.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 2766.50 | 2667.80 | 2657.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 2687.00 | 2698.82 | 2698.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 2685.50 | 2696.16 | 2697.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 2685.50 | 2696.16 | 2697.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 2671.00 | 2686.86 | 2692.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 2693.50 | 2688.19 | 2692.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 2693.50 | 2688.19 | 2692.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2693.50 | 2688.19 | 2692.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:45:00 | 2694.30 | 2688.19 | 2692.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2690.30 | 2688.61 | 2692.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 2694.60 | 2688.61 | 2692.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 2694.00 | 2689.69 | 2692.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 2718.20 | 2689.69 | 2692.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 2728.30 | 2697.41 | 2695.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 10:15:00 | 2751.00 | 2708.13 | 2700.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 15:15:00 | 2720.00 | 2721.75 | 2711.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 2726.20 | 2721.75 | 2711.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2724.40 | 2722.28 | 2712.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:15:00 | 2745.60 | 2722.28 | 2712.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-02 12:15:00 | 3020.16 | 2921.92 | 2856.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 2816.80 | 2887.72 | 2893.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 2788.90 | 2822.87 | 2840.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 2750.00 | 2742.49 | 2775.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 2724.70 | 2742.49 | 2775.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2681.30 | 2699.30 | 2718.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 2665.70 | 2686.73 | 2707.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 2664.40 | 2683.56 | 2700.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 2669.00 | 2694.58 | 2699.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:00:00 | 2674.90 | 2686.14 | 2694.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 2692.50 | 2684.08 | 2690.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 2692.50 | 2684.08 | 2690.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 2681.10 | 2683.48 | 2690.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:30:00 | 2699.70 | 2683.48 | 2690.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 2695.00 | 2685.79 | 2690.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 2689.00 | 2685.79 | 2690.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 2705.40 | 2689.71 | 2691.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 2701.00 | 2689.71 | 2691.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 2675.10 | 2686.79 | 2690.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 2702.80 | 2693.11 | 2692.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 2702.80 | 2693.11 | 2692.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 2740.70 | 2708.58 | 2700.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 14:15:00 | 2717.40 | 2717.80 | 2709.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-22 15:00:00 | 2717.40 | 2717.80 | 2709.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2699.50 | 2713.53 | 2708.66 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2692.60 | 2703.72 | 2705.18 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 2718.60 | 2705.63 | 2705.58 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 2703.20 | 2705.48 | 2705.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 2700.10 | 2704.40 | 2705.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 2709.00 | 2702.52 | 2703.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 10:15:00 | 2709.00 | 2702.52 | 2703.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 2709.00 | 2702.52 | 2703.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 2709.00 | 2702.52 | 2703.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 2701.90 | 2702.39 | 2703.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 2700.10 | 2702.39 | 2703.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 2719.50 | 2704.93 | 2704.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 2719.50 | 2704.93 | 2704.28 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 2678.00 | 2699.51 | 2702.14 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 2766.40 | 2711.84 | 2707.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 13:15:00 | 2805.00 | 2777.99 | 2753.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 2781.60 | 2789.64 | 2768.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 10:15:00 | 2781.60 | 2789.64 | 2768.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 2781.60 | 2789.64 | 2768.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:30:00 | 2770.40 | 2789.64 | 2768.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 2772.80 | 2786.92 | 2770.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 2772.80 | 2786.92 | 2770.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 2776.30 | 2784.80 | 2771.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 2789.50 | 2784.28 | 2772.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 2785.00 | 2812.64 | 2813.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 2785.00 | 2812.64 | 2813.44 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 2820.30 | 2810.74 | 2810.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 2854.90 | 2820.89 | 2815.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 12:15:00 | 2899.40 | 2900.67 | 2887.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 14:15:00 | 2891.80 | 2897.96 | 2888.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 2891.80 | 2897.96 | 2888.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:30:00 | 2888.00 | 2897.96 | 2888.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 2899.90 | 2898.34 | 2889.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 2876.00 | 2898.34 | 2889.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 2906.10 | 2899.90 | 2891.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:30:00 | 2912.80 | 2904.54 | 2894.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 2915.20 | 2911.87 | 2900.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 2949.20 | 2956.32 | 2956.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 2944.40 | 2953.94 | 2955.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 2944.40 | 2953.94 | 2955.08 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 2970.00 | 2953.26 | 2952.48 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 2948.00 | 2957.46 | 2957.71 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 2967.70 | 2957.49 | 2957.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 3020.00 | 2972.72 | 2964.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 12:15:00 | 3002.00 | 3015.95 | 2996.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 13:00:00 | 3002.00 | 3015.95 | 2996.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 2979.90 | 3008.74 | 2994.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 2979.90 | 3008.74 | 2994.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 2975.00 | 3002.00 | 2993.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 2994.90 | 3002.00 | 2993.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:30:00 | 2981.30 | 3004.14 | 2998.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2938.00 | 2985.27 | 2990.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 2938.00 | 2985.27 | 2990.97 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 2975.00 | 2950.28 | 2949.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 2996.00 | 2959.42 | 2953.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2958.50 | 2981.22 | 2971.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2958.50 | 2981.22 | 2971.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2958.50 | 2981.22 | 2971.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 11:30:00 | 2963.00 | 2969.87 | 2967.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 12:15:00 | 2950.90 | 2966.08 | 2966.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 2950.90 | 2966.08 | 2966.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 2944.10 | 2961.68 | 2964.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 2993.30 | 2962.61 | 2963.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 2993.30 | 2962.61 | 2963.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2993.30 | 2962.61 | 2963.34 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 3000.70 | 2970.23 | 2966.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 3012.30 | 2992.92 | 2980.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 3036.90 | 3088.09 | 3062.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 3036.90 | 3088.09 | 3062.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3036.90 | 3088.09 | 3062.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:45:00 | 3044.50 | 3088.09 | 3062.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 3026.90 | 3075.85 | 3059.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 12:45:00 | 3040.90 | 3061.42 | 3055.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:00:00 | 3038.00 | 3056.74 | 3053.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 15:15:00 | 3030.00 | 3047.11 | 3049.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 3030.00 | 3047.11 | 3049.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 2991.30 | 3035.95 | 3044.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3054.50 | 2988.48 | 3007.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 3054.50 | 2988.48 | 3007.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3054.50 | 2988.48 | 3007.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3054.50 | 2988.48 | 3007.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3059.40 | 3002.66 | 3011.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:30:00 | 3063.70 | 3002.66 | 3011.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 3024.40 | 3013.58 | 3015.13 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 3024.90 | 3017.00 | 3016.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 3044.00 | 3022.40 | 3018.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3040.20 | 3052.79 | 3041.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3040.20 | 3052.79 | 3041.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3040.20 | 3052.79 | 3041.07 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 3001.00 | 3033.47 | 3033.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2940.80 | 3008.38 | 3021.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 3020.00 | 2937.91 | 2963.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 3020.00 | 2937.91 | 2963.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 3020.00 | 2937.91 | 2963.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 2854.30 | 2937.91 | 2963.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 2925.10 | 2882.33 | 2879.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 2925.10 | 2882.33 | 2879.28 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 2840.90 | 2876.01 | 2878.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 2804.20 | 2845.11 | 2860.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2860.20 | 2816.19 | 2830.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2860.20 | 2816.19 | 2830.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2860.20 | 2816.19 | 2830.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 2829.30 | 2820.05 | 2830.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 2818.20 | 2824.58 | 2831.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 2756.20 | 2740.72 | 2739.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 14:15:00 | 2756.20 | 2740.72 | 2739.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 2808.30 | 2755.72 | 2746.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 2883.00 | 2886.80 | 2855.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:00:00 | 2883.00 | 2886.80 | 2855.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 2830.20 | 2874.26 | 2857.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 2824.60 | 2874.26 | 2857.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 2824.30 | 2864.27 | 2854.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 2823.50 | 2864.27 | 2854.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 2822.70 | 2848.68 | 2848.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 13:15:00 | 2807.70 | 2827.93 | 2836.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 2798.20 | 2793.43 | 2808.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 2798.20 | 2793.43 | 2808.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 2798.20 | 2793.43 | 2808.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:45:00 | 2800.90 | 2793.43 | 2808.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 2793.10 | 2791.51 | 2803.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 2798.90 | 2791.51 | 2803.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 2797.60 | 2791.76 | 2800.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 2791.10 | 2791.76 | 2800.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 2755.00 | 2784.41 | 2796.37 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 2803.60 | 2784.53 | 2784.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 2815.00 | 2796.69 | 2790.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 2783.00 | 2798.14 | 2793.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 2783.00 | 2798.14 | 2793.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 2783.00 | 2798.14 | 2793.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:30:00 | 2783.00 | 2798.14 | 2793.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 2774.60 | 2793.43 | 2791.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 2774.60 | 2793.43 | 2791.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 2761.20 | 2786.98 | 2788.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 2752.20 | 2780.03 | 2785.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 2805.00 | 2775.70 | 2780.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 2805.00 | 2775.70 | 2780.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 2805.00 | 2775.70 | 2780.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 2816.50 | 2775.70 | 2780.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 2815.60 | 2783.68 | 2783.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 2833.30 | 2808.69 | 2798.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 2883.60 | 2884.24 | 2855.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 2883.60 | 2884.24 | 2855.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 2900.00 | 2892.53 | 2872.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 2975.70 | 2892.53 | 2872.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-27 14:15:00 | 2424.05 | 2024-05-29 09:15:00 | 2389.65 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-05-28 10:15:00 | 2427.65 | 2024-05-29 09:15:00 | 2389.65 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-05-31 09:15:00 | 2372.55 | 2024-06-04 09:15:00 | 2253.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-31 09:15:00 | 2372.55 | 2024-06-04 12:15:00 | 2135.30 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-11 12:30:00 | 2419.55 | 2024-06-11 14:15:00 | 2379.55 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-06-11 13:15:00 | 2419.70 | 2024-06-11 14:15:00 | 2379.55 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-06-13 09:45:00 | 2424.00 | 2024-06-14 14:15:00 | 2380.50 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-06-13 11:30:00 | 2422.45 | 2024-06-14 14:15:00 | 2380.50 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-06-19 09:15:00 | 2375.60 | 2024-07-01 14:15:00 | 2256.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-19 09:15:00 | 2375.60 | 2024-07-03 09:15:00 | 2259.15 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2024-07-11 10:15:00 | 2241.10 | 2024-07-15 13:15:00 | 2241.30 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-07-11 11:00:00 | 2243.15 | 2024-07-15 13:15:00 | 2241.30 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-07-15 11:45:00 | 2242.65 | 2024-07-15 13:15:00 | 2241.30 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-07-15 12:30:00 | 2242.40 | 2024-07-15 13:15:00 | 2241.30 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-07-22 11:15:00 | 2301.00 | 2024-07-30 14:15:00 | 2531.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 11:45:00 | 2300.60 | 2024-07-30 14:15:00 | 2530.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 12:30:00 | 2316.00 | 2024-07-30 14:15:00 | 2547.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 15:00:00 | 2301.85 | 2024-07-30 14:15:00 | 2532.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-23 09:45:00 | 2316.80 | 2024-07-30 14:15:00 | 2548.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-20 14:45:00 | 2940.85 | 2024-08-21 15:15:00 | 3080.00 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2024-09-04 09:15:00 | 3242.00 | 2024-09-13 12:15:00 | 3297.00 | STOP_HIT | 1.00 | 1.70% |
| SELL | retest2 | 2024-09-23 12:45:00 | 3139.80 | 2024-09-24 14:15:00 | 3235.70 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2024-09-23 13:15:00 | 3139.00 | 2024-09-24 14:15:00 | 3235.70 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-09-24 09:30:00 | 3130.15 | 2024-09-24 14:15:00 | 3235.70 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-09-24 10:15:00 | 3138.05 | 2024-09-24 14:15:00 | 3235.70 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-09-24 11:15:00 | 3106.45 | 2024-09-24 14:15:00 | 3235.70 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2024-10-04 11:30:00 | 3304.90 | 2024-10-10 15:15:00 | 3335.00 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2024-10-04 15:00:00 | 3315.95 | 2024-10-10 15:15:00 | 3335.00 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2024-10-08 09:30:00 | 3358.60 | 2024-10-10 15:15:00 | 3335.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-10-15 12:15:00 | 3254.75 | 2024-10-16 09:15:00 | 3374.50 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2024-10-15 13:45:00 | 3255.65 | 2024-10-16 09:15:00 | 3374.50 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2024-10-15 14:30:00 | 3257.50 | 2024-10-16 09:15:00 | 3374.50 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2024-10-28 09:15:00 | 2938.10 | 2024-10-30 12:15:00 | 3018.05 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-10-29 12:15:00 | 2947.85 | 2024-10-30 12:15:00 | 3018.05 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-11-04 11:15:00 | 3019.60 | 2024-11-04 12:15:00 | 2987.85 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-11-04 11:45:00 | 3020.80 | 2024-11-04 12:15:00 | 2987.85 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-12-03 09:15:00 | 3051.95 | 2024-12-03 11:15:00 | 2980.05 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-12-12 11:30:00 | 2807.20 | 2024-12-16 11:15:00 | 2841.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-12-12 12:15:00 | 2804.90 | 2024-12-16 11:15:00 | 2841.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-12-12 12:45:00 | 2806.65 | 2024-12-16 11:15:00 | 2841.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-12-12 14:00:00 | 2805.30 | 2024-12-16 11:15:00 | 2841.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-12-18 09:15:00 | 2879.95 | 2024-12-20 15:15:00 | 2819.90 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-12-19 09:45:00 | 2891.25 | 2024-12-20 15:15:00 | 2819.90 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-12-19 11:15:00 | 2882.55 | 2024-12-20 15:15:00 | 2819.90 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-12-19 13:00:00 | 2885.05 | 2024-12-20 15:15:00 | 2819.90 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest1 | 2024-12-30 12:15:00 | 2951.80 | 2025-01-02 09:15:00 | 2930.55 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2024-12-31 09:15:00 | 2954.00 | 2025-01-02 09:15:00 | 2930.55 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest1 | 2024-12-31 09:45:00 | 2951.50 | 2025-01-02 09:15:00 | 2930.55 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-01-01 10:30:00 | 2942.65 | 2025-01-03 09:15:00 | 2906.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-01-02 10:30:00 | 2944.05 | 2025-01-03 10:15:00 | 2899.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-01-02 11:15:00 | 2944.95 | 2025-01-03 10:15:00 | 2899.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-01-02 11:45:00 | 2943.70 | 2025-01-03 10:15:00 | 2899.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-01-02 14:15:00 | 2978.05 | 2025-01-03 10:15:00 | 2899.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-01-14 09:15:00 | 2767.00 | 2025-01-15 09:15:00 | 2851.05 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-01-14 10:00:00 | 2760.45 | 2025-01-15 09:15:00 | 2851.05 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-01-30 11:45:00 | 2658.15 | 2025-01-30 15:15:00 | 2674.45 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-01-30 13:00:00 | 2659.45 | 2025-01-30 15:15:00 | 2674.45 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-02-03 12:00:00 | 2752.00 | 2025-02-03 14:15:00 | 3027.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-24 12:30:00 | 2586.90 | 2025-03-03 09:15:00 | 2457.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 13:30:00 | 2587.00 | 2025-03-03 09:15:00 | 2457.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 2553.10 | 2025-03-03 09:15:00 | 2425.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 12:30:00 | 2586.90 | 2025-03-04 09:15:00 | 2467.10 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2025-02-24 13:30:00 | 2587.00 | 2025-03-04 09:15:00 | 2467.10 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2025-02-28 09:15:00 | 2553.10 | 2025-03-04 09:15:00 | 2467.10 | STOP_HIT | 0.50 | 3.37% |
| BUY | retest2 | 2025-03-10 10:15:00 | 2605.50 | 2025-03-10 12:15:00 | 2521.90 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-03-18 14:30:00 | 2614.55 | 2025-03-21 13:15:00 | 2876.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 15:15:00 | 2623.00 | 2025-03-21 13:15:00 | 2875.73 | TARGET_HIT | 1.00 | 9.64% |
| BUY | retest2 | 2025-03-19 12:15:00 | 2614.30 | 2025-03-21 13:15:00 | 2875.24 | TARGET_HIT | 1.00 | 9.98% |
| BUY | retest2 | 2025-03-19 12:45:00 | 2613.85 | 2025-03-27 09:15:00 | 2662.60 | STOP_HIT | 1.00 | 1.87% |
| SELL | retest2 | 2025-04-02 09:15:00 | 2600.45 | 2025-04-03 09:15:00 | 2663.35 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-04-04 09:15:00 | 2582.60 | 2025-04-07 09:15:00 | 2453.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 2582.60 | 2025-04-08 11:15:00 | 2421.65 | STOP_HIT | 0.50 | 6.23% |
| BUY | retest2 | 2025-04-22 10:45:00 | 2693.50 | 2025-04-25 14:15:00 | 2667.90 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-04-22 13:15:00 | 2695.00 | 2025-04-25 14:15:00 | 2667.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-04-22 14:00:00 | 2691.50 | 2025-04-25 14:15:00 | 2667.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-04-23 09:30:00 | 2696.20 | 2025-04-25 14:15:00 | 2667.90 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-04-24 10:45:00 | 2732.90 | 2025-04-25 14:15:00 | 2667.90 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-05-08 15:15:00 | 2486.20 | 2025-05-09 14:15:00 | 2538.20 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-05-09 09:45:00 | 2512.20 | 2025-05-09 14:15:00 | 2538.20 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-05-09 10:15:00 | 2512.10 | 2025-05-09 14:15:00 | 2538.20 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-05-14 14:45:00 | 2585.80 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-05-15 11:45:00 | 2597.80 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-05-15 13:00:00 | 2590.70 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-05-16 10:15:00 | 2589.00 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-05-21 09:30:00 | 2695.50 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2025-06-05 14:15:00 | 2575.10 | 2025-06-06 15:15:00 | 2572.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-06-05 14:45:00 | 2574.90 | 2025-06-06 15:15:00 | 2572.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-06-06 12:15:00 | 2575.50 | 2025-06-06 15:15:00 | 2572.00 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-06-06 13:00:00 | 2575.50 | 2025-06-06 15:15:00 | 2572.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-06-19 10:15:00 | 2547.80 | 2025-06-20 15:15:00 | 2590.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-06-19 11:30:00 | 2551.50 | 2025-06-20 15:15:00 | 2590.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-06-20 11:00:00 | 2550.80 | 2025-06-20 15:15:00 | 2590.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-06-27 13:15:00 | 2547.50 | 2025-06-30 15:15:00 | 2575.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-06-27 14:45:00 | 2535.70 | 2025-06-30 15:15:00 | 2575.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-30 12:00:00 | 2547.20 | 2025-06-30 15:15:00 | 2575.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-09 11:15:00 | 2618.50 | 2025-07-11 11:15:00 | 2634.80 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-10 09:30:00 | 2616.00 | 2025-07-11 11:15:00 | 2634.80 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-07-11 10:45:00 | 2613.50 | 2025-07-11 11:15:00 | 2634.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-08 11:15:00 | 2594.40 | 2025-08-11 10:15:00 | 2618.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-08 13:15:00 | 2592.60 | 2025-08-11 10:15:00 | 2618.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-08 13:45:00 | 2591.80 | 2025-08-11 10:15:00 | 2618.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-08-18 10:45:00 | 2716.50 | 2025-08-19 12:15:00 | 2655.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-09-01 10:15:00 | 2449.60 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-09-01 12:30:00 | 2450.10 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-09-01 14:45:00 | 2450.50 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-09-01 15:15:00 | 2448.70 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-09-02 15:15:00 | 2475.00 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-09-03 10:15:00 | 2451.00 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-09-17 11:15:00 | 2547.10 | 2025-09-17 14:15:00 | 2562.10 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-18 12:30:00 | 2547.00 | 2025-09-26 09:15:00 | 2419.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 15:15:00 | 2546.50 | 2025-09-26 09:15:00 | 2419.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:45:00 | 2544.00 | 2025-09-26 09:15:00 | 2416.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2519.20 | 2025-09-26 09:15:00 | 2400.55 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-18 12:30:00 | 2547.00 | 2025-09-30 10:15:00 | 2455.00 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-09-18 15:15:00 | 2546.50 | 2025-09-30 10:15:00 | 2455.00 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-09-19 10:45:00 | 2544.00 | 2025-09-30 10:15:00 | 2455.00 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2519.20 | 2025-09-30 10:15:00 | 2455.00 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2025-09-24 13:45:00 | 2524.00 | 2025-09-30 13:15:00 | 2393.24 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-09-24 14:45:00 | 2526.90 | 2025-09-30 13:15:00 | 2397.80 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-09-24 15:15:00 | 2520.00 | 2025-09-30 13:15:00 | 2394.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 10:30:00 | 2508.00 | 2025-09-30 13:15:00 | 2382.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 13:45:00 | 2524.00 | 2025-10-01 09:15:00 | 2428.40 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2025-09-24 14:45:00 | 2526.90 | 2025-10-01 09:15:00 | 2428.40 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2025-09-24 15:15:00 | 2520.00 | 2025-10-01 09:15:00 | 2428.40 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-09-25 10:30:00 | 2508.00 | 2025-10-01 09:15:00 | 2428.40 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2025-10-14 09:15:00 | 2397.90 | 2025-10-15 09:15:00 | 2429.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-10-15 12:15:00 | 2398.50 | 2025-10-16 09:15:00 | 2460.10 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-10-27 12:15:00 | 2398.40 | 2025-10-29 10:15:00 | 2435.10 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-28 11:45:00 | 2397.50 | 2025-10-29 10:15:00 | 2435.10 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-03 15:00:00 | 2542.90 | 2025-11-07 13:15:00 | 2523.10 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-04 09:30:00 | 2543.70 | 2025-11-07 13:15:00 | 2523.10 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-11 09:15:00 | 2516.80 | 2025-11-11 10:15:00 | 2562.20 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest1 | 2025-11-17 09:15:00 | 2500.00 | 2025-11-19 11:15:00 | 2495.70 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-11-25 11:15:00 | 2517.60 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 4.12% |
| BUY | retest2 | 2025-11-25 13:45:00 | 2517.30 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 4.14% |
| BUY | retest2 | 2025-11-25 14:15:00 | 2513.30 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2025-11-26 09:15:00 | 2519.70 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 4.04% |
| BUY | retest2 | 2025-11-26 10:15:00 | 2533.30 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.48% |
| BUY | retest2 | 2025-11-26 11:30:00 | 2540.10 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.20% |
| BUY | retest2 | 2025-11-26 13:15:00 | 2530.00 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-11-26 14:15:00 | 2530.70 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.58% |
| BUY | retest2 | 2025-11-27 12:15:00 | 2525.20 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.81% |
| SELL | retest2 | 2025-12-18 11:15:00 | 2565.00 | 2025-12-19 09:15:00 | 2670.00 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2025-12-18 12:30:00 | 2564.50 | 2025-12-19 09:15:00 | 2670.00 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-12-24 09:15:00 | 2766.50 | 2025-12-29 09:15:00 | 2685.50 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-12-29 09:15:00 | 2687.00 | 2025-12-29 09:15:00 | 2685.50 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-12-31 10:15:00 | 2745.60 | 2026-01-02 12:15:00 | 3020.16 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-16 13:00:00 | 2665.70 | 2026-01-21 12:15:00 | 2702.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-01-19 09:15:00 | 2664.40 | 2026-01-21 12:15:00 | 2702.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-01-19 15:15:00 | 2669.00 | 2026-01-21 12:15:00 | 2702.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-20 11:00:00 | 2674.90 | 2026-01-21 12:15:00 | 2702.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-01-28 12:15:00 | 2700.10 | 2026-01-28 14:15:00 | 2719.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-02-02 14:30:00 | 2789.50 | 2026-02-05 10:15:00 | 2785.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-02-13 10:30:00 | 2912.80 | 2026-02-20 10:15:00 | 2944.40 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2026-02-13 14:15:00 | 2915.20 | 2026-02-20 10:15:00 | 2944.40 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2026-02-20 09:30:00 | 2949.20 | 2026-02-20 10:15:00 | 2944.40 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-02-26 15:15:00 | 2994.90 | 2026-03-02 09:15:00 | 2938.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-27 13:30:00 | 2981.30 | 2026-03-02 09:15:00 | 2938.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-03-09 11:30:00 | 2963.00 | 2026-03-09 12:15:00 | 2950.90 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2026-03-13 12:45:00 | 3040.90 | 2026-03-13 15:15:00 | 3030.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2026-03-13 14:00:00 | 3038.00 | 2026-03-13 15:15:00 | 3030.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2854.30 | 2026-03-25 11:15:00 | 2925.10 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-04-01 10:45:00 | 2829.30 | 2026-04-09 14:15:00 | 2756.20 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2026-04-01 13:15:00 | 2818.20 | 2026-04-09 14:15:00 | 2756.20 | STOP_HIT | 1.00 | 2.20% |
