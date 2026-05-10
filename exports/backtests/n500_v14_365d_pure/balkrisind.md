# Balkrishna Industries Ltd. (BALKRISIND)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 2265.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 7 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 4
- **Target hits / Stop hits / Partials:** 2 / 6 / 3
- **Avg / median % per leg:** 2.49% / 4.22%
- **Sum % (uncompounded):** 27.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 10 | 6 | 60.0% | 1 | 6 | 3 | 1.74% | 17.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.82% | -5.8% |
| SELL @ 3rd Alert (retest2) | 9 | 6 | 66.7% | 1 | 5 | 3 | 2.58% | 23.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.82% | -5.8% |
| retest2 (combined) | 10 | 7 | 70.0% | 2 | 5 | 3 | 3.32% | 33.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 2750.00 | 2608.56 | 2608.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 2755.00 | 2611.40 | 2609.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 2635.30 | 2643.25 | 2627.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:00:00 | 2635.30 | 2643.25 | 2627.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 2630.30 | 2644.73 | 2629.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 2630.30 | 2644.73 | 2629.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 2647.10 | 2644.65 | 2629.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 2648.10 | 2644.65 | 2629.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 2634.00 | 2645.01 | 2630.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:00:00 | 2634.00 | 2645.01 | 2630.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2432.40 | 2643.35 | 2629.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 2412.90 | 2643.35 | 2629.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 13:15:00 | 2476.30 | 2617.13 | 2617.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 2448.70 | 2595.16 | 2605.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 2551.90 | 2550.72 | 2578.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:45:00 | 2547.90 | 2550.72 | 2578.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2569.60 | 2480.62 | 2523.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 2569.60 | 2480.62 | 2523.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2575.40 | 2481.57 | 2523.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 2566.00 | 2481.57 | 2523.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 2670.20 | 2551.44 | 2551.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 2724.20 | 2559.99 | 2555.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 2625.20 | 2648.61 | 2610.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 10:00:00 | 2625.20 | 2648.61 | 2610.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 2597.20 | 2648.10 | 2610.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 2597.20 | 2648.10 | 2610.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 2612.00 | 2647.74 | 2610.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 2599.50 | 2647.74 | 2610.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 2605.00 | 2647.32 | 2610.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 2605.00 | 2647.32 | 2610.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 2590.00 | 2646.75 | 2610.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:45:00 | 2594.00 | 2646.75 | 2610.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 2413.60 | 2582.03 | 2582.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 2402.20 | 2578.59 | 2580.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 2409.30 | 2407.80 | 2466.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 2409.30 | 2407.80 | 2466.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2460.90 | 2408.95 | 2462.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 2460.90 | 2408.95 | 2462.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 2455.20 | 2409.41 | 2462.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 2455.20 | 2409.41 | 2462.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 2452.70 | 2410.26 | 2462.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:45:00 | 2460.20 | 2410.26 | 2462.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 2468.30 | 2411.27 | 2462.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 2468.30 | 2411.27 | 2462.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 2470.00 | 2411.85 | 2462.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 2480.00 | 2411.85 | 2462.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 2484.00 | 2412.57 | 2462.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 2441.00 | 2440.09 | 2469.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:00:00 | 2455.40 | 2440.24 | 2469.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2332.63 | 2430.70 | 2461.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:15:00 | 2318.95 | 2429.54 | 2461.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-14 09:15:00 | 2209.86 | 2356.79 | 2409.92 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 2351.70 | 2327.75 | 2384.81 | SL hit (close>ema200) qty=0.50 sl=2327.75 alert=retest2 |

### Cycle 5 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 2413.30 | 2350.56 | 2350.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 2429.60 | 2357.25 | 2353.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 2376.50 | 2384.41 | 2370.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 2376.50 | 2384.41 | 2370.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 2376.50 | 2384.41 | 2370.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 2381.80 | 2384.41 | 2370.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 2356.30 | 2384.13 | 2370.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:45:00 | 2355.50 | 2384.13 | 2370.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 2367.90 | 2383.97 | 2370.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:45:00 | 2352.20 | 2383.97 | 2370.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 2362.80 | 2383.54 | 2370.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 2362.80 | 2383.54 | 2370.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 2365.00 | 2383.36 | 2370.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 2363.70 | 2383.36 | 2370.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 2339.00 | 2382.15 | 2369.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 2474.20 | 2366.65 | 2362.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 2721.62 | 2367.23 | 2363.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 2239.00 | 2423.36 | 2424.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 2233.50 | 2411.85 | 2418.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2221.40 | 2209.58 | 2287.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:45:00 | 2187.80 | 2209.99 | 2284.86 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 2277.20 | 2211.94 | 2282.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:30:00 | 2277.90 | 2211.94 | 2282.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 2281.80 | 2212.64 | 2282.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 2281.80 | 2212.64 | 2282.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 2271.10 | 2213.22 | 2282.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:00:00 | 2258.10 | 2213.66 | 2282.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 2262.00 | 2214.31 | 2282.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 2315.20 | 2219.34 | 2280.67 | SL hit (close>ema400) qty=1.00 sl=2280.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 2315.20 | 2219.34 | 2280.67 | SL hit (close>static) qty=1.00 sl=2285.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 2315.20 | 2219.34 | 2280.67 | SL hit (close>static) qty=1.00 sl=2285.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 2264.20 | 2238.77 | 2284.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 12:15:00 | 2302.00 | 2240.17 | 2284.34 | SL hit (close>static) qty=1.00 sl=2285.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:30:00 | 2259.90 | 2247.68 | 2285.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 2146.91 | 2238.53 | 2274.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 2236.00 | 2228.41 | 2264.84 | SL hit (close>ema200) qty=0.50 sl=2228.41 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 2269.70 | 2229.25 | 2264.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:45:00 | 2257.70 | 2229.25 | 2264.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 2267.90 | 2229.63 | 2264.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:30:00 | 2269.00 | 2229.63 | 2264.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 2271.00 | 2230.28 | 2264.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 2266.90 | 2230.28 | 2264.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 2285.60 | 2230.83 | 2264.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 2285.60 | 2230.83 | 2264.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 2266.00 | 2231.18 | 2264.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 2297.80 | 2231.18 | 2264.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 2265.10 | 2233.61 | 2265.05 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-24 09:15:00 | 2441.00 | 2025-09-29 09:15:00 | 2332.63 | PARTIAL | 0.50 | 4.44% |
| SELL | retest2 | 2025-09-24 10:00:00 | 2455.40 | 2025-09-29 10:15:00 | 2318.95 | PARTIAL | 0.50 | 5.56% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2441.00 | 2025-10-14 09:15:00 | 2209.86 | TARGET_HIT | 0.50 | 9.47% |
| SELL | retest2 | 2025-09-24 10:00:00 | 2455.40 | 2025-10-23 09:15:00 | 2351.70 | STOP_HIT | 0.50 | 4.22% |
| BUY | retest2 | 2026-02-03 09:15:00 | 2474.20 | 2026-02-03 09:15:00 | 2721.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-04-09 09:45:00 | 2187.80 | 2026-04-15 14:15:00 | 2315.20 | STOP_HIT | 1.00 | -5.82% |
| SELL | retest2 | 2026-04-10 14:00:00 | 2258.10 | 2026-04-15 14:15:00 | 2315.20 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-04-10 15:15:00 | 2262.00 | 2026-04-15 14:15:00 | 2315.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-04-21 10:15:00 | 2264.20 | 2026-04-21 12:15:00 | 2302.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-04-23 12:30:00 | 2259.90 | 2026-04-30 10:15:00 | 2146.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 12:30:00 | 2259.90 | 2026-05-06 15:15:00 | 2236.00 | STOP_HIT | 0.50 | 1.06% |
