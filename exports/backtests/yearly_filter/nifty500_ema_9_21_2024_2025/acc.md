# ACC Ltd. (ACC)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1393.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 120 |
| ALERT1 | 88 |
| ALERT2 | 84 |
| ALERT2_SKIP | 40 |
| ALERT3 | 241 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 148 |
| PARTIAL | 16 |
| TARGET_HIT | 0 |
| STOP_HIT | 150 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 165 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 104
- **Target hits / Stop hits / Partials:** 0 / 149 / 16
- **Avg / median % per leg:** 0.34% / -0.28%
- **Sum % (uncompounded):** 55.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 13 | 21.0% | 0 | 62 | 0 | -0.63% | -39.2% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.06% | 0.1% |
| BUY @ 3rd Alert (retest2) | 61 | 12 | 19.7% | 0 | 61 | 0 | -0.64% | -39.3% |
| SELL (all) | 103 | 48 | 46.6% | 0 | 87 | 16 | 0.92% | 95.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 103 | 48 | 46.6% | 0 | 87 | 16 | 0.92% | 95.1% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.06% | 0.1% |
| retest2 (combined) | 164 | 60 | 36.6% | 0 | 148 | 16 | 0.34% | 55.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 2452.75 | 2401.09 | 2399.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 2461.00 | 2413.08 | 2404.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 2466.90 | 2473.00 | 2457.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:30:00 | 2467.10 | 2473.00 | 2457.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 2506.90 | 2512.92 | 2496.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 2492.40 | 2512.92 | 2496.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 2514.50 | 2523.10 | 2511.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 2520.05 | 2523.10 | 2511.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 2508.65 | 2520.21 | 2511.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 2508.65 | 2520.21 | 2511.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 2532.15 | 2522.60 | 2513.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:15:00 | 2537.00 | 2522.60 | 2513.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:15:00 | 2537.30 | 2523.68 | 2514.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:00:00 | 2544.95 | 2535.69 | 2524.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 2573.45 | 2582.38 | 2583.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 2573.45 | 2582.38 | 2583.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 2535.00 | 2569.37 | 2576.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 2515.00 | 2513.19 | 2531.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 2515.00 | 2513.19 | 2531.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 2515.00 | 2513.19 | 2531.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 2536.05 | 2513.19 | 2531.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 2526.50 | 2514.81 | 2527.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:45:00 | 2532.75 | 2514.81 | 2527.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 2562.00 | 2524.25 | 2530.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 2562.00 | 2524.25 | 2530.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 2548.70 | 2529.14 | 2532.37 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 2633.00 | 2553.64 | 2543.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 15:15:00 | 2701.75 | 2644.23 | 2599.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 2553.65 | 2626.12 | 2595.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 2553.65 | 2626.12 | 2595.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 2553.65 | 2626.12 | 2595.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 2514.85 | 2626.12 | 2595.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 2414.55 | 2583.80 | 2579.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 2414.55 | 2583.80 | 2579.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 2280.40 | 2523.12 | 2552.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 15:15:00 | 2280.00 | 2393.31 | 2474.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 2385.85 | 2380.39 | 2453.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 2385.85 | 2380.39 | 2453.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 2483.00 | 2410.68 | 2436.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 2476.40 | 2410.68 | 2436.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 2493.90 | 2427.32 | 2441.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 2493.90 | 2427.32 | 2441.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 2461.00 | 2450.84 | 2450.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 2473.55 | 2455.39 | 2452.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 11:15:00 | 2655.80 | 2657.21 | 2636.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 11:30:00 | 2657.10 | 2657.21 | 2636.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 2611.05 | 2645.89 | 2639.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 2615.00 | 2645.89 | 2639.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 2621.40 | 2640.99 | 2637.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:15:00 | 2622.75 | 2640.99 | 2637.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 2629.35 | 2635.61 | 2635.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 2629.35 | 2635.61 | 2635.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 2618.55 | 2632.19 | 2634.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 13:15:00 | 2629.00 | 2627.28 | 2630.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 14:00:00 | 2629.00 | 2627.28 | 2630.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 2621.65 | 2625.20 | 2628.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 13:00:00 | 2603.35 | 2622.34 | 2626.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 2600.50 | 2619.82 | 2625.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 2587.85 | 2613.43 | 2621.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 2633.00 | 2595.97 | 2593.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 2633.00 | 2595.97 | 2593.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 2656.45 | 2609.65 | 2602.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 2603.80 | 2617.00 | 2609.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 2603.80 | 2617.00 | 2609.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 2603.80 | 2617.00 | 2609.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 2603.80 | 2617.00 | 2609.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 2640.25 | 2621.65 | 2611.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 2618.00 | 2621.65 | 2611.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 2619.75 | 2628.49 | 2619.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 2619.75 | 2628.49 | 2619.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 2619.80 | 2626.75 | 2619.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:45:00 | 2616.50 | 2626.75 | 2619.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 2620.00 | 2625.40 | 2619.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 2623.00 | 2625.40 | 2619.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 2616.20 | 2623.56 | 2618.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 2616.20 | 2623.56 | 2618.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 2615.05 | 2621.86 | 2618.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 2633.00 | 2621.86 | 2618.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 2721.40 | 2734.85 | 2735.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 2721.40 | 2734.85 | 2735.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 12:15:00 | 2691.65 | 2720.79 | 2728.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 2703.00 | 2697.98 | 2713.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-08 10:00:00 | 2703.00 | 2697.98 | 2713.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 2698.45 | 2673.59 | 2686.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 2698.45 | 2673.59 | 2686.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 2688.65 | 2676.60 | 2686.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 2673.00 | 2678.26 | 2686.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 2672.00 | 2657.47 | 2665.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:15:00 | 2675.15 | 2662.35 | 2666.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:00:00 | 2677.00 | 2665.28 | 2667.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 2668.95 | 2666.01 | 2667.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 2677.35 | 2669.24 | 2668.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 2677.35 | 2669.24 | 2668.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 12:15:00 | 2690.10 | 2674.98 | 2671.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 2695.55 | 2698.39 | 2688.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 15:00:00 | 2695.55 | 2698.39 | 2688.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 2708.50 | 2705.69 | 2696.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 13:15:00 | 2713.25 | 2705.69 | 2696.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:15:00 | 2710.40 | 2706.55 | 2697.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 10:15:00 | 2679.05 | 2701.42 | 2698.36 | SL hit (close<static) qty=1.00 sl=2695.30 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 2688.40 | 2696.61 | 2696.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 15:15:00 | 2675.50 | 2692.39 | 2694.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 2662.45 | 2642.44 | 2660.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 2662.45 | 2642.44 | 2660.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2662.45 | 2642.44 | 2660.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 2655.65 | 2642.44 | 2660.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2652.90 | 2644.53 | 2659.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 2663.30 | 2644.53 | 2659.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 2652.90 | 2646.20 | 2659.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 2652.90 | 2646.20 | 2659.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 2647.65 | 2645.08 | 2656.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 2647.65 | 2645.08 | 2656.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 2648.10 | 2644.10 | 2652.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 2602.40 | 2657.39 | 2657.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 2634.75 | 2655.46 | 2656.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 10:15:00 | 2638.60 | 2652.03 | 2654.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:15:00 | 2635.05 | 2650.27 | 2653.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 2628.00 | 2605.00 | 2619.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 2628.00 | 2605.00 | 2619.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 2636.00 | 2611.20 | 2621.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:45:00 | 2638.10 | 2611.20 | 2621.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 2632.50 | 2615.46 | 2622.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:15:00 | 2633.00 | 2615.46 | 2622.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 2616.90 | 2619.70 | 2622.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 2640.75 | 2624.60 | 2624.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 2640.75 | 2624.60 | 2624.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 2659.40 | 2631.56 | 2627.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 2603.20 | 2634.85 | 2631.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 14:15:00 | 2603.20 | 2634.85 | 2631.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 2603.20 | 2634.85 | 2631.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:30:00 | 2596.00 | 2634.85 | 2631.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 2614.25 | 2630.73 | 2630.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 2601.30 | 2630.73 | 2630.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 2583.85 | 2621.35 | 2625.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 10:15:00 | 2520.45 | 2570.80 | 2589.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2422.00 | 2402.49 | 2441.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 2422.00 | 2402.49 | 2441.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 2397.50 | 2385.65 | 2398.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 2377.60 | 2385.65 | 2398.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 15:15:00 | 2339.80 | 2316.85 | 2313.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 2339.80 | 2316.85 | 2313.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 10:15:00 | 2341.50 | 2325.14 | 2318.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 2335.80 | 2340.86 | 2331.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:45:00 | 2337.55 | 2340.86 | 2331.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 2320.00 | 2336.69 | 2330.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:00:00 | 2320.00 | 2336.69 | 2330.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 2325.90 | 2334.53 | 2330.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 2339.40 | 2330.83 | 2329.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 12:15:00 | 2322.00 | 2328.52 | 2328.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 12:15:00 | 2322.00 | 2328.52 | 2328.71 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 10:15:00 | 2336.95 | 2329.18 | 2328.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 11:15:00 | 2358.55 | 2335.05 | 2331.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 2339.90 | 2343.49 | 2337.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 2339.90 | 2343.49 | 2337.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2339.90 | 2343.49 | 2337.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 2339.90 | 2343.49 | 2337.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 2326.55 | 2340.10 | 2336.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 2326.55 | 2340.10 | 2336.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 2323.85 | 2336.85 | 2335.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 2323.85 | 2336.85 | 2335.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 2322.05 | 2333.89 | 2334.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 2317.20 | 2330.55 | 2332.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 2335.00 | 2330.02 | 2331.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 2335.00 | 2330.02 | 2331.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 2335.00 | 2330.02 | 2331.84 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 2343.25 | 2334.08 | 2333.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 15:15:00 | 2344.00 | 2336.06 | 2334.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 2341.90 | 2343.57 | 2339.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 2341.90 | 2343.57 | 2339.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 2341.90 | 2343.57 | 2339.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 2340.45 | 2343.57 | 2339.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 2345.10 | 2343.88 | 2340.45 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 2333.00 | 2339.23 | 2339.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 2308.55 | 2333.09 | 2336.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 2324.70 | 2314.40 | 2322.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 2324.70 | 2314.40 | 2322.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 2324.70 | 2314.40 | 2322.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 2323.20 | 2314.40 | 2322.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 2333.30 | 2318.18 | 2323.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 2333.30 | 2318.18 | 2323.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 2328.65 | 2320.27 | 2323.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:30:00 | 2335.15 | 2320.27 | 2323.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 15:15:00 | 2331.00 | 2326.66 | 2326.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 2334.65 | 2328.64 | 2327.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 2382.25 | 2403.13 | 2381.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 2382.25 | 2403.13 | 2381.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 2382.25 | 2403.13 | 2381.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 2382.25 | 2403.13 | 2381.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 2405.75 | 2403.66 | 2383.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:30:00 | 2422.20 | 2410.73 | 2388.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:45:00 | 2421.70 | 2420.10 | 2403.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 10:30:00 | 2419.00 | 2422.16 | 2405.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 2471.85 | 2495.37 | 2495.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 2471.85 | 2495.37 | 2495.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 2466.15 | 2489.53 | 2493.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 2468.15 | 2445.64 | 2453.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 2468.15 | 2445.64 | 2453.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 2468.15 | 2445.64 | 2453.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 2466.25 | 2445.64 | 2453.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 2461.15 | 2448.74 | 2454.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:30:00 | 2468.90 | 2448.74 | 2454.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 2484.55 | 2461.14 | 2459.32 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 2457.95 | 2466.06 | 2466.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 2439.95 | 2459.07 | 2463.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 12:15:00 | 2453.55 | 2452.74 | 2458.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 13:00:00 | 2453.55 | 2452.74 | 2458.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 2473.05 | 2457.12 | 2459.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 2473.05 | 2457.12 | 2459.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 2474.90 | 2460.68 | 2461.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 2485.05 | 2460.68 | 2461.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 2503.50 | 2469.24 | 2465.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 09:15:00 | 2528.00 | 2493.23 | 2481.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 2509.00 | 2509.65 | 2496.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:15:00 | 2502.50 | 2509.65 | 2496.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 2496.40 | 2507.00 | 2496.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 2496.40 | 2507.00 | 2496.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 2492.25 | 2504.05 | 2496.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:45:00 | 2492.55 | 2504.05 | 2496.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 2504.05 | 2504.05 | 2497.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 13:00:00 | 2513.10 | 2505.86 | 2498.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 14:45:00 | 2510.20 | 2508.50 | 2501.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:45:00 | 2512.85 | 2506.66 | 2502.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 2471.30 | 2499.59 | 2499.28 | SL hit (close<static) qty=1.00 sl=2492.25 alert=retest2 |

### Cycle 24 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 2461.85 | 2492.04 | 2495.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 2446.00 | 2482.83 | 2491.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 2473.50 | 2473.42 | 2483.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:45:00 | 2475.90 | 2473.42 | 2483.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 2388.20 | 2374.35 | 2392.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 2377.05 | 2374.35 | 2392.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2376.30 | 2374.74 | 2390.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 2363.50 | 2374.34 | 2387.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 2363.15 | 2372.10 | 2385.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 2245.32 | 2272.48 | 2285.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 2244.99 | 2272.48 | 2285.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 13:15:00 | 2281.95 | 2272.00 | 2280.65 | SL hit (close>ema200) qty=0.50 sl=2272.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 11:15:00 | 2299.00 | 2284.46 | 2284.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 12:15:00 | 2315.00 | 2290.57 | 2286.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 2286.90 | 2295.15 | 2290.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 2286.90 | 2295.15 | 2290.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 2286.90 | 2295.15 | 2290.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 2286.90 | 2295.15 | 2290.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 2277.80 | 2291.68 | 2289.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 2273.10 | 2291.68 | 2289.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 2273.75 | 2288.10 | 2288.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 2262.20 | 2282.92 | 2285.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 2274.60 | 2268.89 | 2276.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 10:15:00 | 2274.60 | 2268.89 | 2276.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 2274.60 | 2268.89 | 2276.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 2274.60 | 2268.89 | 2276.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 2269.40 | 2268.99 | 2275.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 2286.00 | 2268.99 | 2275.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 2271.25 | 2269.44 | 2275.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 2271.25 | 2269.44 | 2275.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 2264.65 | 2268.48 | 2274.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 2272.70 | 2268.48 | 2274.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 2280.90 | 2267.98 | 2272.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 2280.90 | 2267.98 | 2272.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 2273.50 | 2269.08 | 2272.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:30:00 | 2266.70 | 2266.74 | 2271.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 15:00:00 | 2269.00 | 2261.48 | 2267.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:30:00 | 2260.60 | 2244.59 | 2248.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 12:15:00 | 2293.05 | 2254.28 | 2252.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 2293.05 | 2254.28 | 2252.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 2297.85 | 2262.99 | 2256.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 10:15:00 | 2277.55 | 2278.95 | 2267.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 11:00:00 | 2277.55 | 2278.95 | 2267.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 2320.00 | 2333.80 | 2317.47 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 2285.15 | 2311.90 | 2313.97 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 2320.55 | 2306.69 | 2306.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 2338.70 | 2315.38 | 2310.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 2302.05 | 2337.10 | 2327.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 2302.05 | 2337.10 | 2327.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 2302.05 | 2337.10 | 2327.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 2302.05 | 2337.10 | 2327.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 2331.40 | 2335.96 | 2327.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 2311.75 | 2335.96 | 2327.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 2322.10 | 2333.19 | 2327.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 2322.10 | 2333.19 | 2327.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 2325.95 | 2331.74 | 2327.26 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 2309.05 | 2322.57 | 2324.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 2292.55 | 2316.57 | 2321.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 2281.10 | 2280.06 | 2292.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:15:00 | 2290.00 | 2280.06 | 2292.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 2282.35 | 2280.52 | 2291.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 2287.75 | 2280.52 | 2291.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 2224.20 | 2201.44 | 2210.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 2224.20 | 2201.44 | 2210.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 2221.90 | 2205.53 | 2211.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 2226.95 | 2205.53 | 2211.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 2216.50 | 2209.02 | 2212.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:00:00 | 2216.50 | 2209.02 | 2212.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 2210.85 | 2209.39 | 2212.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:30:00 | 2216.70 | 2209.39 | 2212.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 2185.05 | 2204.52 | 2209.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 1967.15 | 2200.58 | 2207.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 1868.79 | 2147.27 | 2182.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 2068.10 | 2056.68 | 2107.51 | SL hit (close>ema200) qty=0.50 sl=2056.68 alert=retest2 |

### Cycle 31 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 2152.25 | 2117.37 | 2114.89 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 2116.00 | 2121.48 | 2121.81 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 2140.60 | 2125.30 | 2123.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 11:15:00 | 2155.25 | 2132.11 | 2126.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 2185.05 | 2191.39 | 2172.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 15:00:00 | 2185.05 | 2191.39 | 2172.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 2249.95 | 2265.94 | 2248.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:00:00 | 2249.95 | 2265.94 | 2248.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2251.95 | 2263.14 | 2248.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 2246.95 | 2263.14 | 2248.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 2249.70 | 2260.45 | 2248.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:00:00 | 2249.70 | 2260.45 | 2248.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 2245.00 | 2257.36 | 2248.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:00:00 | 2245.00 | 2257.36 | 2248.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 2239.70 | 2253.83 | 2247.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:45:00 | 2239.00 | 2253.83 | 2247.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2260.00 | 2251.90 | 2247.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 11:30:00 | 2264.00 | 2255.90 | 2250.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:00:00 | 2264.80 | 2257.68 | 2251.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 14:15:00 | 2268.20 | 2257.72 | 2251.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 2268.80 | 2259.93 | 2253.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 2259.05 | 2264.00 | 2259.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:45:00 | 2260.00 | 2264.00 | 2259.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 2257.35 | 2262.67 | 2258.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:30:00 | 2257.80 | 2262.67 | 2258.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 2257.05 | 2261.54 | 2258.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 2248.00 | 2258.84 | 2257.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 2263.85 | 2259.84 | 2258.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 2249.45 | 2258.65 | 2258.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 2249.45 | 2258.65 | 2258.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 2242.60 | 2255.44 | 2257.46 | Break + close below crossover candle low |

### Cycle 35 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 2282.00 | 2257.61 | 2257.33 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 2248.70 | 2257.32 | 2257.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 2231.20 | 2251.60 | 2255.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 10:15:00 | 2258.25 | 2252.93 | 2255.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 10:15:00 | 2258.25 | 2252.93 | 2255.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 2258.25 | 2252.93 | 2255.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:45:00 | 2258.70 | 2252.93 | 2255.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 2246.35 | 2251.62 | 2254.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 13:15:00 | 2241.45 | 2250.66 | 2253.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:00:00 | 2241.20 | 2240.04 | 2241.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:15:00 | 2241.70 | 2240.78 | 2241.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 14:15:00 | 2246.55 | 2243.02 | 2242.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 2246.55 | 2243.02 | 2242.56 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 2231.95 | 2240.60 | 2241.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 2215.25 | 2234.66 | 2238.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 11:15:00 | 2124.05 | 2122.74 | 2145.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 12:00:00 | 2124.05 | 2122.74 | 2145.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 2096.80 | 2096.82 | 2107.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 2102.25 | 2096.82 | 2107.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 2090.80 | 2086.24 | 2093.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:45:00 | 2090.10 | 2086.24 | 2093.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 2092.90 | 2087.57 | 2093.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 2085.45 | 2089.68 | 2093.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:30:00 | 2083.85 | 2083.81 | 2086.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 2083.00 | 2083.81 | 2086.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 2078.25 | 2061.21 | 2060.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 2078.25 | 2061.21 | 2060.16 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 2052.75 | 2061.61 | 2061.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 2038.00 | 2054.75 | 2058.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 2014.35 | 2009.10 | 2028.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 2014.35 | 2009.10 | 2028.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 2014.35 | 2009.10 | 2028.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 2006.50 | 2009.10 | 2028.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 2020.90 | 2011.06 | 2021.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 2020.90 | 2011.06 | 2021.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 2015.75 | 2012.00 | 2021.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 2009.00 | 2012.00 | 2021.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2025.50 | 2014.70 | 2021.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 15:15:00 | 2005.60 | 2013.77 | 2018.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1905.32 | 1937.69 | 1961.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 1907.20 | 1893.46 | 1922.71 | SL hit (close>ema200) qty=0.50 sl=1893.46 alert=retest2 |

### Cycle 41 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 1956.85 | 1933.06 | 1931.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 2030.85 | 1969.84 | 1952.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 14:15:00 | 1986.95 | 1988.85 | 1970.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 15:00:00 | 1986.95 | 1988.85 | 1970.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1997.35 | 1990.04 | 1979.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:15:00 | 2005.15 | 1990.04 | 1979.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:45:00 | 2003.35 | 1998.93 | 1987.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:30:00 | 2003.05 | 2001.18 | 1990.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 09:45:00 | 2008.30 | 2018.81 | 2010.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 2002.80 | 2015.61 | 2009.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 2002.80 | 2015.61 | 2009.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 1986.10 | 2009.71 | 2007.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 1986.10 | 2009.71 | 2007.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-22 12:15:00 | 1981.60 | 2004.09 | 2005.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 1981.60 | 2004.09 | 2005.38 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 2021.35 | 2007.91 | 2006.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 2044.75 | 2020.34 | 2012.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 2035.95 | 2044.02 | 2033.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 2035.95 | 2044.02 | 2033.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 2035.95 | 2044.02 | 2033.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:30:00 | 2027.10 | 2044.02 | 2033.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 2034.15 | 2042.05 | 2033.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 11:15:00 | 2045.10 | 2042.05 | 2033.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 14:15:00 | 1999.75 | 2036.12 | 2033.81 | SL hit (close<static) qty=1.00 sl=2023.65 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1996.00 | 2028.10 | 2030.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1976.25 | 2017.73 | 2025.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 2040.65 | 2017.83 | 2023.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 2040.65 | 2017.83 | 2023.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 2040.65 | 2017.83 | 2023.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 2040.65 | 2017.83 | 2023.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 2023.00 | 2018.86 | 2023.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 2005.70 | 2015.49 | 2021.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 10:30:00 | 2009.10 | 2012.43 | 2018.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:00:00 | 2006.35 | 2012.43 | 2018.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:45:00 | 2004.80 | 2010.74 | 2016.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 2015.90 | 2011.78 | 2016.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:45:00 | 2028.05 | 2011.78 | 2016.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 2005.40 | 2010.50 | 2015.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:30:00 | 2015.85 | 2010.50 | 2015.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1984.25 | 2003.42 | 2011.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:45:00 | 1966.25 | 1995.77 | 2006.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 1966.40 | 1987.06 | 2000.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 2008.25 | 2000.05 | 1999.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 15:15:00 | 2008.25 | 2000.05 | 1999.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 2021.75 | 2004.39 | 2001.87 | Break + close above crossover candle high |

### Cycle 46 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1969.35 | 1999.75 | 2000.66 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 2038.50 | 2003.16 | 1998.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 2056.75 | 2030.26 | 2016.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 2015.55 | 2037.15 | 2027.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 2015.55 | 2037.15 | 2027.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 2015.55 | 2037.15 | 2027.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:45:00 | 2013.70 | 2037.15 | 2027.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 2014.35 | 2032.59 | 2026.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 2014.35 | 2032.59 | 2026.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 1995.00 | 2019.07 | 2021.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 1988.10 | 2012.88 | 2018.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 2001.80 | 1999.72 | 2006.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 15:00:00 | 2001.80 | 1999.72 | 2006.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1977.10 | 1995.12 | 2003.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:45:00 | 1972.85 | 1990.27 | 2000.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1874.21 | 1931.30 | 1955.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 11:15:00 | 1930.90 | 1929.55 | 1950.70 | SL hit (close>ema200) qty=0.50 sl=1929.55 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 1889.20 | 1878.52 | 1878.38 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1870.25 | 1877.41 | 1877.94 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 13:15:00 | 1884.60 | 1879.11 | 1878.49 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 1863.10 | 1877.06 | 1877.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 1848.85 | 1869.39 | 1873.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 1827.00 | 1826.64 | 1839.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 1827.00 | 1826.64 | 1839.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 1786.95 | 1809.60 | 1821.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 10:30:00 | 1783.00 | 1805.77 | 1818.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 14:15:00 | 1829.90 | 1821.00 | 1820.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 1829.90 | 1821.00 | 1820.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 1850.30 | 1828.98 | 1824.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1862.00 | 1867.37 | 1856.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 11:45:00 | 1871.40 | 1867.37 | 1856.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1900.55 | 1880.79 | 1867.53 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 1833.60 | 1866.31 | 1869.69 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 1868.10 | 1863.48 | 1863.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 1877.25 | 1867.20 | 1864.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 12:15:00 | 1880.50 | 1881.60 | 1874.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 12:45:00 | 1882.45 | 1881.60 | 1874.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 1893.80 | 1895.25 | 1888.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:45:00 | 1894.80 | 1895.25 | 1888.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 1896.85 | 1894.68 | 1889.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 1899.80 | 1892.12 | 1889.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:30:00 | 1900.50 | 1893.10 | 1890.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 11:15:00 | 1901.30 | 1893.10 | 1890.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:00:00 | 1901.75 | 1894.83 | 1891.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1925.35 | 1936.65 | 1928.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 1947.45 | 1936.65 | 1928.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1948.00 | 1938.92 | 1929.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 1954.00 | 1940.54 | 1931.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 1952.60 | 1939.52 | 1936.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 11:30:00 | 1950.20 | 1950.09 | 1944.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 1956.20 | 1943.19 | 1942.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1947.05 | 1943.96 | 1942.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:00:00 | 1964.65 | 1952.63 | 1948.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:00:00 | 1967.90 | 1974.96 | 1970.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1892.00 | 1956.30 | 1963.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1892.00 | 1956.30 | 1963.02 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 1979.00 | 1953.38 | 1951.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 13:15:00 | 1985.00 | 1970.83 | 1963.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2049.10 | 2050.55 | 2033.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 2049.10 | 2050.55 | 2033.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2056.00 | 2079.24 | 2074.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 2056.00 | 2079.24 | 2074.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 2055.00 | 2074.39 | 2073.13 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 2058.30 | 2071.17 | 2071.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 13:15:00 | 2050.20 | 2065.64 | 2069.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 2087.40 | 2066.32 | 2068.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 2087.40 | 2066.32 | 2068.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 2087.40 | 2066.32 | 2068.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 2087.40 | 2066.32 | 2068.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 10:15:00 | 2091.70 | 2071.39 | 2070.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 13:15:00 | 2110.00 | 2084.38 | 2076.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 14:15:00 | 2064.50 | 2080.40 | 2075.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 14:15:00 | 2064.50 | 2080.40 | 2075.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 2064.50 | 2080.40 | 2075.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 2064.50 | 2080.40 | 2075.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 2065.00 | 2077.32 | 2074.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 2022.60 | 2077.32 | 2074.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1959.90 | 2053.84 | 2064.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1943.50 | 2031.77 | 2053.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 1892.70 | 1876.41 | 1890.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 10:15:00 | 1892.70 | 1876.41 | 1890.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1892.70 | 1876.41 | 1890.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 1892.70 | 1876.41 | 1890.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1897.80 | 1880.69 | 1890.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1897.80 | 1880.69 | 1890.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1899.90 | 1884.53 | 1891.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:30:00 | 1903.00 | 1884.53 | 1891.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1886.60 | 1886.19 | 1891.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 1876.70 | 1886.15 | 1890.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1782.87 | 1824.13 | 1840.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 1810.10 | 1809.95 | 1824.72 | SL hit (close>ema200) qty=0.50 sl=1809.95 alert=retest2 |

### Cycle 61 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1854.00 | 1833.14 | 1832.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1864.70 | 1853.62 | 1844.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 1853.20 | 1857.44 | 1849.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 1853.20 | 1857.44 | 1849.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1920.00 | 1934.75 | 1924.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 1920.00 | 1934.75 | 1924.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1920.50 | 1931.90 | 1924.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 1924.60 | 1931.90 | 1924.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1939.60 | 1933.44 | 1925.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 1941.70 | 1932.18 | 1926.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:15:00 | 1942.00 | 1933.84 | 1927.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 1942.20 | 1935.51 | 1929.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 1941.90 | 1935.67 | 1930.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1928.80 | 1934.29 | 1930.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 1928.80 | 1934.29 | 1930.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1932.70 | 1933.98 | 1930.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 1929.50 | 1933.98 | 1930.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1935.40 | 1934.26 | 1930.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:00:00 | 1937.50 | 1934.91 | 1931.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:30:00 | 1939.50 | 1937.07 | 1932.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:30:00 | 1937.00 | 1951.98 | 1950.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 1937.70 | 1951.98 | 1950.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 1926.30 | 1933.89 | 1939.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 1926.90 | 1924.59 | 1931.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 1926.90 | 1924.59 | 1931.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1882.80 | 1884.24 | 1894.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:45:00 | 1872.10 | 1880.22 | 1889.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 13:30:00 | 1875.00 | 1879.36 | 1888.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 1875.00 | 1879.36 | 1888.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1872.30 | 1879.30 | 1886.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1878.00 | 1874.66 | 1880.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 1878.00 | 1874.66 | 1880.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1885.70 | 1876.92 | 1880.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 1887.00 | 1876.92 | 1880.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1891.10 | 1879.76 | 1881.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:00:00 | 1891.10 | 1879.76 | 1881.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 1890.90 | 1883.88 | 1883.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1890.90 | 1883.88 | 1883.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 1897.20 | 1888.42 | 1885.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 1913.30 | 1913.45 | 1906.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 15:00:00 | 1913.30 | 1913.45 | 1906.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1898.80 | 1912.24 | 1909.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1899.10 | 1912.24 | 1909.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1904.60 | 1910.71 | 1908.92 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 1890.00 | 1904.32 | 1906.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1886.60 | 1900.77 | 1904.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1854.80 | 1851.30 | 1865.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 1854.80 | 1851.30 | 1865.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1862.20 | 1853.48 | 1864.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 1862.00 | 1853.48 | 1864.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1871.50 | 1857.09 | 1865.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1871.50 | 1857.09 | 1865.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1874.30 | 1860.53 | 1866.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1874.30 | 1860.53 | 1866.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1870.10 | 1862.44 | 1866.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1870.70 | 1862.44 | 1866.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1863.00 | 1863.14 | 1866.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:15:00 | 1872.80 | 1863.14 | 1866.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1868.80 | 1864.27 | 1866.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:30:00 | 1872.90 | 1864.27 | 1866.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 1863.70 | 1864.16 | 1866.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 1866.50 | 1864.16 | 1866.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 1865.70 | 1864.47 | 1866.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 1865.00 | 1864.47 | 1866.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 1859.70 | 1863.51 | 1865.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 1857.00 | 1863.51 | 1865.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 1852.80 | 1861.39 | 1863.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1858.80 | 1831.28 | 1831.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1858.80 | 1831.28 | 1831.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 1869.80 | 1838.98 | 1834.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1846.20 | 1846.32 | 1839.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1846.20 | 1846.32 | 1839.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1914.80 | 1918.23 | 1909.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 1911.10 | 1918.23 | 1909.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1951.50 | 1961.37 | 1953.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 1951.50 | 1961.37 | 1953.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1947.00 | 1958.50 | 1953.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 1945.80 | 1958.50 | 1953.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1957.20 | 1960.03 | 1956.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1957.20 | 1960.03 | 1956.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1949.90 | 1958.00 | 1955.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 1949.90 | 1958.00 | 1955.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1960.30 | 1958.46 | 1956.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:30:00 | 1965.50 | 1960.47 | 1957.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1971.40 | 1986.30 | 1987.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 10:15:00 | 1971.40 | 1986.30 | 1987.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 11:15:00 | 1970.00 | 1983.04 | 1985.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1986.00 | 1980.25 | 1982.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1986.00 | 1980.25 | 1982.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1986.00 | 1980.25 | 1982.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 1986.50 | 1980.25 | 1982.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1991.00 | 1982.40 | 1983.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1991.00 | 1982.40 | 1983.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1986.80 | 1983.28 | 1983.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 1983.90 | 1982.61 | 1983.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 1985.00 | 1979.98 | 1981.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 1993.00 | 1983.77 | 1983.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1993.00 | 1983.77 | 1983.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 14:15:00 | 1996.50 | 1986.32 | 1984.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1980.30 | 1985.27 | 1984.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1980.30 | 1985.27 | 1984.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1980.30 | 1985.27 | 1984.16 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 1979.70 | 1983.31 | 1983.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1973.20 | 1979.16 | 1981.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 15:15:00 | 1973.30 | 1973.02 | 1976.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 1975.40 | 1973.02 | 1976.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1974.60 | 1973.34 | 1976.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 10:30:00 | 1966.30 | 1971.27 | 1975.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 1969.00 | 1975.12 | 1975.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1970.10 | 1974.08 | 1975.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 1871.59 | 1913.12 | 1933.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 1867.98 | 1903.32 | 1927.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 1870.55 | 1903.32 | 1927.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1824.90 | 1813.94 | 1825.66 | SL hit (close>ema200) qty=0.50 sl=1813.94 alert=retest2 |

### Cycle 69 — BUY (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 09:15:00 | 1828.00 | 1804.21 | 1802.31 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 1802.70 | 1808.74 | 1808.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 1799.70 | 1806.93 | 1808.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 1790.50 | 1790.49 | 1795.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1790.50 | 1790.49 | 1795.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1790.50 | 1790.49 | 1795.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 1783.10 | 1789.89 | 1792.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 12:15:00 | 1785.00 | 1789.89 | 1792.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 1785.70 | 1789.87 | 1792.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 1785.20 | 1789.46 | 1791.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1791.30 | 1789.14 | 1791.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 1785.90 | 1788.72 | 1790.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 1785.80 | 1787.16 | 1789.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:45:00 | 1784.60 | 1787.11 | 1789.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 14:15:00 | 1785.10 | 1787.11 | 1789.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 1848.20 | 1820.12 | 1805.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 1859.50 | 1860.79 | 1850.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:45:00 | 1859.60 | 1860.79 | 1850.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1848.00 | 1857.96 | 1853.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:45:00 | 1848.90 | 1857.96 | 1853.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1850.00 | 1856.37 | 1852.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1845.30 | 1856.37 | 1852.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1833.90 | 1849.85 | 1850.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 1828.70 | 1845.62 | 1848.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1805.60 | 1804.89 | 1814.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 1805.60 | 1804.89 | 1814.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1804.30 | 1803.54 | 1809.22 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1817.00 | 1809.43 | 1809.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1824.20 | 1816.09 | 1812.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1814.90 | 1821.35 | 1816.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1814.90 | 1821.35 | 1816.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1814.90 | 1821.35 | 1816.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1814.90 | 1821.35 | 1816.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1824.00 | 1821.88 | 1817.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 1819.50 | 1821.88 | 1817.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1815.00 | 1820.50 | 1817.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1825.20 | 1820.50 | 1817.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 1832.70 | 1823.22 | 1818.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 12:15:00 | 1827.50 | 1833.56 | 1833.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 1827.50 | 1833.56 | 1833.98 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 1844.40 | 1834.15 | 1833.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 1850.00 | 1841.80 | 1838.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 1847.80 | 1849.32 | 1846.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 14:15:00 | 1847.80 | 1849.32 | 1846.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1847.80 | 1849.32 | 1846.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 1846.10 | 1849.32 | 1846.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1850.40 | 1849.54 | 1846.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1852.00 | 1849.54 | 1846.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1840.00 | 1847.98 | 1847.06 | SL hit (close<static) qty=1.00 sl=1846.70 alert=retest2 |

### Cycle 76 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 1837.20 | 1845.82 | 1846.16 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1854.40 | 1847.10 | 1846.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 1860.00 | 1851.68 | 1849.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 14:15:00 | 1866.00 | 1866.69 | 1861.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 15:00:00 | 1866.00 | 1866.69 | 1861.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1863.10 | 1866.02 | 1861.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 1865.40 | 1866.02 | 1861.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1861.90 | 1865.20 | 1861.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:45:00 | 1862.40 | 1865.20 | 1861.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1860.80 | 1864.32 | 1861.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 1860.80 | 1864.32 | 1861.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1858.90 | 1863.23 | 1861.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:30:00 | 1857.10 | 1863.23 | 1861.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 1854.80 | 1859.24 | 1859.78 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 1864.00 | 1860.11 | 1860.01 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 1850.00 | 1858.08 | 1859.10 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 1869.40 | 1860.76 | 1859.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 1875.50 | 1865.27 | 1862.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1886.00 | 1889.99 | 1879.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:45:00 | 1884.90 | 1889.99 | 1879.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1884.00 | 1888.80 | 1879.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 1891.20 | 1888.80 | 1879.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 1893.60 | 1889.76 | 1880.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 1890.80 | 1888.29 | 1881.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:15:00 | 1887.10 | 1886.94 | 1881.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1879.40 | 1885.02 | 1881.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 1877.50 | 1885.02 | 1881.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1879.50 | 1883.92 | 1881.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 1868.30 | 1883.92 | 1881.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1876.60 | 1882.46 | 1881.02 | SL hit (close<static) qty=1.00 sl=1878.30 alert=retest2 |

### Cycle 82 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1873.60 | 1878.96 | 1879.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 1864.50 | 1874.62 | 1877.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 1837.00 | 1834.67 | 1844.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 1837.00 | 1834.67 | 1844.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1825.50 | 1826.10 | 1836.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 1831.90 | 1826.10 | 1836.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1840.50 | 1828.50 | 1832.35 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1844.70 | 1835.00 | 1834.63 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 15:15:00 | 1829.90 | 1833.67 | 1834.16 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1844.30 | 1835.80 | 1835.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 1856.10 | 1843.83 | 1841.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1844.30 | 1850.43 | 1846.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 1844.30 | 1850.43 | 1846.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1844.30 | 1850.43 | 1846.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 1844.90 | 1850.43 | 1846.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1851.40 | 1850.62 | 1846.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 1857.00 | 1853.08 | 1848.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 1863.00 | 1867.65 | 1868.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1863.00 | 1867.65 | 1868.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 1855.30 | 1862.74 | 1864.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1862.90 | 1862.06 | 1864.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 1862.90 | 1862.06 | 1864.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1862.90 | 1862.06 | 1864.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 1856.90 | 1862.36 | 1863.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 1854.90 | 1860.55 | 1862.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 1853.90 | 1859.22 | 1861.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 1859.00 | 1844.49 | 1843.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 1859.00 | 1844.49 | 1843.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 1863.10 | 1851.44 | 1848.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 1850.80 | 1852.40 | 1849.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 10:15:00 | 1850.80 | 1852.40 | 1849.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1850.80 | 1852.40 | 1849.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 1848.00 | 1852.40 | 1849.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1860.50 | 1853.77 | 1850.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 1865.50 | 1855.23 | 1851.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 1862.70 | 1869.90 | 1865.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 1864.00 | 1865.87 | 1864.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1862.00 | 1864.70 | 1863.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1863.00 | 1864.36 | 1863.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 1877.90 | 1864.30 | 1863.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 1839.90 | 1864.79 | 1869.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1835.00 | 1831.61 | 1840.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:45:00 | 1836.70 | 1831.61 | 1840.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1846.00 | 1834.48 | 1840.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1846.00 | 1834.48 | 1840.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1843.20 | 1836.23 | 1840.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 1850.00 | 1836.23 | 1840.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1839.50 | 1836.88 | 1840.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1836.70 | 1838.69 | 1841.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 1850.00 | 1842.85 | 1842.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1850.00 | 1842.85 | 1842.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1860.50 | 1847.41 | 1845.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 1852.40 | 1852.61 | 1849.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 1852.40 | 1852.61 | 1849.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1852.40 | 1852.61 | 1849.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:45:00 | 1850.80 | 1852.61 | 1849.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1854.70 | 1853.86 | 1850.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 1852.10 | 1853.86 | 1850.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1849.50 | 1852.98 | 1850.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 1849.50 | 1852.98 | 1850.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1847.70 | 1851.93 | 1850.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1847.70 | 1851.93 | 1850.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1845.10 | 1849.29 | 1849.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1839.00 | 1846.93 | 1848.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1844.10 | 1843.75 | 1846.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1844.10 | 1843.75 | 1846.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1844.10 | 1843.75 | 1846.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:45:00 | 1840.60 | 1844.24 | 1845.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 1841.10 | 1843.22 | 1844.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 1840.40 | 1839.53 | 1842.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:00:00 | 1840.00 | 1838.68 | 1841.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1852.20 | 1841.30 | 1841.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 1850.00 | 1841.30 | 1841.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1858.00 | 1844.64 | 1843.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1858.00 | 1844.64 | 1843.01 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1839.10 | 1844.36 | 1844.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 1828.00 | 1838.52 | 1841.50 | Break + close below crossover candle low |

### Cycle 93 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 1916.20 | 1845.47 | 1841.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 1930.00 | 1862.38 | 1849.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 11:15:00 | 1864.90 | 1865.67 | 1854.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 1864.90 | 1865.67 | 1854.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1874.90 | 1867.70 | 1859.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 1880.10 | 1871.96 | 1865.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1856.10 | 1865.22 | 1866.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1856.10 | 1865.22 | 1866.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 1850.00 | 1862.17 | 1864.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 1854.60 | 1852.12 | 1855.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1854.60 | 1852.12 | 1855.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1854.60 | 1852.12 | 1855.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 1854.60 | 1852.12 | 1855.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1853.90 | 1851.48 | 1854.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1853.90 | 1851.48 | 1854.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1855.00 | 1852.18 | 1854.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1845.90 | 1852.18 | 1854.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1839.00 | 1849.54 | 1852.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 1837.00 | 1847.62 | 1851.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 1838.00 | 1844.67 | 1849.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:45:00 | 1838.30 | 1843.64 | 1848.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:00:00 | 1837.20 | 1842.35 | 1847.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1791.60 | 1789.70 | 1798.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1786.10 | 1789.10 | 1796.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1786.90 | 1781.94 | 1785.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 1784.80 | 1783.01 | 1785.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 1786.50 | 1783.01 | 1785.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1785.40 | 1783.49 | 1785.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 1783.80 | 1782.79 | 1785.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 10:15:00 | 1783.40 | 1779.84 | 1782.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 1787.40 | 1769.39 | 1765.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1761.90 | 1769.05 | 1766.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1761.90 | 1769.05 | 1766.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1761.90 | 1769.05 | 1766.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1760.70 | 1769.05 | 1766.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1763.10 | 1767.86 | 1765.93 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 1757.10 | 1764.20 | 1764.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 1755.00 | 1762.36 | 1763.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 1736.40 | 1728.76 | 1732.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 1736.40 | 1728.76 | 1732.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 1736.40 | 1728.76 | 1732.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 1734.40 | 1728.76 | 1732.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1744.40 | 1731.89 | 1734.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 1744.40 | 1731.89 | 1734.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1730.00 | 1731.44 | 1733.28 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1738.10 | 1733.76 | 1733.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 1742.00 | 1735.86 | 1734.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 1738.20 | 1738.34 | 1736.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1748.90 | 1738.34 | 1736.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1742.20 | 1739.91 | 1737.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:30:00 | 1741.90 | 1739.91 | 1737.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1750.00 | 1759.96 | 1753.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 1750.00 | 1759.96 | 1753.87 | SL hit (close<ema400) qty=1.00 sl=1753.87 alert=retest1 |

### Cycle 98 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 1742.00 | 1753.31 | 1753.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1734.50 | 1747.32 | 1750.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 1704.40 | 1703.10 | 1715.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 1704.40 | 1703.10 | 1715.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1706.70 | 1705.47 | 1713.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1702.30 | 1705.30 | 1712.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 1719.00 | 1708.04 | 1713.33 | SL hit (close>static) qty=1.00 sl=1717.00 alert=retest2 |

### Cycle 99 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1725.60 | 1714.94 | 1714.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 1728.60 | 1717.67 | 1715.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 10:15:00 | 1737.30 | 1738.12 | 1730.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:30:00 | 1739.00 | 1738.12 | 1730.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1730.00 | 1736.07 | 1732.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1736.90 | 1736.07 | 1732.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1738.30 | 1736.51 | 1732.89 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 1720.90 | 1730.59 | 1730.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1706.80 | 1725.83 | 1728.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1706.80 | 1704.52 | 1714.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 1706.80 | 1704.52 | 1714.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 1714.20 | 1706.16 | 1713.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 1714.20 | 1706.16 | 1713.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1731.00 | 1711.13 | 1715.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1727.10 | 1715.00 | 1716.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1729.00 | 1719.40 | 1718.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 1733.60 | 1722.24 | 1719.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1719.60 | 1724.83 | 1722.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1719.60 | 1724.83 | 1722.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1719.60 | 1724.83 | 1722.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:15:00 | 1716.00 | 1724.83 | 1722.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1719.00 | 1723.66 | 1721.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:45:00 | 1716.90 | 1723.66 | 1721.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1698.20 | 1716.79 | 1718.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1684.90 | 1710.41 | 1715.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1698.00 | 1696.94 | 1707.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 1686.00 | 1687.38 | 1695.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1686.00 | 1687.38 | 1695.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 14:15:00 | 1680.00 | 1689.66 | 1694.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 15:00:00 | 1684.50 | 1688.63 | 1693.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:00:00 | 1682.00 | 1680.29 | 1686.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:45:00 | 1679.70 | 1680.11 | 1685.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1635.70 | 1668.09 | 1676.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:00:00 | 1617.90 | 1643.94 | 1658.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:15:00 | 1600.27 | 1630.80 | 1648.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:30:00 | 1616.00 | 1630.80 | 1648.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 1596.00 | 1626.96 | 1645.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 1597.90 | 1626.96 | 1645.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 1595.71 | 1626.96 | 1645.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 1630.80 | 1626.29 | 1641.72 | SL hit (close>ema200) qty=0.50 sl=1626.29 alert=retest2 |

### Cycle 103 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1681.10 | 1650.27 | 1648.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 1687.00 | 1671.64 | 1662.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1677.00 | 1681.35 | 1671.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1677.00 | 1681.35 | 1671.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1670.80 | 1678.08 | 1671.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 1670.80 | 1678.08 | 1671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 1673.10 | 1677.09 | 1671.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 1670.00 | 1677.09 | 1671.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 1671.60 | 1675.99 | 1671.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 1671.60 | 1675.99 | 1671.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1677.50 | 1676.29 | 1672.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 1676.00 | 1676.29 | 1672.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 1674.90 | 1676.01 | 1672.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 1668.50 | 1676.01 | 1672.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1656.10 | 1672.03 | 1670.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1656.10 | 1672.03 | 1670.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 1660.40 | 1669.70 | 1670.03 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1676.00 | 1668.66 | 1668.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1690.10 | 1672.95 | 1670.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1694.10 | 1698.07 | 1689.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 1694.10 | 1698.07 | 1689.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1688.50 | 1695.08 | 1690.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1688.60 | 1695.08 | 1690.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1690.70 | 1694.20 | 1690.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:15:00 | 1687.60 | 1694.20 | 1690.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1688.00 | 1692.96 | 1690.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 1688.00 | 1692.96 | 1690.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1690.00 | 1692.37 | 1690.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 1695.00 | 1692.37 | 1690.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1681.30 | 1691.01 | 1690.63 | SL hit (close<static) qty=1.00 sl=1684.90 alert=retest2 |

### Cycle 106 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 1679.80 | 1688.77 | 1689.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 1676.00 | 1684.65 | 1687.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 1644.20 | 1638.51 | 1648.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 1644.20 | 1638.51 | 1648.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1638.50 | 1638.51 | 1647.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1640.00 | 1638.51 | 1647.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1644.40 | 1639.11 | 1646.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1633.00 | 1637.23 | 1643.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 1634.10 | 1637.74 | 1640.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 1624.60 | 1621.70 | 1621.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1624.60 | 1621.70 | 1621.32 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 1617.30 | 1621.29 | 1621.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 1615.60 | 1620.15 | 1620.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1522.50 | 1519.08 | 1536.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 1526.30 | 1519.08 | 1536.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1531.00 | 1522.41 | 1534.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 1531.00 | 1522.41 | 1534.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1477.70 | 1472.74 | 1480.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 1474.30 | 1473.47 | 1479.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:30:00 | 1474.90 | 1473.98 | 1479.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 13:00:00 | 1474.30 | 1474.04 | 1479.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1400.58 | 1419.90 | 1440.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1401.15 | 1419.90 | 1440.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1400.58 | 1419.90 | 1440.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1378.60 | 1376.98 | 1390.92 | SL hit (close>ema200) qty=0.50 sl=1376.98 alert=retest2 |

### Cycle 109 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 1404.60 | 1394.39 | 1394.08 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1373.20 | 1393.41 | 1394.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 1366.60 | 1388.05 | 1391.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1370.80 | 1367.32 | 1377.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:00:00 | 1370.80 | 1367.32 | 1377.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1369.80 | 1368.12 | 1376.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1345.00 | 1375.21 | 1377.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1379.70 | 1345.08 | 1348.65 | SL hit (close>static) qty=1.00 sl=1376.40 alert=retest2 |

### Cycle 111 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1377.50 | 1351.57 | 1351.28 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1317.60 | 1348.54 | 1352.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1315.90 | 1332.31 | 1342.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1310.50 | 1284.62 | 1305.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1310.50 | 1284.62 | 1305.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1310.50 | 1284.62 | 1305.05 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 1329.00 | 1314.78 | 1313.58 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1304.10 | 1311.50 | 1312.22 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1319.80 | 1313.71 | 1313.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 1328.00 | 1316.56 | 1314.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1339.50 | 1343.06 | 1332.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 1339.50 | 1343.06 | 1332.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1337.50 | 1341.43 | 1333.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 1335.00 | 1341.43 | 1333.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1389.60 | 1411.58 | 1401.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:30:00 | 1410.40 | 1406.73 | 1402.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1427.10 | 1406.73 | 1402.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1426.60 | 1433.93 | 1434.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1426.60 | 1433.93 | 1434.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1410.80 | 1424.91 | 1429.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1428.70 | 1415.06 | 1420.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1428.70 | 1415.06 | 1420.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1428.70 | 1415.06 | 1420.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1432.80 | 1415.06 | 1420.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1433.00 | 1418.65 | 1421.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1433.00 | 1418.65 | 1421.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1454.90 | 1427.35 | 1425.03 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1410.20 | 1434.70 | 1435.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 1409.00 | 1429.56 | 1433.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 1425.60 | 1421.79 | 1426.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 10:15:00 | 1425.60 | 1421.79 | 1426.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1425.60 | 1421.79 | 1426.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 1425.60 | 1421.79 | 1426.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1425.00 | 1422.43 | 1426.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 1432.90 | 1422.43 | 1426.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 1411.00 | 1420.14 | 1424.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 1406.40 | 1420.14 | 1424.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:15:00 | 1396.50 | 1418.31 | 1423.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1422.40 | 1408.96 | 1407.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1422.40 | 1408.96 | 1407.37 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 1400.10 | 1411.33 | 1411.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 1395.00 | 1408.07 | 1410.17 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 12:15:00 | 2537.00 | 2024-05-28 12:15:00 | 2573.45 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2024-05-22 13:15:00 | 2537.30 | 2024-05-28 12:15:00 | 2573.45 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2024-05-23 11:00:00 | 2544.95 | 2024-05-28 12:15:00 | 2573.45 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-06-19 11:15:00 | 2622.75 | 2024-06-19 13:15:00 | 2629.35 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2024-06-21 13:00:00 | 2603.35 | 2024-06-26 10:15:00 | 2633.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-06-21 14:15:00 | 2600.50 | 2024-06-26 10:15:00 | 2633.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-06-21 15:00:00 | 2587.85 | 2024-06-26 10:15:00 | 2633.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-07-01 09:15:00 | 2633.00 | 2024-07-05 09:15:00 | 2721.40 | STOP_HIT | 1.00 | 3.36% |
| SELL | retest2 | 2024-07-10 09:15:00 | 2673.00 | 2024-07-12 10:15:00 | 2677.35 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-07-11 10:15:00 | 2672.00 | 2024-07-12 10:15:00 | 2677.35 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-07-11 11:15:00 | 2675.15 | 2024-07-12 10:15:00 | 2677.35 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-07-11 12:00:00 | 2677.00 | 2024-07-12 10:15:00 | 2677.35 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-07-16 13:15:00 | 2713.25 | 2024-07-18 10:15:00 | 2679.05 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-07-16 14:15:00 | 2710.40 | 2024-07-18 10:15:00 | 2679.05 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-23 12:15:00 | 2602.40 | 2024-07-29 09:15:00 | 2640.75 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-07-23 13:30:00 | 2634.75 | 2024-07-29 09:15:00 | 2640.75 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-07-24 10:15:00 | 2638.60 | 2024-07-29 09:15:00 | 2640.75 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-07-24 11:15:00 | 2635.05 | 2024-07-29 09:15:00 | 2640.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-08-08 09:15:00 | 2377.60 | 2024-08-16 15:15:00 | 2339.80 | STOP_HIT | 1.00 | 1.59% |
| BUY | retest2 | 2024-08-21 09:15:00 | 2339.40 | 2024-08-21 12:15:00 | 2322.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-09-06 11:30:00 | 2422.20 | 2024-09-18 12:15:00 | 2471.85 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2024-09-09 09:45:00 | 2421.70 | 2024-09-18 12:15:00 | 2471.85 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2024-09-09 10:30:00 | 2419.00 | 2024-09-18 12:15:00 | 2471.85 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2024-10-01 13:00:00 | 2513.10 | 2024-10-03 11:15:00 | 2471.30 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-10-01 14:45:00 | 2510.20 | 2024-10-03 11:15:00 | 2471.30 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-10-03 10:45:00 | 2512.85 | 2024-10-03 11:15:00 | 2471.30 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-10-09 11:45:00 | 2363.50 | 2024-10-18 09:15:00 | 2245.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 13:00:00 | 2363.15 | 2024-10-18 09:15:00 | 2244.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 11:45:00 | 2363.50 | 2024-10-18 13:15:00 | 2281.95 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2024-10-09 13:00:00 | 2363.15 | 2024-10-18 13:15:00 | 2281.95 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2024-10-24 11:30:00 | 2266.70 | 2024-10-28 12:15:00 | 2293.05 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-10-24 15:00:00 | 2269.00 | 2024-10-28 12:15:00 | 2293.05 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-10-28 11:30:00 | 2260.60 | 2024-10-28 12:15:00 | 2293.05 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-11-21 09:15:00 | 1967.15 | 2024-11-21 09:15:00 | 1868.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-21 09:15:00 | 1967.15 | 2024-11-22 09:15:00 | 2068.10 | STOP_HIT | 0.50 | -5.13% |
| BUY | retest2 | 2024-12-05 11:30:00 | 2264.00 | 2024-12-10 11:15:00 | 2249.45 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-12-05 13:00:00 | 2264.80 | 2024-12-10 11:15:00 | 2249.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-12-05 14:15:00 | 2268.20 | 2024-12-10 11:15:00 | 2249.45 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-12-05 15:00:00 | 2268.80 | 2024-12-10 11:15:00 | 2249.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-12 13:15:00 | 2241.45 | 2024-12-16 14:15:00 | 2246.55 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-12-16 11:00:00 | 2241.20 | 2024-12-16 14:15:00 | 2246.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-12-16 13:15:00 | 2241.70 | 2024-12-16 14:15:00 | 2246.55 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-12-27 11:15:00 | 2085.45 | 2025-01-02 13:15:00 | 2078.25 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2024-12-30 11:30:00 | 2083.85 | 2025-01-02 13:15:00 | 2078.25 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-12-30 12:15:00 | 2083.00 | 2025-01-02 13:15:00 | 2078.25 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-01-08 15:15:00 | 2005.60 | 2025-01-13 09:15:00 | 1905.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 15:15:00 | 2005.60 | 2025-01-14 09:15:00 | 1907.20 | STOP_HIT | 0.50 | 4.91% |
| BUY | retest2 | 2025-01-17 14:15:00 | 2005.15 | 2025-01-22 12:15:00 | 1981.60 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-01-20 10:45:00 | 2003.35 | 2025-01-22 12:15:00 | 1981.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-01-20 12:30:00 | 2003.05 | 2025-01-22 12:15:00 | 1981.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-01-22 09:45:00 | 2008.30 | 2025-01-22 12:15:00 | 1981.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-01-27 11:15:00 | 2045.10 | 2025-01-27 14:15:00 | 1999.75 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-01-28 14:45:00 | 2005.70 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-01-29 10:30:00 | 2009.10 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-01-29 11:00:00 | 2006.35 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-01-29 11:45:00 | 2004.80 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-01-30 10:45:00 | 1966.25 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-01-30 13:15:00 | 1966.40 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-02-10 10:45:00 | 1972.85 | 2025-02-12 09:15:00 | 1874.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:45:00 | 1972.85 | 2025-02-12 11:15:00 | 1930.90 | STOP_HIT | 0.50 | 2.13% |
| SELL | retest2 | 2025-03-03 10:30:00 | 1783.00 | 2025-03-04 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-03-21 09:15:00 | 1899.80 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-03-21 10:30:00 | 1900.50 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-03-21 11:15:00 | 1901.30 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-03-21 12:00:00 | 1901.75 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-03-26 10:45:00 | 1954.00 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-03-27 14:15:00 | 1952.60 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-03-28 11:30:00 | 1950.20 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-04-01 09:15:00 | 1956.20 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-04-02 12:00:00 | 1964.65 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-04-04 15:00:00 | 1967.90 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1876.70 | 2025-05-09 09:15:00 | 1782.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1876.70 | 2025-05-09 15:15:00 | 1810.10 | STOP_HIT | 0.50 | 3.55% |
| BUY | retest2 | 2025-05-21 13:00:00 | 1941.70 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-05-21 14:15:00 | 1942.00 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-21 15:00:00 | 1942.20 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-05-22 09:45:00 | 1941.90 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-22 14:00:00 | 1937.50 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-05-22 14:30:00 | 1939.50 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-05-27 10:30:00 | 1937.00 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-05-27 11:15:00 | 1937.70 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-06-03 12:45:00 | 1872.10 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-03 13:30:00 | 1875.00 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-03 14:15:00 | 1875.00 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-04 09:15:00 | 1872.30 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-17 15:15:00 | 1857.00 | 2025-06-24 10:15:00 | 1858.80 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-06-18 11:15:00 | 1852.80 | 2025-06-24 10:15:00 | 1858.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-08 12:30:00 | 1965.50 | 2025-07-14 10:15:00 | 1971.40 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-07-15 12:30:00 | 1983.90 | 2025-07-16 13:15:00 | 1993.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-16 12:15:00 | 1985.00 | 2025-07-16 13:15:00 | 1993.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-21 10:30:00 | 1966.30 | 2025-07-25 09:15:00 | 1871.59 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1969.00 | 2025-07-25 10:15:00 | 1867.98 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1970.10 | 2025-07-25 10:15:00 | 1870.55 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-07-21 10:30:00 | 1966.30 | 2025-07-31 12:15:00 | 1824.90 | STOP_HIT | 0.50 | 7.19% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1969.00 | 2025-07-31 12:15:00 | 1824.90 | STOP_HIT | 0.50 | 7.32% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1970.10 | 2025-07-31 12:15:00 | 1824.90 | STOP_HIT | 0.50 | 7.37% |
| SELL | retest2 | 2025-08-13 11:45:00 | 1783.10 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-08-13 12:15:00 | 1785.00 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-08-13 13:15:00 | 1785.70 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-08-13 15:15:00 | 1785.20 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-08-14 11:15:00 | 1785.90 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-08-14 12:45:00 | 1785.80 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-08-14 13:45:00 | 1784.60 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-08-14 14:15:00 | 1785.10 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1825.20 | 2025-09-05 12:15:00 | 1827.50 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-09-03 09:45:00 | 1832.70 | 2025-09-05 12:15:00 | 1827.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1852.00 | 2025-09-11 12:15:00 | 1840.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-09-23 09:15:00 | 1891.20 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-09-23 10:00:00 | 1893.60 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-23 10:30:00 | 1890.80 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-23 13:15:00 | 1887.10 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-08 14:30:00 | 1857.00 | 2025-10-14 11:15:00 | 1863.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-10-16 13:15:00 | 1856.90 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-10-17 10:15:00 | 1854.90 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-10-17 11:00:00 | 1853.90 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-28 15:15:00 | 1865.50 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-30 12:15:00 | 1862.70 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-30 15:00:00 | 1864.00 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1862.00 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-31 12:15:00 | 1877.90 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1836.70 | 2025-11-10 12:15:00 | 1850.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-18 09:45:00 | 1840.60 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-18 11:30:00 | 1841.10 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-19 09:30:00 | 1840.40 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-11-19 13:00:00 | 1840.00 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-11-26 14:45:00 | 1880.10 | 2025-11-28 11:15:00 | 1856.10 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-03 10:30:00 | 1837.00 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.34% |
| SELL | retest2 | 2025-12-03 12:45:00 | 1838.00 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.39% |
| SELL | retest2 | 2025-12-03 13:45:00 | 1838.30 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.41% |
| SELL | retest2 | 2025-12-03 15:00:00 | 1837.20 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.35% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1786.10 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-12-12 10:15:00 | 1786.90 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-12-12 10:45:00 | 1784.80 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2025-12-12 11:15:00 | 1786.50 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-12-12 12:30:00 | 1783.80 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-12-15 10:15:00 | 1783.40 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest1 | 2026-01-02 09:15:00 | 1748.90 | 2026-01-06 10:15:00 | 1750.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-01-07 14:45:00 | 1758.00 | 2026-01-08 09:15:00 | 1742.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-07 15:15:00 | 1758.70 | 2026-01-08 09:15:00 | 1742.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1702.30 | 2026-01-13 11:15:00 | 1719.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-28 14:15:00 | 1680.00 | 2026-02-02 10:15:00 | 1600.27 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-28 15:00:00 | 1684.50 | 2026-02-02 11:15:00 | 1596.00 | PARTIAL | 0.50 | 5.25% |
| SELL | retest2 | 2026-01-29 14:00:00 | 1682.00 | 2026-02-02 11:15:00 | 1597.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1679.70 | 2026-02-02 11:15:00 | 1595.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 14:15:00 | 1680.00 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2026-01-28 15:00:00 | 1684.50 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-01-29 14:00:00 | 1682.00 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1679.70 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2026-02-01 15:00:00 | 1617.90 | 2026-02-03 11:15:00 | 1681.10 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2026-02-02 10:30:00 | 1616.00 | 2026-02-03 11:15:00 | 1681.10 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2026-02-02 14:30:00 | 1624.30 | 2026-02-03 11:15:00 | 1681.10 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2026-02-11 13:15:00 | 1695.00 | 2026-02-12 10:15:00 | 1681.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1633.00 | 2026-02-25 11:15:00 | 1624.60 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2026-02-19 11:45:00 | 1634.10 | 2026-02-25 11:15:00 | 1624.60 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1474.30 | 2026-03-13 12:15:00 | 1400.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:30:00 | 1474.90 | 2026-03-13 12:15:00 | 1401.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 1474.30 | 2026-03-13 12:15:00 | 1400.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1474.30 | 2026-03-17 13:15:00 | 1378.60 | STOP_HIT | 0.50 | 6.49% |
| SELL | retest2 | 2026-03-11 11:30:00 | 1474.90 | 2026-03-17 13:15:00 | 1378.60 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2026-03-11 13:00:00 | 1474.30 | 2026-03-17 13:15:00 | 1378.60 | STOP_HIT | 0.50 | 6.49% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1345.00 | 2026-03-25 09:15:00 | 1379.70 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-04-13 14:30:00 | 1410.40 | 2026-04-23 11:15:00 | 1426.60 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1427.10 | 2026-04-23 11:15:00 | 1426.60 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2026-05-04 13:15:00 | 1406.40 | 2026-05-07 09:15:00 | 1422.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-05-04 14:15:00 | 1396.50 | 2026-05-07 09:15:00 | 1422.40 | STOP_HIT | 1.00 | -1.85% |
