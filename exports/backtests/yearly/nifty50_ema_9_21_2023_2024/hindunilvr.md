# HINDUNILVR (HINDUNILVR)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 2286.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 221 |
| ALERT1 | 144 |
| ALERT2 | 142 |
| ALERT2_SKIP | 67 |
| ALERT3 | 397 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 237 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 237 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 248 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 95 / 153
- **Target hits / Stop hits / Partials:** 5 / 236 / 7
- **Avg / median % per leg:** 0.20% / -0.44%
- **Sum % (uncompounded):** 49.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 121 | 46 | 38.0% | 5 | 115 | 1 | 0.30% | 36.9% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 4 | 1 | 1.71% | 8.5% |
| BUY @ 3rd Alert (retest2) | 116 | 43 | 37.1% | 5 | 111 | 0 | 0.24% | 28.4% |
| SELL (all) | 127 | 49 | 38.6% | 0 | 121 | 6 | 0.10% | 12.8% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.21% | 1.2% |
| SELL @ 3rd Alert (retest2) | 126 | 48 | 38.1% | 0 | 120 | 6 | 0.09% | 11.6% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 5 | 1 | 1.62% | 9.7% |
| retest2 (combined) | 242 | 91 | 37.6% | 5 | 231 | 6 | 0.17% | 40.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 13:15:00 | 2596.30 | 2603.84 | 2604.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 14:15:00 | 2584.01 | 2599.88 | 2602.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 14:15:00 | 2596.35 | 2583.01 | 2590.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 14:15:00 | 2596.35 | 2583.01 | 2590.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 2596.35 | 2583.01 | 2590.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 2596.35 | 2583.01 | 2590.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 2597.88 | 2585.98 | 2590.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:30:00 | 2604.12 | 2588.40 | 2591.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 2600.14 | 2590.75 | 2592.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 11:00:00 | 2600.14 | 2590.75 | 2592.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 12:15:00 | 2596.74 | 2592.53 | 2592.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 13:00:00 | 2596.74 | 2592.53 | 2592.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 13:15:00 | 2595.22 | 2593.07 | 2593.16 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 14:15:00 | 2595.66 | 2593.59 | 2593.39 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 13:15:00 | 2586.12 | 2592.90 | 2593.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 09:15:00 | 2581.64 | 2589.22 | 2591.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 10:15:00 | 2591.14 | 2589.60 | 2591.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 10:15:00 | 2591.14 | 2589.60 | 2591.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 2591.14 | 2589.60 | 2591.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 11:00:00 | 2591.14 | 2589.60 | 2591.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 11:15:00 | 2580.07 | 2587.70 | 2590.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 12:15:00 | 2577.76 | 2587.70 | 2590.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 11:45:00 | 2578.20 | 2567.80 | 2571.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 12:15:00 | 2577.22 | 2567.80 | 2571.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 13:15:00 | 2592.81 | 2575.09 | 2574.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 13:15:00 | 2592.81 | 2575.09 | 2574.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 14:15:00 | 2610.66 | 2582.20 | 2577.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 14:15:00 | 2606.43 | 2610.50 | 2597.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 15:00:00 | 2606.43 | 2610.50 | 2597.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 2598.96 | 2610.62 | 2605.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 10:00:00 | 2598.96 | 2610.62 | 2605.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 10:15:00 | 2604.81 | 2609.46 | 2605.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 14:15:00 | 2622.96 | 2608.91 | 2606.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-06 11:15:00 | 2631.81 | 2648.54 | 2650.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 11:15:00 | 2631.81 | 2648.54 | 2650.67 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 10:15:00 | 2680.95 | 2655.71 | 2652.62 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 2635.25 | 2653.56 | 2655.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 14:15:00 | 2633.14 | 2646.68 | 2651.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 10:15:00 | 2602.55 | 2601.64 | 2618.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-12 11:00:00 | 2602.55 | 2601.64 | 2618.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 2647.01 | 2612.07 | 2615.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:00:00 | 2647.01 | 2612.07 | 2615.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 2646.08 | 2618.87 | 2618.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 14:15:00 | 2653.45 | 2638.58 | 2631.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 11:15:00 | 2642.58 | 2647.24 | 2638.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 12:00:00 | 2642.58 | 2647.24 | 2638.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 2639.63 | 2645.72 | 2638.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 13:00:00 | 2639.63 | 2645.72 | 2638.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 2640.03 | 2644.58 | 2638.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:15:00 | 2644.11 | 2644.58 | 2638.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 2649.27 | 2645.52 | 2639.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:15:00 | 2656.01 | 2645.30 | 2640.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 11:00:00 | 2663.29 | 2652.11 | 2644.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 13:15:00 | 2653.94 | 2649.71 | 2644.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 14:15:00 | 2641.16 | 2645.93 | 2646.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 14:15:00 | 2641.16 | 2645.93 | 2646.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 15:15:00 | 2633.29 | 2643.40 | 2645.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 14:15:00 | 2630.63 | 2629.51 | 2635.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-20 15:00:00 | 2630.63 | 2629.51 | 2635.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 2633.63 | 2630.33 | 2635.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 2650.99 | 2630.33 | 2635.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 2658.76 | 2636.02 | 2637.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:45:00 | 2657.78 | 2636.02 | 2637.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 2637.22 | 2636.26 | 2637.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 12:45:00 | 2635.89 | 2636.77 | 2637.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 14:15:00 | 2634.27 | 2637.74 | 2638.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 11:15:00 | 2629.30 | 2612.71 | 2611.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 11:15:00 | 2629.30 | 2612.71 | 2611.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 12:15:00 | 2632.40 | 2616.65 | 2613.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 2620.16 | 2620.28 | 2615.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 2620.16 | 2620.28 | 2615.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 2620.16 | 2620.28 | 2615.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 2620.16 | 2620.28 | 2615.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 2625.86 | 2621.40 | 2616.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 2626.84 | 2621.40 | 2616.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-30 10:15:00 | 2614.60 | 2619.58 | 2616.59 | SL hit (close<static) qty=1.00 sl=2615.19 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 15:15:00 | 2654.93 | 2680.15 | 2683.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 2622.91 | 2668.70 | 2677.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 2639.53 | 2635.34 | 2652.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-11 09:45:00 | 2635.06 | 2635.34 | 2652.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 2660.09 | 2640.29 | 2653.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:00:00 | 2660.09 | 2640.29 | 2653.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 2669.49 | 2646.13 | 2654.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 12:00:00 | 2669.49 | 2646.13 | 2654.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 2638.45 | 2644.59 | 2653.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 09:15:00 | 2627.53 | 2643.74 | 2650.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 10:45:00 | 2631.32 | 2630.04 | 2636.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 11:15:00 | 2626.45 | 2630.04 | 2636.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 12:00:00 | 2626.70 | 2629.38 | 2635.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 2631.81 | 2622.88 | 2628.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:00:00 | 2631.81 | 2622.88 | 2628.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 2630.58 | 2624.42 | 2629.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:15:00 | 2633.93 | 2624.42 | 2629.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 2640.17 | 2627.57 | 2630.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 13:00:00 | 2640.17 | 2627.57 | 2630.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 2638.50 | 2629.76 | 2630.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-14 14:15:00 | 2634.03 | 2629.76 | 2630.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 09:15:00 | 2652.96 | 2635.35 | 2633.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 2652.96 | 2635.35 | 2633.17 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 10:15:00 | 2620.50 | 2635.38 | 2636.25 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 12:15:00 | 2658.27 | 2637.37 | 2635.16 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 10:15:00 | 2591.97 | 2629.57 | 2633.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 11:15:00 | 2577.81 | 2619.22 | 2628.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 2542.69 | 2534.46 | 2552.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-26 09:30:00 | 2541.71 | 2534.46 | 2552.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 2553.46 | 2538.26 | 2552.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:00:00 | 2553.46 | 2538.26 | 2552.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 2548.10 | 2540.23 | 2551.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 15:00:00 | 2540.63 | 2543.79 | 2550.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 10:30:00 | 2543.77 | 2534.76 | 2539.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 15:15:00 | 2537.87 | 2536.50 | 2538.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 12:15:00 | 2522.13 | 2512.71 | 2511.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 12:15:00 | 2522.13 | 2512.71 | 2511.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 13:15:00 | 2524.35 | 2515.04 | 2512.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 2519.03 | 2520.29 | 2516.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 11:00:00 | 2519.03 | 2520.29 | 2516.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 2518.20 | 2519.87 | 2516.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 12:45:00 | 2521.20 | 2520.51 | 2517.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 09:15:00 | 2506.39 | 2518.38 | 2517.46 | SL hit (close<static) qty=1.00 sl=2513.82 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 2500.74 | 2514.85 | 2515.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 10:15:00 | 2489.03 | 2504.50 | 2509.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 09:15:00 | 2480.03 | 2473.35 | 2484.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 09:15:00 | 2480.03 | 2473.35 | 2484.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 2480.03 | 2473.35 | 2484.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 10:00:00 | 2480.03 | 2473.35 | 2484.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 2474.92 | 2473.66 | 2483.86 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 09:15:00 | 2504.03 | 2487.60 | 2487.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 12:15:00 | 2507.28 | 2494.36 | 2490.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 09:15:00 | 2500.00 | 2500.80 | 2495.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 2500.00 | 2500.80 | 2495.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 2500.00 | 2500.80 | 2495.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:30:00 | 2496.07 | 2500.80 | 2495.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 2508.46 | 2502.89 | 2497.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 12:15:00 | 2513.33 | 2502.89 | 2497.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 13:30:00 | 2509.79 | 2505.48 | 2499.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-18 11:15:00 | 2511.66 | 2503.22 | 2500.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-18 12:00:00 | 2513.53 | 2505.28 | 2501.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 2504.57 | 2510.25 | 2505.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:45:00 | 2501.67 | 2510.25 | 2505.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 2529.02 | 2514.00 | 2507.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 11:15:00 | 2531.63 | 2514.00 | 2507.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 11:45:00 | 2531.58 | 2521.48 | 2515.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 10:30:00 | 2531.58 | 2524.69 | 2519.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 12:45:00 | 2531.13 | 2527.50 | 2522.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 2541.17 | 2536.41 | 2530.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-25 14:15:00 | 2520.76 | 2528.25 | 2528.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 14:15:00 | 2520.76 | 2528.25 | 2528.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 2512.94 | 2525.19 | 2527.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 14:15:00 | 2514.41 | 2514.32 | 2519.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 15:00:00 | 2514.41 | 2514.32 | 2519.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 2482.98 | 2489.37 | 2496.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 10:15:00 | 2479.83 | 2489.37 | 2496.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 10:15:00 | 2475.90 | 2470.36 | 2469.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 10:15:00 | 2475.90 | 2470.36 | 2469.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 14:15:00 | 2482.15 | 2472.89 | 2471.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 09:15:00 | 2466.56 | 2473.05 | 2471.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 09:15:00 | 2466.56 | 2473.05 | 2471.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 2466.56 | 2473.05 | 2471.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:00:00 | 2466.56 | 2473.05 | 2471.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 2464.88 | 2471.41 | 2471.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:30:00 | 2463.36 | 2471.41 | 2471.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 11:15:00 | 2463.85 | 2469.90 | 2470.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 12:15:00 | 2459.67 | 2467.86 | 2469.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 15:15:00 | 2468.82 | 2467.91 | 2469.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 15:15:00 | 2468.82 | 2467.91 | 2469.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 2468.82 | 2467.91 | 2469.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:15:00 | 2472.70 | 2467.91 | 2469.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 2472.41 | 2468.81 | 2469.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:30:00 | 2473.59 | 2468.81 | 2469.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 2469.16 | 2468.88 | 2469.36 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 12:15:00 | 2477.28 | 2470.78 | 2470.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 09:15:00 | 2485.24 | 2474.84 | 2472.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 2464.59 | 2483.10 | 2479.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 2464.59 | 2483.10 | 2479.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 2464.59 | 2483.10 | 2479.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:00:00 | 2464.59 | 2483.10 | 2479.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 2459.18 | 2478.32 | 2477.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:45:00 | 2459.18 | 2478.32 | 2477.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 2457.21 | 2474.09 | 2475.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 09:15:00 | 2436.50 | 2456.14 | 2461.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-18 11:15:00 | 2439.50 | 2435.82 | 2444.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 11:15:00 | 2439.50 | 2435.82 | 2444.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 2439.50 | 2435.82 | 2444.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 12:00:00 | 2439.50 | 2435.82 | 2444.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 2446.69 | 2438.00 | 2444.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:00:00 | 2446.69 | 2438.00 | 2444.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 2449.78 | 2440.35 | 2445.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:30:00 | 2449.69 | 2440.35 | 2445.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 2431.63 | 2420.27 | 2427.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 15:00:00 | 2431.63 | 2420.27 | 2427.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 2440.19 | 2424.25 | 2428.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 09:15:00 | 2420.37 | 2424.25 | 2428.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 11:30:00 | 2430.21 | 2429.54 | 2429.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-22 12:15:00 | 2439.55 | 2431.54 | 2430.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 12:15:00 | 2439.55 | 2431.54 | 2430.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 13:15:00 | 2448.46 | 2434.93 | 2432.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 09:15:00 | 2426.86 | 2435.19 | 2433.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 2426.86 | 2435.19 | 2433.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 2426.86 | 2435.19 | 2433.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:00:00 | 2426.86 | 2435.19 | 2433.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 2425.29 | 2433.21 | 2432.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:30:00 | 2422.88 | 2433.21 | 2432.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 2434.49 | 2436.21 | 2434.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 15:00:00 | 2434.49 | 2436.21 | 2434.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 2438.72 | 2436.71 | 2434.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:15:00 | 2427.06 | 2436.71 | 2434.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 2424.70 | 2434.31 | 2433.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:45:00 | 2423.67 | 2434.31 | 2433.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 10:15:00 | 2424.40 | 2432.33 | 2433.03 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 15:15:00 | 2447.37 | 2435.64 | 2434.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 10:15:00 | 2455.24 | 2439.00 | 2435.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 09:15:00 | 2447.91 | 2451.46 | 2444.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 09:15:00 | 2447.91 | 2451.46 | 2444.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 2447.91 | 2451.46 | 2444.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:30:00 | 2442.65 | 2451.46 | 2444.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 2435.57 | 2448.28 | 2443.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:00:00 | 2435.57 | 2448.28 | 2443.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 2442.55 | 2447.13 | 2443.83 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 13:15:00 | 2423.86 | 2438.31 | 2440.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 14:15:00 | 2417.81 | 2434.21 | 2438.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 15:15:00 | 2427.75 | 2426.36 | 2430.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 15:15:00 | 2427.75 | 2426.36 | 2430.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 2427.75 | 2426.36 | 2430.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:15:00 | 2449.73 | 2426.36 | 2430.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 2450.32 | 2431.15 | 2432.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:00:00 | 2450.32 | 2431.15 | 2432.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 2455.24 | 2435.97 | 2434.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-04 14:15:00 | 2467.44 | 2449.56 | 2443.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 09:15:00 | 2459.67 | 2467.48 | 2458.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 2459.67 | 2467.48 | 2458.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 2459.67 | 2467.48 | 2458.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 10:00:00 | 2459.67 | 2467.48 | 2458.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 2465.08 | 2467.00 | 2459.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 12:15:00 | 2469.06 | 2466.61 | 2459.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 10:15:00 | 2468.97 | 2464.27 | 2461.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 11:15:00 | 2469.85 | 2464.95 | 2461.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 12:00:00 | 2468.97 | 2465.76 | 2462.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 12:15:00 | 2459.33 | 2464.47 | 2462.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 13:00:00 | 2459.33 | 2464.47 | 2462.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 13:15:00 | 2472.70 | 2466.12 | 2463.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 10:00:00 | 2475.90 | 2468.67 | 2465.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 11:15:00 | 2478.31 | 2469.63 | 2465.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 13:15:00 | 2476.93 | 2471.88 | 2467.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 09:15:00 | 2490.16 | 2473.69 | 2469.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 2506.64 | 2511.94 | 2504.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 11:00:00 | 2506.64 | 2511.94 | 2504.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 2517.85 | 2522.77 | 2517.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 15:00:00 | 2517.85 | 2522.77 | 2517.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 15:15:00 | 2513.28 | 2520.87 | 2517.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 09:15:00 | 2520.21 | 2520.87 | 2517.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 2520.21 | 2520.74 | 2517.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 10:30:00 | 2526.02 | 2521.02 | 2517.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 13:30:00 | 2525.48 | 2520.88 | 2518.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 10:15:00 | 2501.97 | 2514.32 | 2515.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 2501.97 | 2514.32 | 2515.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 2456.62 | 2497.46 | 2505.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-23 10:15:00 | 2466.41 | 2464.66 | 2479.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-23 11:00:00 | 2466.41 | 2464.66 | 2479.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 13:15:00 | 2438.91 | 2435.95 | 2445.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:15:00 | 2447.03 | 2435.95 | 2445.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 2439.75 | 2436.71 | 2444.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 15:15:00 | 2430.11 | 2436.71 | 2444.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 09:30:00 | 2429.77 | 2434.11 | 2442.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 13:15:00 | 2450.82 | 2439.49 | 2442.09 | SL hit (close>static) qty=1.00 sl=2449.34 alert=retest2 |

### Cycle 30 — BUY (started 2023-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 13:15:00 | 2444.57 | 2440.85 | 2440.74 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 10:15:00 | 2432.62 | 2439.33 | 2440.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 11:15:00 | 2431.63 | 2437.79 | 2439.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 12:15:00 | 2435.72 | 2432.99 | 2435.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 12:15:00 | 2435.72 | 2432.99 | 2435.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 2435.72 | 2432.99 | 2435.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:00:00 | 2435.72 | 2432.99 | 2435.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 2439.60 | 2434.31 | 2435.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:45:00 | 2439.85 | 2434.31 | 2435.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 14:15:00 | 2447.62 | 2436.97 | 2436.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 15:15:00 | 2452.29 | 2440.03 | 2438.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 12:15:00 | 2460.95 | 2462.77 | 2455.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-06 13:00:00 | 2460.95 | 2462.77 | 2455.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 2455.39 | 2461.30 | 2455.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:30:00 | 2452.00 | 2461.30 | 2455.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 2455.34 | 2460.11 | 2455.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 2460.36 | 2455.66 | 2454.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 10:00:00 | 2462.47 | 2457.02 | 2455.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 11:15:00 | 2459.23 | 2457.29 | 2455.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-09 10:15:00 | 2449.64 | 2463.91 | 2461.43 | SL hit (close<static) qty=1.00 sl=2452.34 alert=retest2 |

### Cycle 33 — SELL (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 12:15:00 | 2443.24 | 2457.60 | 2458.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 13:15:00 | 2436.45 | 2453.37 | 2456.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 14:15:00 | 2446.29 | 2442.86 | 2447.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 14:15:00 | 2446.29 | 2442.86 | 2447.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 2446.29 | 2442.86 | 2447.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 15:00:00 | 2446.29 | 2442.86 | 2447.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 2450.32 | 2444.76 | 2447.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 09:15:00 | 2435.57 | 2444.76 | 2447.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 14:00:00 | 2442.65 | 2441.67 | 2444.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 09:30:00 | 2441.03 | 2439.90 | 2443.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 14:15:00 | 2444.62 | 2441.89 | 2443.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 14:15:00 | 2444.77 | 2442.47 | 2443.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 15:15:00 | 2448.16 | 2442.47 | 2443.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 15:15:00 | 2448.16 | 2443.61 | 2443.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 09:15:00 | 2438.62 | 2443.61 | 2443.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-16 09:15:00 | 2454.36 | 2445.76 | 2444.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 09:15:00 | 2454.36 | 2445.76 | 2444.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 10:15:00 | 2462.96 | 2449.20 | 2446.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 12:15:00 | 2444.91 | 2449.43 | 2446.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 12:15:00 | 2444.91 | 2449.43 | 2446.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 12:15:00 | 2444.91 | 2449.43 | 2446.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 13:00:00 | 2444.91 | 2449.43 | 2446.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 13:15:00 | 2451.31 | 2449.80 | 2447.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 14:15:00 | 2452.93 | 2449.80 | 2447.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 15:15:00 | 2453.47 | 2448.79 | 2447.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 15:15:00 | 2461.69 | 2472.73 | 2473.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 15:15:00 | 2461.69 | 2472.73 | 2473.74 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 11:15:00 | 2477.62 | 2474.34 | 2474.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 13:15:00 | 2483.77 | 2476.80 | 2475.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 15:15:00 | 2517.21 | 2518.71 | 2505.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:15:00 | 2552.63 | 2518.71 | 2505.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 2535.85 | 2542.99 | 2532.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-05 13:15:00 | 2520.41 | 2538.48 | 2531.81 | SL hit (close<ema400) qty=1.00 sl=2531.81 alert=retest1 |

### Cycle 37 — SELL (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 09:15:00 | 2472.90 | 2519.58 | 2525.58 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 14:15:00 | 2475.36 | 2470.15 | 2469.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 15:15:00 | 2477.82 | 2471.68 | 2470.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 10:15:00 | 2470.00 | 2471.52 | 2470.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 10:15:00 | 2470.00 | 2471.52 | 2470.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 2470.00 | 2471.52 | 2470.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 10:45:00 | 2468.92 | 2471.52 | 2470.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 11:15:00 | 2470.54 | 2471.32 | 2470.74 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 13:15:00 | 2467.19 | 2470.18 | 2470.30 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 14:15:00 | 2482.79 | 2472.70 | 2471.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 15:15:00 | 2488.64 | 2475.89 | 2473.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 2514.02 | 2516.86 | 2507.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 14:00:00 | 2514.02 | 2516.86 | 2507.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 2516.13 | 2516.72 | 2508.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:30:00 | 2510.92 | 2516.72 | 2508.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 2512.30 | 2515.83 | 2508.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 09:15:00 | 2503.84 | 2515.83 | 2508.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 2515.49 | 2515.76 | 2509.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 10:15:00 | 2520.61 | 2515.76 | 2509.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:30:00 | 2518.15 | 2518.11 | 2512.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 09:15:00 | 2521.15 | 2517.58 | 2513.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 09:45:00 | 2525.03 | 2519.28 | 2514.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 2588.97 | 2602.19 | 2594.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:00:00 | 2588.97 | 2602.19 | 2594.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 2581.15 | 2597.98 | 2593.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:45:00 | 2579.63 | 2597.98 | 2593.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-02 12:15:00 | 2576.23 | 2590.00 | 2590.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 12:15:00 | 2576.23 | 2590.00 | 2590.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 13:15:00 | 2574.86 | 2586.97 | 2589.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 10:15:00 | 2575.15 | 2570.58 | 2576.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 10:15:00 | 2575.15 | 2570.58 | 2576.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 2575.15 | 2570.58 | 2576.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 10:45:00 | 2579.23 | 2570.58 | 2576.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 2568.37 | 2570.14 | 2575.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-04 14:00:00 | 2564.53 | 2568.58 | 2573.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 10:45:00 | 2566.40 | 2564.07 | 2569.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 11:15:00 | 2578.01 | 2566.86 | 2570.20 | SL hit (close>static) qty=1.00 sl=2576.14 alert=retest2 |

### Cycle 42 — BUY (started 2024-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 14:15:00 | 2577.91 | 2572.94 | 2572.46 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 2541.02 | 2567.24 | 2570.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 10:15:00 | 2533.30 | 2560.45 | 2566.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 10:15:00 | 2544.76 | 2541.84 | 2551.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-09 11:00:00 | 2544.76 | 2541.84 | 2551.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 2539.84 | 2540.17 | 2546.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 11:00:00 | 2529.46 | 2538.03 | 2544.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 13:30:00 | 2527.10 | 2533.33 | 2540.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 14:00:00 | 2526.46 | 2533.33 | 2540.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 10:00:00 | 2529.95 | 2532.64 | 2538.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 2510.33 | 2504.03 | 2511.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:45:00 | 2508.56 | 2504.03 | 2511.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 11:15:00 | 2510.23 | 2505.27 | 2511.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 12:00:00 | 2510.23 | 2505.27 | 2511.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 12:15:00 | 2500.25 | 2504.27 | 2510.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-15 15:15:00 | 2532.95 | 2516.45 | 2514.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 15:15:00 | 2532.95 | 2516.45 | 2514.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 10:15:00 | 2540.04 | 2523.96 | 2518.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 15:15:00 | 2527.20 | 2529.25 | 2523.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 15:15:00 | 2527.20 | 2529.25 | 2523.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 2527.20 | 2529.25 | 2523.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:15:00 | 2505.51 | 2529.25 | 2523.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 2505.07 | 2524.41 | 2522.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:00:00 | 2505.07 | 2524.41 | 2522.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 2508.36 | 2521.20 | 2520.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:45:00 | 2504.52 | 2521.20 | 2520.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 2518.25 | 2520.61 | 2520.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 13:15:00 | 2503.74 | 2511.74 | 2515.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 2510.62 | 2509.73 | 2513.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 2510.62 | 2509.73 | 2513.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 2510.62 | 2509.73 | 2513.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 10:15:00 | 2504.38 | 2509.73 | 2513.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 14:15:00 | 2525.72 | 2513.04 | 2513.31 | SL hit (close>static) qty=1.00 sl=2524.59 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 15:15:00 | 2525.87 | 2515.61 | 2514.45 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 09:15:00 | 2454.16 | 2503.32 | 2508.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 10:15:00 | 2442.46 | 2491.15 | 2502.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 2383.63 | 2370.68 | 2408.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 10:30:00 | 2379.50 | 2370.68 | 2408.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 2405.08 | 2387.43 | 2402.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 2425.09 | 2387.43 | 2402.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 2423.67 | 2394.68 | 2404.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 13:15:00 | 2395.24 | 2402.84 | 2406.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 09:30:00 | 2392.98 | 2399.18 | 2403.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 11:00:00 | 2397.45 | 2398.84 | 2402.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 14:15:00 | 2396.42 | 2398.48 | 2401.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 2406.06 | 2399.99 | 2401.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 15:00:00 | 2406.06 | 2399.99 | 2401.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 15:15:00 | 2400.16 | 2400.03 | 2401.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:15:00 | 2430.90 | 2400.03 | 2401.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-30 09:15:00 | 2449.39 | 2409.90 | 2406.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 2449.39 | 2409.90 | 2406.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-01 10:15:00 | 2464.34 | 2438.49 | 2427.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 12:15:00 | 2438.96 | 2440.41 | 2430.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 13:00:00 | 2438.96 | 2440.41 | 2430.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 2432.08 | 2439.08 | 2431.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 15:00:00 | 2432.08 | 2439.08 | 2431.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 2431.24 | 2437.51 | 2431.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:15:00 | 2445.46 | 2437.51 | 2431.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 2431.49 | 2436.31 | 2431.71 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 13:15:00 | 2407.53 | 2426.76 | 2428.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 2401.68 | 2418.05 | 2423.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 12:15:00 | 2388.85 | 2386.56 | 2394.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-07 13:00:00 | 2388.85 | 2386.56 | 2394.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 2377.48 | 2385.17 | 2391.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 10:30:00 | 2365.43 | 2381.17 | 2389.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 10:00:00 | 2371.14 | 2380.27 | 2382.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 11:00:00 | 2371.34 | 2378.48 | 2381.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 11:30:00 | 2371.68 | 2376.52 | 2380.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 2350.29 | 2345.29 | 2352.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-15 09:15:00 | 2338.19 | 2345.29 | 2352.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 10:15:00 | 2350.29 | 2337.14 | 2336.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 10:15:00 | 2350.29 | 2337.14 | 2336.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 12:15:00 | 2353.29 | 2342.65 | 2339.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 2332.04 | 2342.98 | 2340.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 2332.04 | 2342.98 | 2340.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 2332.04 | 2342.98 | 2340.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 2332.04 | 2342.98 | 2340.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 2337.35 | 2341.85 | 2340.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:45:00 | 2334.25 | 2341.85 | 2340.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 2347.43 | 2344.99 | 2342.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:00:00 | 2347.43 | 2344.99 | 2342.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 2371.29 | 2367.56 | 2358.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:45:00 | 2360.12 | 2367.56 | 2358.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 2341.19 | 2361.84 | 2357.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:00:00 | 2341.19 | 2361.84 | 2357.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 2330.81 | 2355.63 | 2354.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:45:00 | 2331.69 | 2355.63 | 2354.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 11:15:00 | 2330.81 | 2350.67 | 2352.48 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 12:15:00 | 2354.61 | 2351.19 | 2351.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 15:15:00 | 2365.73 | 2355.14 | 2353.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 2346.35 | 2353.38 | 2352.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 2346.35 | 2353.38 | 2352.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 2346.35 | 2353.38 | 2352.45 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 11:15:00 | 2344.14 | 2350.79 | 2351.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 12:15:00 | 2342.02 | 2349.04 | 2350.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 13:15:00 | 2355.45 | 2350.32 | 2350.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 13:15:00 | 2355.45 | 2350.32 | 2350.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 13:15:00 | 2355.45 | 2350.32 | 2350.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 14:00:00 | 2355.45 | 2350.32 | 2350.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 14:15:00 | 2366.07 | 2353.47 | 2352.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 15:15:00 | 2368.14 | 2361.22 | 2357.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-29 09:15:00 | 2360.96 | 2372.01 | 2366.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 09:15:00 | 2360.96 | 2372.01 | 2366.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 2360.96 | 2372.01 | 2366.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 10:00:00 | 2360.96 | 2372.01 | 2366.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 2377.34 | 2373.07 | 2367.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 13:45:00 | 2380.14 | 2374.82 | 2369.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 09:45:00 | 2384.86 | 2377.62 | 2372.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-02 09:45:00 | 2378.52 | 2378.10 | 2375.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 10:15:00 | 2381.91 | 2377.28 | 2375.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 2385.50 | 2378.92 | 2376.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-05 09:15:00 | 2360.71 | 2376.87 | 2377.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 2360.71 | 2376.87 | 2377.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 10:15:00 | 2358.11 | 2373.12 | 2375.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 2360.37 | 2355.16 | 2361.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 14:15:00 | 2360.37 | 2355.16 | 2361.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 2360.37 | 2355.16 | 2361.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 2360.37 | 2355.16 | 2361.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 2360.81 | 2356.29 | 2361.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 2377.68 | 2356.29 | 2361.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 2366.71 | 2358.37 | 2362.11 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 2380.78 | 2366.05 | 2365.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 12:15:00 | 2383.68 | 2369.58 | 2366.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 2365.73 | 2372.49 | 2369.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 09:15:00 | 2365.73 | 2372.49 | 2369.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 2365.73 | 2372.49 | 2369.55 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 13:15:00 | 2357.37 | 2366.18 | 2367.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 2346.06 | 2362.15 | 2365.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 2296.33 | 2294.95 | 2313.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:30:00 | 2296.77 | 2294.95 | 2313.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 10:15:00 | 2211.39 | 2211.55 | 2220.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 10:30:00 | 2211.29 | 2211.55 | 2220.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 2223.88 | 2214.01 | 2221.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 11:45:00 | 2222.16 | 2214.01 | 2221.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 12:15:00 | 2214.93 | 2214.20 | 2220.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 13:30:00 | 2212.82 | 2213.61 | 2219.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-22 15:15:00 | 2225.06 | 2216.79 | 2220.24 | SL hit (close>static) qty=1.00 sl=2225.01 alert=retest2 |

### Cycle 58 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 11:15:00 | 2229.79 | 2214.46 | 2213.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 2235.84 | 2218.73 | 2215.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 10:15:00 | 2242.87 | 2244.58 | 2235.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-02 11:00:00 | 2242.87 | 2244.58 | 2235.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 2230.38 | 2243.96 | 2239.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:45:00 | 2226.88 | 2243.96 | 2239.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 2233.43 | 2241.85 | 2239.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 11:15:00 | 2235.98 | 2241.85 | 2239.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-03 14:15:00 | 2228.06 | 2237.11 | 2237.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 14:15:00 | 2228.06 | 2237.11 | 2237.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 09:15:00 | 2215.77 | 2231.59 | 2234.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 11:15:00 | 2232.59 | 2231.06 | 2234.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-04 12:00:00 | 2232.59 | 2231.06 | 2234.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 2235.54 | 2231.96 | 2234.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:30:00 | 2240.80 | 2231.96 | 2234.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 2234.51 | 2232.47 | 2234.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 13:30:00 | 2239.82 | 2232.47 | 2234.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 2228.16 | 2231.61 | 2233.70 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-04-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 12:15:00 | 2242.23 | 2234.57 | 2234.35 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 13:15:00 | 2231.75 | 2234.01 | 2234.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 14:15:00 | 2230.52 | 2233.31 | 2233.79 | Break + close below crossover candle low |

### Cycle 62 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 2245.23 | 2235.08 | 2234.47 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 10:15:00 | 2229.83 | 2233.95 | 2234.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 11:15:00 | 2222.60 | 2231.68 | 2233.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 11:15:00 | 2180.95 | 2168.61 | 2182.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 12:00:00 | 2180.95 | 2168.61 | 2182.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 2181.63 | 2173.36 | 2181.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 15:00:00 | 2181.63 | 2173.36 | 2181.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 2193.78 | 2177.44 | 2182.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 2199.49 | 2177.44 | 2182.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 2180.60 | 2178.07 | 2182.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:30:00 | 2197.91 | 2178.07 | 2182.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 2190.93 | 2180.64 | 2183.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:00:00 | 2190.93 | 2180.64 | 2183.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 2194.08 | 2183.33 | 2184.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:45:00 | 2194.23 | 2183.33 | 2184.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 12:15:00 | 2194.57 | 2185.58 | 2185.20 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-04-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 15:15:00 | 2180.75 | 2184.35 | 2184.79 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 09:15:00 | 2191.08 | 2185.70 | 2185.36 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 2182.13 | 2184.98 | 2185.07 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 11:15:00 | 2191.67 | 2186.32 | 2185.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 12:15:00 | 2198.65 | 2188.79 | 2186.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 09:15:00 | 2222.06 | 2223.10 | 2213.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 09:15:00 | 2222.06 | 2223.10 | 2213.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 2222.06 | 2223.10 | 2213.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:30:00 | 2219.41 | 2223.10 | 2213.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 11:15:00 | 2216.65 | 2220.47 | 2213.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 13:15:00 | 2220.74 | 2219.58 | 2213.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 14:00:00 | 2221.62 | 2219.99 | 2214.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 09:15:00 | 2188.37 | 2214.77 | 2213.71 | SL hit (close<static) qty=1.00 sl=2208.98 alert=retest2 |

### Cycle 69 — SELL (started 2024-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 10:15:00 | 2183.90 | 2208.60 | 2211.00 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 09:15:00 | 2205.29 | 2198.38 | 2197.58 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 14:15:00 | 2187.73 | 2195.97 | 2196.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 09:15:00 | 2179.82 | 2192.05 | 2194.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 2193.14 | 2183.12 | 2186.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 2193.14 | 2183.12 | 2186.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 2193.14 | 2183.12 | 2186.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 2193.14 | 2183.12 | 2186.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 2200.57 | 2186.61 | 2188.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:45:00 | 2202.34 | 2186.61 | 2188.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 13:15:00 | 2212.33 | 2193.40 | 2190.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 14:15:00 | 2218.42 | 2198.40 | 2193.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 10:15:00 | 2293.23 | 2302.69 | 2267.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:30:00 | 2293.04 | 2302.69 | 2267.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 2289.99 | 2295.47 | 2285.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 14:15:00 | 2291.31 | 2295.47 | 2285.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 14:45:00 | 2290.23 | 2293.94 | 2285.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:15:00 | 2315.76 | 2291.77 | 2285.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 10:15:00 | 2297.56 | 2310.35 | 2311.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 10:15:00 | 2297.56 | 2310.35 | 2311.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 14:15:00 | 2284.23 | 2296.99 | 2303.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 2287.13 | 2286.73 | 2296.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 12:00:00 | 2287.13 | 2286.73 | 2296.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 2303.31 | 2287.80 | 2294.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 2303.31 | 2287.80 | 2294.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 2300.36 | 2290.31 | 2294.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 2286.45 | 2290.31 | 2294.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 2293.23 | 2290.45 | 2293.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:30:00 | 2293.28 | 2290.45 | 2293.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 2291.95 | 2290.75 | 2293.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:30:00 | 2294.66 | 2290.75 | 2293.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 2289.40 | 2290.48 | 2293.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 14:45:00 | 2283.59 | 2289.19 | 2292.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 2273.36 | 2289.01 | 2291.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 09:15:00 | 2322.94 | 2285.36 | 2285.66 | SL hit (close>static) qty=1.00 sl=2293.43 alert=retest2 |

### Cycle 74 — BUY (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 10:15:00 | 2327.86 | 2293.86 | 2289.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 2339.61 | 2326.50 | 2313.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 2329.43 | 2331.39 | 2319.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 10:00:00 | 2329.43 | 2331.39 | 2319.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 2334.74 | 2332.81 | 2326.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:30:00 | 2333.07 | 2332.81 | 2326.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 2338.38 | 2353.14 | 2347.49 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 2338.43 | 2344.31 | 2344.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 2333.46 | 2342.14 | 2343.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 2322.15 | 2320.71 | 2328.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 2322.15 | 2320.71 | 2328.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 2322.15 | 2320.71 | 2328.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 2332.97 | 2320.71 | 2328.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 2320.48 | 2320.67 | 2328.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 13:45:00 | 2309.17 | 2316.17 | 2324.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:30:00 | 2307.74 | 2315.87 | 2319.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 2367.94 | 2326.79 | 2323.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 09:15:00 | 2367.94 | 2326.79 | 2323.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 11:15:00 | 2398.93 | 2349.99 | 2335.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 2510.33 | 2537.97 | 2479.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 2510.33 | 2537.97 | 2479.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 2510.33 | 2537.97 | 2479.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 2494.59 | 2537.97 | 2479.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 2504.67 | 2512.83 | 2487.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 2493.11 | 2512.83 | 2487.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 2499.36 | 2510.21 | 2490.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:45:00 | 2498.62 | 2510.21 | 2490.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 2512.39 | 2523.44 | 2508.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 2512.39 | 2523.44 | 2508.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 2524.94 | 2523.74 | 2510.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 13:45:00 | 2531.77 | 2525.08 | 2514.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 2530.99 | 2524.46 | 2515.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:45:00 | 2530.64 | 2525.06 | 2519.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 09:15:00 | 2497.54 | 2516.11 | 2516.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 09:15:00 | 2497.54 | 2516.11 | 2516.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 11:15:00 | 2489.87 | 2507.07 | 2512.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 10:15:00 | 2451.21 | 2445.31 | 2457.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-18 11:00:00 | 2451.21 | 2445.31 | 2457.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 2437.14 | 2443.63 | 2451.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 2433.60 | 2443.63 | 2451.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 12:15:00 | 2434.09 | 2440.95 | 2448.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 13:15:00 | 2433.90 | 2440.05 | 2447.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 10:30:00 | 2430.95 | 2432.78 | 2440.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 2440.49 | 2434.32 | 2440.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 2417.86 | 2438.96 | 2440.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 2421.45 | 2409.19 | 2408.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 2421.45 | 2409.19 | 2408.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 10:15:00 | 2438.96 | 2415.15 | 2410.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 15:15:00 | 2423.77 | 2423.89 | 2417.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:15:00 | 2438.77 | 2423.89 | 2417.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 2426.86 | 2433.82 | 2427.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-28 15:15:00 | 2426.86 | 2433.82 | 2427.41 | SL hit (close<ema400) qty=1.00 sl=2427.41 alert=retest1 |

### Cycle 79 — SELL (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 13:15:00 | 2666.58 | 2690.42 | 2691.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 10:15:00 | 2634.76 | 2673.03 | 2682.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 13:15:00 | 2667.77 | 2663.94 | 2675.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 14:00:00 | 2667.77 | 2663.94 | 2675.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 2666.14 | 2664.17 | 2673.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 2647.65 | 2664.17 | 2673.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 2661.08 | 2663.55 | 2672.16 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 2685.42 | 2669.08 | 2668.54 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 2647.94 | 2667.48 | 2668.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 09:15:00 | 2638.50 | 2658.48 | 2664.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 13:15:00 | 2655.91 | 2652.15 | 2658.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-31 14:00:00 | 2655.91 | 2652.15 | 2658.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 2662.21 | 2654.16 | 2659.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 2662.21 | 2654.16 | 2659.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 2662.80 | 2655.89 | 2659.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 2673.91 | 2655.89 | 2659.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 2664.37 | 2657.58 | 2659.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 2674.65 | 2657.58 | 2659.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 2655.17 | 2657.10 | 2659.38 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 13:15:00 | 2668.70 | 2660.50 | 2660.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 14:15:00 | 2673.91 | 2663.18 | 2661.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 10:15:00 | 2663.68 | 2667.78 | 2664.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 10:15:00 | 2663.68 | 2667.78 | 2664.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 2663.68 | 2667.78 | 2664.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:00:00 | 2663.68 | 2667.78 | 2664.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 2654.63 | 2665.15 | 2663.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:30:00 | 2653.01 | 2665.15 | 2663.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 2650.80 | 2662.28 | 2662.44 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 09:15:00 | 2688.62 | 2663.68 | 2662.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 09:15:00 | 2705.39 | 2679.29 | 2672.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 14:15:00 | 2700.37 | 2700.92 | 2687.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-06 15:00:00 | 2700.37 | 2700.92 | 2687.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 2703.77 | 2708.47 | 2699.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:15:00 | 2697.72 | 2708.47 | 2699.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 2697.72 | 2706.32 | 2699.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 2706.67 | 2706.32 | 2699.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 2690.34 | 2703.12 | 2698.47 | SL hit (close<static) qty=1.00 sl=2690.83 alert=retest2 |

### Cycle 85 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 2687.29 | 2695.18 | 2695.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 15:15:00 | 2685.52 | 2693.24 | 2694.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 12:15:00 | 2690.34 | 2688.76 | 2692.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 2690.34 | 2688.76 | 2692.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 2689.26 | 2688.86 | 2691.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 2689.26 | 2688.86 | 2691.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 2703.77 | 2691.84 | 2692.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 2703.77 | 2691.84 | 2692.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 15:15:00 | 2700.18 | 2693.51 | 2693.51 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 2681.19 | 2691.05 | 2692.39 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 11:15:00 | 2705.10 | 2693.66 | 2693.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 2713.55 | 2699.54 | 2696.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 10:15:00 | 2701.36 | 2702.35 | 2698.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 10:15:00 | 2701.36 | 2702.35 | 2698.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 2701.36 | 2702.35 | 2698.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:00:00 | 2701.36 | 2702.35 | 2698.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 2708.49 | 2703.58 | 2699.67 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 2670.72 | 2692.97 | 2695.76 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 2704.11 | 2691.56 | 2690.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 2705.93 | 2694.44 | 2691.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 2700.18 | 2701.44 | 2696.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 15:00:00 | 2700.18 | 2701.44 | 2696.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 2694.28 | 2700.01 | 2696.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 2703.08 | 2700.01 | 2696.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 2707.06 | 2701.42 | 2697.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 2719.16 | 2699.95 | 2698.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 13:15:00 | 2733.77 | 2751.90 | 2753.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 2733.77 | 2751.90 | 2753.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 2725.85 | 2746.69 | 2751.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 13:15:00 | 2733.57 | 2726.45 | 2736.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-28 14:00:00 | 2733.57 | 2726.45 | 2736.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 2719.26 | 2725.01 | 2735.19 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 12:15:00 | 2761.12 | 2737.56 | 2737.53 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 2734.61 | 2736.97 | 2737.27 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 2746.16 | 2738.95 | 2738.09 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 2727.62 | 2736.37 | 2737.29 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 2758.56 | 2740.90 | 2739.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 10:15:00 | 2778.08 | 2753.38 | 2747.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 2747.34 | 2757.14 | 2751.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 2747.34 | 2757.14 | 2751.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 2747.34 | 2757.14 | 2751.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 2747.34 | 2757.14 | 2751.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 2749.31 | 2755.57 | 2751.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 2750.05 | 2755.57 | 2751.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 2794.07 | 2790.80 | 2779.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:45:00 | 2781.53 | 2790.80 | 2779.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 2791.41 | 2795.89 | 2788.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 2789.84 | 2795.89 | 2788.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 2855.84 | 2867.16 | 2854.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 2855.84 | 2867.16 | 2854.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 2863.47 | 2866.42 | 2855.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 2866.32 | 2866.42 | 2855.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 2872.17 | 2867.57 | 2857.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:00:00 | 2880.34 | 2870.32 | 2861.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 2827.46 | 2874.76 | 2873.32 | SL hit (close<static) qty=1.00 sl=2846.94 alert=retest2 |

### Cycle 97 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 2816.25 | 2863.06 | 2868.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 2803.31 | 2851.11 | 2862.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 10:15:00 | 2833.22 | 2830.62 | 2844.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 11:00:00 | 2833.22 | 2830.62 | 2844.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 2831.20 | 2828.92 | 2837.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 13:00:00 | 2824.56 | 2829.62 | 2835.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 15:15:00 | 2823.14 | 2828.17 | 2834.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 2853.33 | 2832.40 | 2834.89 | SL hit (close>static) qty=1.00 sl=2846.30 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 10:15:00 | 2880.63 | 2842.04 | 2839.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 2892.19 | 2867.06 | 2855.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 2936.01 | 2950.39 | 2921.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 10:00:00 | 2936.01 | 2950.39 | 2921.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 2924.80 | 2943.80 | 2923.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:00:00 | 2924.80 | 2943.80 | 2923.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 2931.14 | 2941.27 | 2924.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 13:15:00 | 2932.23 | 2941.27 | 2924.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 13:15:00 | 2917.17 | 2936.45 | 2923.66 | SL hit (close<static) qty=1.00 sl=2918.55 alert=retest2 |

### Cycle 99 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 2879.89 | 2915.46 | 2916.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 10:15:00 | 2870.79 | 2906.53 | 2912.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 2902.67 | 2894.74 | 2903.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 14:15:00 | 2902.67 | 2894.74 | 2903.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 2902.67 | 2894.74 | 2903.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 2902.67 | 2894.74 | 2903.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 2900.90 | 2895.98 | 2903.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 2914.62 | 2895.98 | 2903.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 2904.44 | 2897.67 | 2903.45 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 14:15:00 | 2936.50 | 2907.65 | 2906.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 15:15:00 | 2942.06 | 2914.54 | 2909.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 13:15:00 | 2943.24 | 2943.52 | 2928.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 14:00:00 | 2943.24 | 2943.52 | 2928.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 2919.58 | 2938.73 | 2927.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 2919.58 | 2938.73 | 2927.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 2916.58 | 2934.30 | 2926.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 2923.22 | 2934.30 | 2926.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 11:15:00 | 2907.49 | 2927.25 | 2925.15 | SL hit (close<static) qty=1.00 sl=2908.17 alert=retest2 |

### Cycle 101 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 2907.73 | 2923.35 | 2923.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 2888.65 | 2910.68 | 2916.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2826.14 | 2803.67 | 2821.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 2826.14 | 2803.67 | 2821.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 2826.14 | 2803.67 | 2821.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:30:00 | 2788.71 | 2799.28 | 2814.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 13:30:00 | 2783.44 | 2794.42 | 2811.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 13:15:00 | 2746.85 | 2739.76 | 2738.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 13:15:00 | 2746.85 | 2739.76 | 2738.98 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 10:15:00 | 2737.11 | 2738.62 | 2738.79 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 12:15:00 | 2745.23 | 2739.93 | 2739.35 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 2729.83 | 2738.16 | 2738.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 2693.88 | 2718.95 | 2728.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 15:15:00 | 2678.49 | 2678.09 | 2695.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 09:15:00 | 2681.88 | 2678.09 | 2695.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 2654.04 | 2673.28 | 2691.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:30:00 | 2664.81 | 2673.28 | 2691.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 2527.54 | 2494.97 | 2519.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 2527.54 | 2494.97 | 2519.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 2532.07 | 2502.39 | 2520.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:15:00 | 2518.49 | 2524.19 | 2526.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 2462.33 | 2459.07 | 2458.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 2462.33 | 2459.07 | 2458.78 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 2440.34 | 2455.42 | 2457.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 2437.34 | 2450.36 | 2454.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 13:15:00 | 2368.93 | 2364.16 | 2383.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 14:00:00 | 2368.93 | 2364.16 | 2383.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 2384.47 | 2368.22 | 2383.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 15:00:00 | 2384.47 | 2368.22 | 2383.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 2384.62 | 2371.50 | 2383.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 2385.99 | 2371.50 | 2383.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 2393.32 | 2377.44 | 2384.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 2393.32 | 2377.44 | 2384.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 2391.16 | 2380.18 | 2384.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 2377.24 | 2381.62 | 2384.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 14:15:00 | 2407.98 | 2375.50 | 2372.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 2407.98 | 2375.50 | 2372.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 2448.16 | 2393.98 | 2381.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 2438.67 | 2440.03 | 2423.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 15:00:00 | 2438.67 | 2440.03 | 2423.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 2422.39 | 2436.14 | 2424.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 2422.39 | 2436.14 | 2424.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 2430.65 | 2435.04 | 2425.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:30:00 | 2436.21 | 2435.59 | 2427.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 2434.44 | 2435.12 | 2434.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:15:00 | 2435.13 | 2445.19 | 2442.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:45:00 | 2432.91 | 2442.94 | 2441.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 12:15:00 | 2424.75 | 2439.07 | 2439.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 12:15:00 | 2424.75 | 2439.07 | 2439.93 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 2441.47 | 2439.82 | 2439.61 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 09:15:00 | 2437.39 | 2439.34 | 2439.41 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 2446.44 | 2440.76 | 2440.05 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 12:15:00 | 2435.77 | 2438.95 | 2439.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 2425.19 | 2436.21 | 2438.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 2447.13 | 2433.44 | 2435.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 11:15:00 | 2447.13 | 2433.44 | 2435.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 2447.13 | 2433.44 | 2435.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:00:00 | 2447.13 | 2433.44 | 2435.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 2447.87 | 2436.32 | 2436.68 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 2459.82 | 2441.02 | 2438.79 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 2354.52 | 2425.39 | 2433.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 2325.55 | 2352.20 | 2365.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 2340.06 | 2321.00 | 2337.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 2340.06 | 2321.00 | 2337.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 2340.06 | 2321.00 | 2337.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 2340.06 | 2321.00 | 2337.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 2330.22 | 2322.84 | 2336.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:00:00 | 2321.27 | 2328.42 | 2333.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:00:00 | 2323.14 | 2326.59 | 2331.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:15:00 | 2324.51 | 2328.12 | 2332.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:45:00 | 2322.74 | 2327.59 | 2331.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 2329.78 | 2327.68 | 2330.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 2333.22 | 2327.68 | 2330.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 2327.27 | 2327.60 | 2330.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:15:00 | 2337.10 | 2327.60 | 2330.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 2331.50 | 2328.38 | 2330.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 13:30:00 | 2320.18 | 2326.69 | 2329.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:00:00 | 2322.30 | 2326.69 | 2329.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 14:45:00 | 2321.81 | 2318.57 | 2322.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 2313.59 | 2319.90 | 2322.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2307.10 | 2317.34 | 2321.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 2330.32 | 2317.34 | 2321.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 2321.46 | 2318.17 | 2321.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:45:00 | 2323.04 | 2318.17 | 2321.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 2322.05 | 2318.94 | 2321.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 2322.05 | 2318.94 | 2321.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 2308.87 | 2316.93 | 2320.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:45:00 | 2321.37 | 2316.93 | 2320.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 2290.97 | 2311.83 | 2317.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:30:00 | 2322.45 | 2311.83 | 2317.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 2311.63 | 2301.94 | 2307.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 2311.63 | 2301.94 | 2307.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 2311.09 | 2303.77 | 2307.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 2311.53 | 2303.77 | 2307.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 2309.95 | 2305.01 | 2307.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:45:00 | 2316.15 | 2305.01 | 2307.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 2301.59 | 2304.87 | 2307.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:00:00 | 2294.95 | 2302.89 | 2306.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:30:00 | 2296.28 | 2298.39 | 2301.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:00:00 | 2297.07 | 2296.61 | 2300.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 15:15:00 | 2305.72 | 2301.37 | 2301.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 2305.72 | 2301.37 | 2301.28 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 2296.04 | 2300.31 | 2300.80 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 2305.97 | 2301.44 | 2301.27 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 2292.30 | 2299.82 | 2300.58 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 2306.91 | 2301.24 | 2301.16 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 2299.82 | 2300.96 | 2301.04 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 09:15:00 | 2303.86 | 2301.54 | 2301.29 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 14:15:00 | 2289.74 | 2302.08 | 2302.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 13:15:00 | 2286.59 | 2293.22 | 2297.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 09:15:00 | 2293.77 | 2291.02 | 2294.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 2293.77 | 2291.02 | 2294.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 2293.77 | 2291.02 | 2294.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:15:00 | 2296.82 | 2291.02 | 2294.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 2308.68 | 2294.55 | 2296.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 2308.68 | 2294.55 | 2296.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 2314.33 | 2298.51 | 2297.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 12:15:00 | 2326.87 | 2304.18 | 2300.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 2344.83 | 2347.84 | 2331.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 2344.83 | 2347.84 | 2331.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2344.83 | 2347.84 | 2331.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2334.40 | 2347.84 | 2331.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 2332.19 | 2344.71 | 2331.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 2332.19 | 2344.71 | 2331.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 2342.12 | 2344.19 | 2332.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 2326.28 | 2344.19 | 2332.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 2329.73 | 2340.98 | 2333.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:00:00 | 2329.73 | 2340.98 | 2333.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 2334.15 | 2339.61 | 2333.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 2331.25 | 2339.61 | 2333.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 2340.79 | 2339.85 | 2334.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:45:00 | 2347.53 | 2342.65 | 2335.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:45:00 | 2350.48 | 2344.04 | 2337.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 14:00:00 | 2347.19 | 2346.46 | 2340.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 14:30:00 | 2350.88 | 2347.40 | 2341.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2345.37 | 2347.60 | 2342.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:45:00 | 2341.43 | 2347.60 | 2342.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 2410.04 | 2366.11 | 2354.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 2420.91 | 2366.11 | 2354.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 2410.73 | 2384.76 | 2365.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:00:00 | 2410.54 | 2399.23 | 2384.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:30:00 | 2411.47 | 2401.23 | 2386.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 2410.98 | 2402.79 | 2391.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 2358.50 | 2385.87 | 2388.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 2358.50 | 2385.87 | 2388.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 2351.22 | 2378.94 | 2385.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 14:15:00 | 2336.22 | 2334.96 | 2351.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 15:00:00 | 2336.22 | 2334.96 | 2351.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 2329.33 | 2315.27 | 2328.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 2329.33 | 2315.27 | 2328.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 2333.27 | 2318.87 | 2328.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 2333.27 | 2318.87 | 2328.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 2337.60 | 2322.62 | 2329.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:00:00 | 2337.60 | 2322.62 | 2329.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 2326.53 | 2323.40 | 2329.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 13:15:00 | 2321.22 | 2323.40 | 2329.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:00:00 | 2321.76 | 2313.10 | 2318.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 13:30:00 | 2321.96 | 2316.68 | 2318.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 2320.73 | 2301.93 | 2301.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 11:15:00 | 2320.73 | 2301.93 | 2301.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 15:15:00 | 2341.24 | 2318.92 | 2310.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-28 14:15:00 | 2350.38 | 2351.27 | 2341.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-28 14:45:00 | 2351.96 | 2351.27 | 2341.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 2338.68 | 2347.61 | 2341.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 2334.55 | 2347.61 | 2341.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 2343.74 | 2346.84 | 2341.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 12:00:00 | 2351.91 | 2347.85 | 2342.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 13:15:00 | 2351.96 | 2347.68 | 2342.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 09:15:00 | 2353.83 | 2347.69 | 2344.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 10:00:00 | 2357.71 | 2349.69 | 2345.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 2452.73 | 2423.92 | 2397.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 2409.21 | 2423.92 | 2397.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 2413.34 | 2449.34 | 2427.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 2413.34 | 2449.34 | 2427.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 2388.30 | 2437.14 | 2424.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:00:00 | 2388.30 | 2437.14 | 2424.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-03 14:15:00 | 2400.26 | 2417.17 | 2417.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 2400.26 | 2417.17 | 2417.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 2375.66 | 2406.98 | 2412.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 09:15:00 | 2345.07 | 2331.39 | 2343.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 2345.07 | 2331.39 | 2343.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 2345.07 | 2331.39 | 2343.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 2345.07 | 2331.39 | 2343.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 2346.50 | 2334.41 | 2343.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:30:00 | 2353.29 | 2334.41 | 2343.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 2333.17 | 2334.16 | 2342.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 12:15:00 | 2328.94 | 2334.16 | 2342.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:00:00 | 2319.59 | 2329.11 | 2338.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:45:00 | 2325.89 | 2323.62 | 2333.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 12:15:00 | 2212.49 | 2243.25 | 2260.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 09:15:00 | 2203.61 | 2224.46 | 2245.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 09:15:00 | 2209.60 | 2224.46 | 2245.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-21 14:15:00 | 2209.23 | 2207.38 | 2218.79 | SL hit (close>ema200) qty=0.50 sl=2207.38 alert=retest2 |

### Cycle 128 — BUY (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 12:15:00 | 2227.82 | 2214.09 | 2213.22 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 2198.50 | 2211.63 | 2213.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 2169.09 | 2201.54 | 2207.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 2129.89 | 2125.00 | 2140.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 11:15:00 | 2144.16 | 2130.14 | 2140.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 2144.16 | 2130.14 | 2140.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 2144.16 | 2130.14 | 2140.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 2153.70 | 2134.86 | 2141.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:30:00 | 2153.40 | 2134.86 | 2141.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 2134.71 | 2135.81 | 2140.80 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 11:15:00 | 2170.08 | 2147.83 | 2145.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 12:15:00 | 2185.47 | 2155.36 | 2148.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 10:15:00 | 2171.11 | 2171.54 | 2160.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 10:15:00 | 2171.11 | 2171.54 | 2160.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 2171.11 | 2171.54 | 2160.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:45:00 | 2164.13 | 2171.54 | 2160.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 2165.55 | 2170.34 | 2161.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:45:00 | 2167.72 | 2170.34 | 2161.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2210.31 | 2179.06 | 2168.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:15:00 | 2218.18 | 2185.71 | 2172.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 09:45:00 | 2223.10 | 2209.57 | 2192.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 12:15:00 | 2178.49 | 2189.48 | 2190.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 12:15:00 | 2178.49 | 2189.48 | 2190.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 13:15:00 | 2160.19 | 2183.62 | 2187.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 2162.01 | 2143.07 | 2151.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 2162.01 | 2143.07 | 2151.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 2162.01 | 2143.07 | 2151.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 2162.01 | 2143.07 | 2151.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 2167.91 | 2148.04 | 2153.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 2169.93 | 2148.04 | 2153.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 2169.49 | 2156.84 | 2156.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 2187.19 | 2168.90 | 2163.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 10:15:00 | 2205.88 | 2206.16 | 2195.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 11:00:00 | 2205.88 | 2206.16 | 2195.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 2211.88 | 2225.70 | 2220.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 2212.96 | 2225.70 | 2220.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 2210.21 | 2222.60 | 2219.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:45:00 | 2204.65 | 2222.60 | 2219.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 2204.41 | 2216.88 | 2217.34 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 2247.89 | 2218.53 | 2216.92 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 2189.65 | 2216.15 | 2218.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 2185.37 | 2210.00 | 2215.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 2198.46 | 2196.65 | 2204.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 13:15:00 | 2200.52 | 2197.42 | 2204.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 2200.52 | 2197.42 | 2204.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 2200.52 | 2197.42 | 2204.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 2202.19 | 2198.37 | 2203.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 2202.19 | 2198.37 | 2203.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 2200.47 | 2198.79 | 2203.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 2190.98 | 2198.79 | 2203.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 2205.54 | 2200.14 | 2203.72 | SL hit (close>static) qty=1.00 sl=2205.10 alert=retest2 |

### Cycle 136 — BUY (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 09:15:00 | 2217.19 | 2206.64 | 2205.61 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 2175.09 | 2200.17 | 2203.27 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 14:15:00 | 2210.75 | 2203.73 | 2203.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 2226.24 | 2209.76 | 2206.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 09:15:00 | 2309.17 | 2315.02 | 2294.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 10:00:00 | 2309.17 | 2315.02 | 2294.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 2308.28 | 2322.33 | 2316.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:00:00 | 2333.37 | 2322.89 | 2317.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 14:15:00 | 2311.82 | 2319.02 | 2319.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 14:15:00 | 2311.82 | 2319.02 | 2319.59 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 2337.30 | 2321.80 | 2320.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 11:15:00 | 2349.40 | 2329.44 | 2324.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 10:15:00 | 2294.71 | 2364.90 | 2356.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 10:15:00 | 2294.71 | 2364.90 | 2356.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 2294.71 | 2364.90 | 2356.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:00:00 | 2294.71 | 2364.90 | 2356.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 2293.04 | 2350.53 | 2350.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:30:00 | 2289.79 | 2350.53 | 2350.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 12:15:00 | 2294.41 | 2339.31 | 2345.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 2269.72 | 2305.14 | 2325.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 14:15:00 | 2295.79 | 2286.85 | 2306.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 15:00:00 | 2295.79 | 2286.85 | 2306.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 2295.49 | 2289.09 | 2304.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 12:45:00 | 2282.12 | 2288.45 | 2300.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:45:00 | 2281.82 | 2286.39 | 2297.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:30:00 | 2281.63 | 2283.67 | 2294.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:45:00 | 2283.10 | 2283.89 | 2291.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 2282.02 | 2284.26 | 2290.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:45:00 | 2287.72 | 2284.26 | 2290.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 2298.25 | 2287.50 | 2290.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:30:00 | 2300.51 | 2287.50 | 2290.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 2303.07 | 2290.61 | 2291.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:00:00 | 2303.07 | 2290.61 | 2291.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-30 11:15:00 | 2308.58 | 2294.21 | 2293.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 11:15:00 | 2308.58 | 2294.21 | 2293.49 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 12:15:00 | 2277.20 | 2292.52 | 2294.22 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 2314.48 | 2295.84 | 2294.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 2328.45 | 2309.02 | 2302.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 2327.37 | 2330.02 | 2318.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 2327.37 | 2330.02 | 2318.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2327.37 | 2330.02 | 2318.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:45:00 | 2339.07 | 2333.10 | 2322.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 10:15:00 | 2316.74 | 2328.31 | 2324.97 | SL hit (close<static) qty=1.00 sl=2316.84 alert=retest2 |

### Cycle 145 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 2308.97 | 2322.92 | 2323.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 2280.54 | 2310.89 | 2317.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 2297.86 | 2297.42 | 2306.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 2330.22 | 2297.42 | 2306.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2346.55 | 2307.25 | 2309.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 2344.28 | 2307.25 | 2309.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 2340.55 | 2313.91 | 2312.63 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 09:15:00 | 2309.56 | 2322.83 | 2323.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 12:15:00 | 2303.07 | 2316.06 | 2319.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 14:15:00 | 2314.77 | 2313.91 | 2318.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:30:00 | 2311.53 | 2313.91 | 2318.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 2307.79 | 2312.56 | 2316.70 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 2323.53 | 2317.00 | 2316.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 2336.22 | 2320.84 | 2318.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 2342.02 | 2342.98 | 2334.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 2342.02 | 2342.98 | 2334.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2335.24 | 2341.39 | 2335.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 2335.24 | 2341.39 | 2335.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2333.27 | 2339.76 | 2335.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 2333.27 | 2339.76 | 2335.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 2327.66 | 2337.34 | 2334.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 2326.68 | 2337.34 | 2334.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 2312.51 | 2329.43 | 2331.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 2301.99 | 2323.94 | 2328.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2327.76 | 2322.42 | 2327.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 2327.76 | 2322.42 | 2327.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2327.76 | 2322.42 | 2327.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 2324.41 | 2322.42 | 2327.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2330.51 | 2324.04 | 2327.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 2333.27 | 2324.04 | 2327.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2322.84 | 2323.80 | 2327.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 2287.04 | 2324.68 | 2326.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:45:00 | 2320.09 | 2307.04 | 2311.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 2342.12 | 2318.52 | 2315.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 2342.12 | 2318.52 | 2315.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 2350.09 | 2324.83 | 2318.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 2335.43 | 2345.10 | 2336.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 12:15:00 | 2335.43 | 2345.10 | 2336.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 2335.43 | 2345.10 | 2336.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 2335.43 | 2345.10 | 2336.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 2327.46 | 2341.57 | 2335.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 2327.46 | 2341.57 | 2335.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 2342.22 | 2341.70 | 2335.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 2344.09 | 2341.70 | 2335.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 2324.91 | 2333.05 | 2333.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 2324.91 | 2333.05 | 2333.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 10:15:00 | 2320.97 | 2330.08 | 2332.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 2332.68 | 2330.60 | 2332.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 11:15:00 | 2332.68 | 2330.60 | 2332.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 2332.68 | 2330.60 | 2332.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 2332.68 | 2330.60 | 2332.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 2323.63 | 2329.21 | 2331.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:45:00 | 2316.64 | 2324.18 | 2327.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 2316.64 | 2322.57 | 2326.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 2315.07 | 2321.07 | 2325.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 2306.12 | 2318.08 | 2323.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 2328.35 | 2318.47 | 2322.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 2332.87 | 2321.35 | 2323.89 | SL hit (close>static) qty=1.00 sl=2332.19 alert=retest2 |

### Cycle 152 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 2330.51 | 2325.58 | 2325.39 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 2314.68 | 2324.71 | 2325.18 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 2327.27 | 2320.76 | 2319.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 2337.01 | 2324.01 | 2321.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 2325.00 | 2325.41 | 2322.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 14:00:00 | 2325.00 | 2325.41 | 2322.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 2336.91 | 2327.71 | 2323.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 2342.22 | 2329.61 | 2325.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:30:00 | 2341.14 | 2337.43 | 2331.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:00:00 | 2339.46 | 2347.59 | 2347.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 15:15:00 | 2333.46 | 2344.76 | 2346.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 2333.46 | 2344.76 | 2346.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 2311.73 | 2338.15 | 2342.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 2290.77 | 2286.52 | 2301.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:30:00 | 2290.48 | 2286.52 | 2301.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 2290.18 | 2287.95 | 2292.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 2290.18 | 2287.95 | 2292.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2288.61 | 2288.41 | 2291.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 2280.64 | 2285.77 | 2290.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 2256.15 | 2245.49 | 2244.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 2256.15 | 2245.49 | 2244.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 2257.43 | 2249.40 | 2246.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 2252.71 | 2262.23 | 2256.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 2252.71 | 2262.23 | 2256.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 2252.71 | 2262.23 | 2256.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 2252.71 | 2262.23 | 2256.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 2254.87 | 2260.76 | 2256.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:30:00 | 2251.03 | 2260.76 | 2256.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 2258.41 | 2259.60 | 2256.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 2259.79 | 2259.60 | 2256.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 2264.41 | 2260.56 | 2257.38 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 2252.80 | 2256.61 | 2256.64 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 2261.46 | 2257.58 | 2257.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 2269.92 | 2261.55 | 2259.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 13:15:00 | 2353.73 | 2356.96 | 2336.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 15:15:00 | 2360.81 | 2356.76 | 2338.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 2365.63 | 2376.34 | 2366.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 2365.63 | 2376.34 | 2366.23 | SL hit (close<ema400) qty=1.00 sl=2366.23 alert=retest1 |

### Cycle 159 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 2448.46 | 2467.37 | 2469.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 2444.62 | 2457.16 | 2464.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 12:15:00 | 2430.75 | 2428.91 | 2439.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 12:45:00 | 2430.85 | 2428.91 | 2439.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 2439.11 | 2432.47 | 2439.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 2439.90 | 2432.47 | 2439.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 2441.96 | 2434.37 | 2439.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 2430.45 | 2434.37 | 2439.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2431.83 | 2433.86 | 2438.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 2424.55 | 2431.05 | 2437.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 2409.99 | 2399.10 | 2397.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 2409.99 | 2399.10 | 2397.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 2413.34 | 2401.94 | 2399.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 2395.14 | 2402.88 | 2400.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 2395.14 | 2402.88 | 2400.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 2395.14 | 2402.88 | 2400.14 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 11:15:00 | 2374.29 | 2393.89 | 2396.33 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 09:15:00 | 2483.87 | 2413.52 | 2404.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 2488.88 | 2446.94 | 2423.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 12:15:00 | 2505.71 | 2507.46 | 2487.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 2505.71 | 2507.46 | 2487.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 2481.61 | 2500.23 | 2490.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:15:00 | 2485.24 | 2500.23 | 2490.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2480.62 | 2496.31 | 2489.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:00:00 | 2480.62 | 2496.31 | 2489.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 2490.95 | 2495.24 | 2489.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 2499.02 | 2492.50 | 2489.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 11:15:00 | 2502.16 | 2495.48 | 2491.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 13:30:00 | 2501.48 | 2493.72 | 2491.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 2498.52 | 2493.75 | 2491.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 2498.52 | 2494.71 | 2492.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 2485.15 | 2494.71 | 2492.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 2481.21 | 2492.01 | 2491.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 2481.21 | 2492.01 | 2491.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 2477.97 | 2489.20 | 2490.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 2477.97 | 2489.20 | 2490.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2457.11 | 2480.21 | 2485.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 2468.23 | 2461.84 | 2468.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 11:15:00 | 2468.23 | 2461.84 | 2468.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2468.23 | 2461.84 | 2468.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:30:00 | 2466.75 | 2461.84 | 2468.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 2460.55 | 2461.58 | 2468.20 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 2490.66 | 2471.56 | 2471.16 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2468.03 | 2470.85 | 2470.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 11:15:00 | 2453.47 | 2467.38 | 2469.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 2454.46 | 2448.78 | 2455.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 14:00:00 | 2454.46 | 2448.78 | 2455.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 2452.19 | 2449.47 | 2455.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 2446.29 | 2453.65 | 2455.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 2520.56 | 2462.72 | 2458.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 2520.56 | 2462.72 | 2458.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 2552.92 | 2500.89 | 2479.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 2597.19 | 2602.96 | 2569.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:45:00 | 2592.56 | 2602.96 | 2569.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 2586.86 | 2597.97 | 2585.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 2583.71 | 2597.97 | 2585.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 2585.68 | 2595.51 | 2585.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 2585.68 | 2595.51 | 2585.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 2585.97 | 2593.60 | 2585.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 2578.69 | 2593.60 | 2585.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 2582.63 | 2591.41 | 2585.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 2582.63 | 2591.41 | 2585.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 2587.74 | 2590.67 | 2585.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:30:00 | 2583.81 | 2590.67 | 2585.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 2587.05 | 2589.95 | 2585.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 2578.20 | 2589.95 | 2585.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 2578.20 | 2587.60 | 2584.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 2575.45 | 2587.60 | 2584.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 2584.10 | 2586.90 | 2584.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:45:00 | 2589.02 | 2587.32 | 2585.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 2602.11 | 2587.85 | 2586.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 2596.89 | 2613.53 | 2615.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 09:15:00 | 2596.89 | 2613.53 | 2615.01 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 2637.32 | 2616.06 | 2615.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 2656.31 | 2631.35 | 2627.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 2627.09 | 2630.50 | 2627.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 2627.09 | 2630.50 | 2627.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2627.09 | 2630.50 | 2627.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 2627.09 | 2630.50 | 2627.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 2633.48 | 2631.10 | 2628.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:45:00 | 2643.22 | 2633.50 | 2629.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 2623.16 | 2632.66 | 2629.86 | SL hit (close<static) qty=1.00 sl=2627.09 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 2611.65 | 2626.67 | 2627.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 2601.12 | 2621.56 | 2625.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 2589.51 | 2588.15 | 2597.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:00:00 | 2589.51 | 2588.15 | 2597.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 2601.22 | 2590.68 | 2595.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 2601.22 | 2590.68 | 2595.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 2597.09 | 2591.96 | 2595.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 2607.71 | 2591.96 | 2595.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2590.10 | 2591.59 | 2595.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 2587.35 | 2591.59 | 2595.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 2587.84 | 2589.23 | 2593.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:45:00 | 2583.02 | 2591.20 | 2592.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2457.98 | 2472.20 | 2486.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2458.45 | 2472.20 | 2486.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2453.87 | 2472.20 | 2486.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 2469.70 | 2468.82 | 2481.12 | SL hit (close>ema200) qty=0.50 sl=2468.82 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 2488.39 | 2478.21 | 2477.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 2492.23 | 2481.02 | 2478.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 2484.85 | 2490.58 | 2486.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 2484.85 | 2490.58 | 2486.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 2484.85 | 2490.58 | 2486.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 2491.54 | 2490.58 | 2486.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 2493.41 | 2491.14 | 2486.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:30:00 | 2495.48 | 2491.64 | 2487.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:30:00 | 2494.29 | 2492.39 | 2488.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:00:00 | 2495.38 | 2492.39 | 2488.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 2482.39 | 2494.58 | 2491.32 | SL hit (close<static) qty=1.00 sl=2483.57 alert=retest2 |

### Cycle 171 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 2477.67 | 2487.66 | 2488.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 2473.54 | 2484.83 | 2487.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 15:15:00 | 2488.69 | 2485.60 | 2487.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 15:15:00 | 2488.69 | 2485.60 | 2487.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 2488.69 | 2485.60 | 2487.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 2466.65 | 2485.60 | 2487.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:00:00 | 2470.59 | 2472.61 | 2479.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:45:00 | 2472.56 | 2468.64 | 2474.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:00:00 | 2472.16 | 2469.35 | 2474.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2472.95 | 2470.07 | 2474.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 2472.85 | 2470.07 | 2474.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 2471.38 | 2470.33 | 2473.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:45:00 | 2472.56 | 2470.33 | 2473.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 2476.69 | 2471.60 | 2474.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 2476.69 | 2471.60 | 2474.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 2481.21 | 2473.52 | 2474.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 2476.79 | 2473.52 | 2474.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 2479.83 | 2474.79 | 2475.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 2481.41 | 2474.79 | 2475.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 2480.82 | 2475.99 | 2475.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 2480.82 | 2475.99 | 2475.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 2484.75 | 2480.09 | 2477.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 2470.00 | 2478.82 | 2477.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 2470.00 | 2478.82 | 2477.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 2470.00 | 2478.82 | 2477.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 2470.00 | 2478.82 | 2477.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 2457.41 | 2474.54 | 2475.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 2452.49 | 2468.06 | 2472.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 11:15:00 | 2462.23 | 2458.94 | 2464.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 2462.23 | 2458.94 | 2464.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 2462.23 | 2458.94 | 2464.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 2462.23 | 2458.94 | 2464.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 2459.67 | 2459.09 | 2464.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:15:00 | 2463.90 | 2459.09 | 2464.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 2459.96 | 2459.26 | 2464.05 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 2483.77 | 2468.23 | 2466.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 12:15:00 | 2503.05 | 2481.94 | 2475.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 2554.50 | 2557.85 | 2539.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 2554.50 | 2557.85 | 2539.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 2572.10 | 2570.06 | 2554.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 2572.79 | 2570.06 | 2554.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 2559.12 | 2567.87 | 2555.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 2559.12 | 2567.87 | 2555.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2555.77 | 2565.45 | 2555.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 2555.77 | 2565.45 | 2555.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 2469.01 | 2545.29 | 2547.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 10:15:00 | 2433.21 | 2446.16 | 2458.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 2419.44 | 2417.93 | 2426.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:15:00 | 2412.16 | 2418.31 | 2426.22 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 2417.37 | 2415.25 | 2421.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:45:00 | 2419.34 | 2415.25 | 2421.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2411.57 | 2410.86 | 2417.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:30:00 | 2404.68 | 2410.97 | 2415.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:00:00 | 2407.14 | 2410.97 | 2415.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:30:00 | 2401.53 | 2408.51 | 2414.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 2382.94 | 2375.98 | 2381.33 | SL hit (close>ema400) qty=1.00 sl=2381.33 alert=retest1 |

### Cycle 176 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 2386.68 | 2383.81 | 2383.55 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 2377.14 | 2383.19 | 2383.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 2372.32 | 2381.02 | 2382.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 2379.40 | 2375.65 | 2378.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 2379.40 | 2375.65 | 2378.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 2379.40 | 2375.65 | 2378.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 2379.40 | 2375.65 | 2378.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 2379.11 | 2376.34 | 2378.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:00:00 | 2379.11 | 2376.34 | 2378.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 2376.65 | 2376.40 | 2378.71 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 2386.39 | 2380.38 | 2380.06 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2369.76 | 2380.59 | 2380.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 2363.66 | 2372.50 | 2376.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 2392.29 | 2375.23 | 2376.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 2392.29 | 2375.23 | 2376.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2392.29 | 2375.23 | 2376.83 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 2399.96 | 2380.18 | 2378.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 2407.63 | 2392.81 | 2385.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 2397.11 | 2401.07 | 2393.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 13:00:00 | 2397.11 | 2401.07 | 2393.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 2391.50 | 2398.85 | 2394.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 2391.50 | 2398.85 | 2394.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 2389.34 | 2396.95 | 2393.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 2384.42 | 2394.19 | 2392.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 2385.30 | 2392.41 | 2391.99 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 2381.57 | 2390.24 | 2391.04 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 2401.73 | 2392.13 | 2391.74 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 2384.42 | 2391.09 | 2391.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 2381.37 | 2388.24 | 2390.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 2382.45 | 2382.13 | 2386.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 2382.45 | 2382.13 | 2386.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2383.73 | 2379.88 | 2383.79 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 2404.78 | 2387.93 | 2385.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 2416.98 | 2393.74 | 2388.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 12:15:00 | 2428.19 | 2430.85 | 2419.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 13:00:00 | 2428.19 | 2430.85 | 2419.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 2421.80 | 2429.57 | 2420.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 2421.80 | 2429.57 | 2420.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 2418.85 | 2427.43 | 2420.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 2432.82 | 2427.43 | 2420.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 2414.81 | 2429.24 | 2425.83 | SL hit (close<static) qty=1.00 sl=2418.85 alert=retest2 |

### Cycle 185 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 2409.80 | 2423.66 | 2424.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 15:15:00 | 2401.63 | 2419.26 | 2422.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 2419.44 | 2417.08 | 2420.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 2419.44 | 2417.08 | 2420.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 2420.32 | 2417.73 | 2420.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 2431.63 | 2417.73 | 2420.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 2425.83 | 2419.35 | 2420.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:30:00 | 2432.52 | 2419.35 | 2420.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 2422.29 | 2419.93 | 2420.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 2433.60 | 2419.93 | 2420.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 2423.86 | 2420.72 | 2421.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 2462.19 | 2408.90 | 2415.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 2309.00 | 2305.67 | 2311.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 2300.90 | 2305.67 | 2311.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 2296.90 | 2289.59 | 2289.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 2296.90 | 2289.59 | 2289.22 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 2286.20 | 2288.91 | 2288.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 2281.40 | 2287.02 | 2288.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 2285.20 | 2284.85 | 2286.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:30:00 | 2281.50 | 2284.85 | 2286.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 2268.30 | 2281.54 | 2284.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 2267.50 | 2277.92 | 2281.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:00:00 | 2265.80 | 2273.12 | 2278.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 2268.10 | 2272.12 | 2277.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 2265.00 | 2269.70 | 2275.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2271.40 | 2269.37 | 2274.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 2275.30 | 2269.37 | 2274.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2270.00 | 2269.50 | 2273.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 2273.20 | 2269.50 | 2273.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2274.90 | 2270.58 | 2273.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 2274.90 | 2270.58 | 2273.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 2280.20 | 2272.50 | 2274.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 2280.20 | 2272.50 | 2274.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 2281.60 | 2274.32 | 2275.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 2281.60 | 2274.32 | 2275.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 2280.90 | 2275.64 | 2275.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2280.90 | 2275.64 | 2275.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2285.90 | 2278.55 | 2276.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 2285.20 | 2285.67 | 2281.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 2283.30 | 2285.67 | 2281.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2298.50 | 2288.23 | 2283.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:30:00 | 2300.20 | 2290.69 | 2284.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:00:00 | 2300.50 | 2290.69 | 2284.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:15:00 | 2300.10 | 2291.14 | 2286.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 2283.00 | 2288.28 | 2288.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 2283.00 | 2288.28 | 2288.40 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 2291.00 | 2287.45 | 2287.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 15:15:00 | 2295.90 | 2289.57 | 2288.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 12:15:00 | 2285.00 | 2290.02 | 2289.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 2285.00 | 2290.02 | 2289.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 2285.00 | 2290.02 | 2289.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:00:00 | 2285.00 | 2290.02 | 2289.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 2285.90 | 2289.20 | 2288.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 2285.90 | 2289.20 | 2288.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2291.00 | 2289.56 | 2288.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 2286.90 | 2289.56 | 2288.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2316.00 | 2294.92 | 2291.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:15:00 | 2320.50 | 2301.79 | 2295.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 13:30:00 | 2318.50 | 2307.10 | 2299.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:45:00 | 2320.20 | 2312.68 | 2306.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 2374.90 | 2391.06 | 2391.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 2374.90 | 2391.06 | 2391.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 2364.10 | 2377.99 | 2384.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 2378.00 | 2377.15 | 2382.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 09:15:00 | 2386.20 | 2377.15 | 2382.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2381.00 | 2377.92 | 2382.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:30:00 | 2388.00 | 2377.92 | 2382.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 2381.90 | 2378.72 | 2382.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 11:00:00 | 2381.90 | 2378.72 | 2382.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 2386.70 | 2380.31 | 2382.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 2386.70 | 2380.31 | 2382.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 2390.90 | 2382.43 | 2383.62 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 2402.20 | 2386.38 | 2385.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 14:15:00 | 2407.90 | 2390.69 | 2387.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 10:15:00 | 2395.70 | 2396.23 | 2391.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:00:00 | 2395.70 | 2396.23 | 2391.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 2382.60 | 2393.50 | 2390.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 2382.60 | 2393.50 | 2390.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 2369.80 | 2388.76 | 2388.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 2369.80 | 2388.76 | 2388.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 13:15:00 | 2372.80 | 2385.57 | 2387.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 2351.40 | 2377.15 | 2382.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 2363.70 | 2360.78 | 2370.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 2363.70 | 2360.78 | 2370.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2363.70 | 2360.78 | 2370.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 2369.60 | 2360.78 | 2370.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2365.00 | 2361.63 | 2369.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 2358.00 | 2362.88 | 2369.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:15:00 | 2355.50 | 2362.71 | 2369.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2382.90 | 2365.82 | 2368.31 | SL hit (close>static) qty=1.00 sl=2372.90 alert=retest2 |

### Cycle 194 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 2392.30 | 2371.11 | 2370.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 11:15:00 | 2397.50 | 2376.39 | 2372.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 13:15:00 | 2413.40 | 2414.09 | 2399.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 14:00:00 | 2413.40 | 2414.09 | 2399.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 2377.10 | 2406.69 | 2397.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 2377.10 | 2406.69 | 2397.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 2382.00 | 2401.76 | 2396.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 2388.20 | 2401.76 | 2396.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 10:45:00 | 2387.70 | 2396.12 | 2394.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 2385.70 | 2392.19 | 2392.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 12:15:00 | 2385.70 | 2392.19 | 2392.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 2364.90 | 2386.73 | 2390.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2381.20 | 2380.01 | 2385.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2381.20 | 2380.01 | 2385.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2381.20 | 2380.01 | 2385.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 2367.80 | 2377.14 | 2383.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2407.50 | 2386.91 | 2385.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 2407.50 | 2386.91 | 2385.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 2424.70 | 2394.47 | 2389.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 2406.00 | 2407.76 | 2399.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:15:00 | 2405.90 | 2407.76 | 2399.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2409.00 | 2408.01 | 2400.17 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 2362.60 | 2394.05 | 2396.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 13:15:00 | 2348.90 | 2373.55 | 2385.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 2378.60 | 2374.56 | 2384.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 2378.60 | 2374.56 | 2384.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 2386.00 | 2376.85 | 2384.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 2344.60 | 2376.85 | 2384.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 2382.40 | 2356.64 | 2353.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 2382.40 | 2356.64 | 2353.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 2385.80 | 2366.27 | 2358.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 2364.80 | 2365.98 | 2359.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 15:00:00 | 2364.80 | 2365.98 | 2359.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2378.70 | 2369.09 | 2362.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 2412.50 | 2369.86 | 2365.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 2353.90 | 2371.94 | 2369.47 | SL hit (close<static) qty=1.00 sl=2360.60 alert=retest2 |

### Cycle 199 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 2355.00 | 2365.68 | 2366.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2339.00 | 2360.34 | 2364.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 2362.40 | 2360.76 | 2364.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:00:00 | 2362.40 | 2360.76 | 2364.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2377.30 | 2364.06 | 2365.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 2377.30 | 2364.06 | 2365.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 2381.50 | 2367.55 | 2366.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 2402.00 | 2374.44 | 2370.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 2376.70 | 2442.33 | 2437.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 2376.70 | 2442.33 | 2437.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2376.70 | 2442.33 | 2437.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 2376.70 | 2442.33 | 2437.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 2414.50 | 2436.76 | 2435.67 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 2418.60 | 2433.13 | 2434.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 2407.00 | 2421.94 | 2428.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 2312.90 | 2312.17 | 2330.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:45:00 | 2311.00 | 2312.17 | 2330.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 2321.00 | 2314.04 | 2324.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 2321.00 | 2314.04 | 2324.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 2325.10 | 2316.25 | 2324.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 2325.10 | 2316.25 | 2324.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2323.00 | 2317.60 | 2324.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:00:00 | 2319.00 | 2321.28 | 2324.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 2329.00 | 2307.58 | 2312.43 | SL hit (close>static) qty=1.00 sl=2328.90 alert=retest2 |

### Cycle 202 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 2334.40 | 2317.02 | 2315.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 2345.70 | 2333.17 | 2325.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 2352.20 | 2356.96 | 2346.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 2352.20 | 2356.96 | 2346.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 2364.40 | 2365.14 | 2355.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 2357.30 | 2365.14 | 2355.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2343.30 | 2367.56 | 2362.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 2343.30 | 2367.56 | 2362.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 2351.30 | 2364.30 | 2361.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:15:00 | 2354.30 | 2360.86 | 2360.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:00:00 | 2360.90 | 2360.87 | 2360.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 2339.50 | 2356.04 | 2357.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 2339.50 | 2356.04 | 2357.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 2315.30 | 2345.33 | 2352.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 2324.50 | 2323.62 | 2337.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:30:00 | 2325.40 | 2323.62 | 2337.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2204.50 | 2199.62 | 2220.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 2195.10 | 2199.88 | 2218.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:45:00 | 2192.30 | 2197.10 | 2215.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 2166.70 | 2163.16 | 2163.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 11:15:00 | 2166.70 | 2163.16 | 2163.15 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 2160.80 | 2162.69 | 2162.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 13:15:00 | 2157.70 | 2161.69 | 2162.46 | Break + close below crossover candle low |

### Cycle 206 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 2177.40 | 2164.83 | 2163.82 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 2151.70 | 2162.94 | 2163.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 12:15:00 | 2145.00 | 2153.97 | 2158.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 11:15:00 | 2097.30 | 2095.39 | 2114.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 11:30:00 | 2098.80 | 2095.39 | 2114.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 2080.30 | 2068.05 | 2085.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:45:00 | 2090.90 | 2068.05 | 2085.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 2052.40 | 2064.92 | 2082.91 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 2134.70 | 2094.75 | 2089.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 2141.30 | 2104.06 | 2094.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2098.80 | 2122.10 | 2110.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2098.80 | 2122.10 | 2110.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2098.80 | 2122.10 | 2110.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 2098.80 | 2122.10 | 2110.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 2096.40 | 2116.96 | 2109.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 2096.40 | 2116.96 | 2109.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 2079.50 | 2102.85 | 2104.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 2071.30 | 2096.54 | 2101.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2084.20 | 2067.77 | 2078.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2084.20 | 2067.77 | 2078.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2084.20 | 2067.77 | 2078.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 2068.90 | 2068.07 | 2077.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 2083.30 | 2062.32 | 2061.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 2083.30 | 2062.32 | 2061.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 2107.00 | 2074.05 | 2068.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 2136.60 | 2137.10 | 2114.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 2134.00 | 2137.10 | 2114.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 2121.10 | 2131.98 | 2119.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 2117.80 | 2131.98 | 2119.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 2133.90 | 2132.36 | 2120.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 2151.60 | 2131.99 | 2121.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2136.30 | 2143.19 | 2136.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 2121.00 | 2131.45 | 2132.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 2121.00 | 2131.45 | 2132.53 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 2162.10 | 2137.58 | 2135.21 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 2140.70 | 2142.68 | 2142.90 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2204.60 | 2154.49 | 2148.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 2226.10 | 2168.81 | 2155.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 2227.20 | 2232.62 | 2212.12 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:15:00 | 2239.40 | 2232.62 | 2212.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 2351.37 | 2286.49 | 2252.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 2353.00 | 2364.49 | 2337.19 | SL hit (close<ema200) qty=0.50 sl=2364.49 alert=retest1 |

### Cycle 215 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 2310.90 | 2328.44 | 2330.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 2291.80 | 2317.99 | 2324.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 2305.70 | 2305.63 | 2315.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 09:30:00 | 2301.70 | 2305.63 | 2315.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 2334.20 | 2311.55 | 2316.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 2334.20 | 2311.55 | 2316.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 2347.90 | 2318.82 | 2319.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 2347.90 | 2318.82 | 2319.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 2328.00 | 2320.65 | 2320.32 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 2313.80 | 2319.28 | 2319.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 2300.80 | 2315.24 | 2317.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 2346.90 | 2283.07 | 2293.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 2346.90 | 2283.07 | 2293.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2346.90 | 2283.07 | 2293.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 2346.90 | 2283.07 | 2293.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 2348.60 | 2296.18 | 2298.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 2354.80 | 2296.18 | 2298.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 2338.80 | 2304.70 | 2301.92 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 2298.40 | 2305.17 | 2306.02 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 2317.50 | 2307.00 | 2306.66 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 2290.30 | 2305.07 | 2305.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 10:15:00 | 2285.80 | 2301.21 | 2304.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 11:15:00 | 2293.00 | 2283.44 | 2290.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 11:15:00 | 2293.00 | 2283.44 | 2290.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 2293.00 | 2283.44 | 2290.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 2295.00 | 2283.44 | 2290.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 2285.00 | 2283.75 | 2290.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 13:30:00 | 2284.40 | 2285.48 | 2290.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-24 12:15:00 | 2577.76 | 2023-05-26 13:15:00 | 2592.81 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-05-26 11:45:00 | 2578.20 | 2023-05-26 13:15:00 | 2592.81 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2023-05-26 12:15:00 | 2577.22 | 2023-05-26 13:15:00 | 2592.81 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-05-31 14:15:00 | 2622.96 | 2023-06-06 11:15:00 | 2631.81 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2023-06-16 09:15:00 | 2656.01 | 2023-06-19 14:15:00 | 2641.16 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2023-06-16 11:00:00 | 2663.29 | 2023-06-19 14:15:00 | 2641.16 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-06-16 13:15:00 | 2653.94 | 2023-06-19 14:15:00 | 2641.16 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-06-21 12:45:00 | 2635.89 | 2023-06-28 11:15:00 | 2629.30 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2023-06-21 14:15:00 | 2634.27 | 2023-06-28 11:15:00 | 2629.30 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2023-06-30 09:15:00 | 2626.84 | 2023-06-30 10:15:00 | 2614.60 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2023-06-30 12:45:00 | 2626.20 | 2023-07-07 15:15:00 | 2654.93 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2023-07-03 10:00:00 | 2630.48 | 2023-07-07 15:15:00 | 2654.93 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2023-07-03 13:00:00 | 2627.58 | 2023-07-07 15:15:00 | 2654.93 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2023-07-04 12:15:00 | 2648.09 | 2023-07-07 15:15:00 | 2654.93 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2023-07-12 09:15:00 | 2627.53 | 2023-07-17 09:15:00 | 2652.96 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-07-13 10:45:00 | 2631.32 | 2023-07-17 09:15:00 | 2652.96 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-07-13 11:15:00 | 2626.45 | 2023-07-17 09:15:00 | 2652.96 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-07-13 12:00:00 | 2626.70 | 2023-07-17 09:15:00 | 2652.96 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-07-14 14:15:00 | 2634.03 | 2023-07-17 09:15:00 | 2652.96 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-07-26 15:00:00 | 2540.63 | 2023-08-07 12:15:00 | 2522.13 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2023-07-28 10:30:00 | 2543.77 | 2023-08-07 12:15:00 | 2522.13 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2023-07-28 15:15:00 | 2537.87 | 2023-08-07 12:15:00 | 2522.13 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2023-08-08 12:45:00 | 2521.20 | 2023-08-09 09:15:00 | 2506.39 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-08-17 12:15:00 | 2513.33 | 2023-08-25 14:15:00 | 2520.76 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2023-08-17 13:30:00 | 2509.79 | 2023-08-25 14:15:00 | 2520.76 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2023-08-18 11:15:00 | 2511.66 | 2023-08-25 14:15:00 | 2520.76 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2023-08-18 12:00:00 | 2513.53 | 2023-08-25 14:15:00 | 2520.76 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2023-08-21 11:15:00 | 2531.63 | 2023-08-25 14:15:00 | 2520.76 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-08-22 11:45:00 | 2531.58 | 2023-08-25 14:15:00 | 2520.76 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-08-23 10:30:00 | 2531.58 | 2023-08-25 14:15:00 | 2520.76 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-08-23 12:45:00 | 2531.13 | 2023-08-25 14:15:00 | 2520.76 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-08-31 10:15:00 | 2479.83 | 2023-09-06 10:15:00 | 2475.90 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2023-09-22 09:15:00 | 2420.37 | 2023-09-22 12:15:00 | 2439.55 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-09-22 11:30:00 | 2430.21 | 2023-09-22 12:15:00 | 2439.55 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-10-06 12:15:00 | 2469.06 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2023-10-09 10:15:00 | 2468.97 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2023-10-09 11:15:00 | 2469.85 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2023-10-09 12:00:00 | 2468.97 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2023-10-10 10:00:00 | 2475.90 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2023-10-10 11:15:00 | 2478.31 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2023-10-10 13:15:00 | 2476.93 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2023-10-11 09:15:00 | 2490.16 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2023-10-17 10:30:00 | 2526.02 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2023-10-17 13:30:00 | 2525.48 | 2023-10-18 10:15:00 | 2501.97 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-10-26 15:15:00 | 2430.11 | 2023-10-27 13:15:00 | 2450.82 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2023-10-27 09:30:00 | 2429.77 | 2023-10-27 13:15:00 | 2450.82 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-10-30 09:15:00 | 2428.78 | 2023-10-31 13:15:00 | 2444.57 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-10-30 14:00:00 | 2432.32 | 2023-10-31 13:15:00 | 2444.57 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-10-31 10:30:00 | 2428.54 | 2023-10-31 13:15:00 | 2444.57 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-11-08 09:15:00 | 2460.36 | 2023-11-09 10:15:00 | 2449.64 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2023-11-08 10:00:00 | 2462.47 | 2023-11-09 10:15:00 | 2449.64 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-11-08 11:15:00 | 2459.23 | 2023-11-09 10:15:00 | 2449.64 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2023-11-13 09:15:00 | 2435.57 | 2023-11-16 09:15:00 | 2454.36 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-11-13 14:00:00 | 2442.65 | 2023-11-16 09:15:00 | 2454.36 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-11-15 09:30:00 | 2441.03 | 2023-11-16 09:15:00 | 2454.36 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-11-15 14:15:00 | 2444.62 | 2023-11-16 09:15:00 | 2454.36 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2023-11-16 09:15:00 | 2438.62 | 2023-11-16 09:15:00 | 2454.36 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-11-16 14:15:00 | 2452.93 | 2023-11-28 15:15:00 | 2461.69 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2023-11-16 15:15:00 | 2453.47 | 2023-11-28 15:15:00 | 2461.69 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest1 | 2023-12-04 09:15:00 | 2552.63 | 2023-12-05 13:15:00 | 2520.41 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-12-06 10:45:00 | 2540.77 | 2023-12-06 14:15:00 | 2524.64 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-12-06 11:15:00 | 2539.84 | 2023-12-06 14:15:00 | 2524.64 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-12-21 10:15:00 | 2520.61 | 2024-01-02 12:15:00 | 2576.23 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2023-12-21 12:30:00 | 2518.15 | 2024-01-02 12:15:00 | 2576.23 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2023-12-22 09:15:00 | 2521.15 | 2024-01-02 12:15:00 | 2576.23 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2023-12-22 09:45:00 | 2525.03 | 2024-01-02 12:15:00 | 2576.23 | STOP_HIT | 1.00 | 2.03% |
| SELL | retest2 | 2024-01-04 14:00:00 | 2564.53 | 2024-01-05 11:15:00 | 2578.01 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-01-05 10:45:00 | 2566.40 | 2024-01-05 11:15:00 | 2578.01 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-01-10 11:00:00 | 2529.46 | 2024-01-15 15:15:00 | 2532.95 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-01-10 13:30:00 | 2527.10 | 2024-01-15 15:15:00 | 2532.95 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-01-10 14:00:00 | 2526.46 | 2024-01-15 15:15:00 | 2532.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-01-11 10:00:00 | 2529.95 | 2024-01-15 15:15:00 | 2532.95 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-01-19 10:15:00 | 2504.38 | 2024-01-19 14:15:00 | 2525.72 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-01-25 13:15:00 | 2395.24 | 2024-01-30 09:15:00 | 2449.39 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-01-29 09:30:00 | 2392.98 | 2024-01-30 09:15:00 | 2449.39 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-01-29 11:00:00 | 2397.45 | 2024-01-30 09:15:00 | 2449.39 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-01-29 14:15:00 | 2396.42 | 2024-01-30 09:15:00 | 2449.39 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-02-08 10:30:00 | 2365.43 | 2024-02-19 10:15:00 | 2350.29 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2024-02-12 10:00:00 | 2371.14 | 2024-02-19 10:15:00 | 2350.29 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2024-02-12 11:00:00 | 2371.34 | 2024-02-19 10:15:00 | 2350.29 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2024-02-12 11:30:00 | 2371.68 | 2024-02-19 10:15:00 | 2350.29 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2024-02-15 09:15:00 | 2338.19 | 2024-02-19 10:15:00 | 2350.29 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-02-29 13:45:00 | 2380.14 | 2024-03-05 09:15:00 | 2360.71 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-03-01 09:45:00 | 2384.86 | 2024-03-05 09:15:00 | 2360.71 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-03-02 09:45:00 | 2378.52 | 2024-03-05 09:15:00 | 2360.71 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-03-04 10:15:00 | 2381.91 | 2024-03-05 09:15:00 | 2360.71 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-03-22 13:30:00 | 2212.82 | 2024-03-22 15:15:00 | 2225.06 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-03-26 09:15:00 | 2210.60 | 2024-03-28 11:15:00 | 2229.79 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-03-26 13:15:00 | 2213.16 | 2024-03-28 11:15:00 | 2229.79 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-03-26 14:00:00 | 2213.21 | 2024-03-28 11:15:00 | 2229.79 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-03-27 11:45:00 | 2211.29 | 2024-03-28 11:15:00 | 2229.79 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-04-03 11:15:00 | 2235.98 | 2024-04-03 14:15:00 | 2228.06 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-04-24 13:15:00 | 2220.74 | 2024-04-25 09:15:00 | 2188.37 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-04-24 14:00:00 | 2221.62 | 2024-04-25 09:15:00 | 2188.37 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-05-09 14:15:00 | 2291.31 | 2024-05-15 10:15:00 | 2297.56 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2024-05-09 14:45:00 | 2290.23 | 2024-05-15 10:15:00 | 2297.56 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2024-05-10 09:15:00 | 2315.76 | 2024-05-15 10:15:00 | 2297.56 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-05-17 14:45:00 | 2283.59 | 2024-05-22 09:15:00 | 2322.94 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-05-21 09:15:00 | 2273.36 | 2024-05-22 09:15:00 | 2322.94 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-05-31 13:45:00 | 2309.17 | 2024-06-04 09:15:00 | 2367.94 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-06-03 12:30:00 | 2307.74 | 2024-06-04 09:15:00 | 2367.94 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-06-10 13:45:00 | 2531.77 | 2024-06-12 09:15:00 | 2497.54 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-06-11 09:15:00 | 2530.99 | 2024-06-12 09:15:00 | 2497.54 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-06-11 13:45:00 | 2530.64 | 2024-06-12 09:15:00 | 2497.54 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-06-19 10:15:00 | 2433.60 | 2024-06-27 09:15:00 | 2421.45 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2024-06-19 12:15:00 | 2434.09 | 2024-06-27 09:15:00 | 2421.45 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2024-06-19 13:15:00 | 2433.90 | 2024-06-27 09:15:00 | 2421.45 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-06-20 10:30:00 | 2430.95 | 2024-06-27 09:15:00 | 2421.45 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-06-21 09:15:00 | 2417.86 | 2024-06-27 09:15:00 | 2421.45 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-06-28 09:15:00 | 2438.77 | 2024-06-28 15:15:00 | 2426.86 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-07-01 10:30:00 | 2476.88 | 2024-07-23 10:15:00 | 2712.51 | TARGET_HIT | 1.00 | 9.51% |
| BUY | retest2 | 2024-07-01 13:30:00 | 2470.78 | 2024-07-23 11:15:00 | 2724.57 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2024-07-02 09:15:00 | 2465.92 | 2024-07-23 11:15:00 | 2717.86 | TARGET_HIT | 1.00 | 10.22% |
| BUY | retest2 | 2024-07-03 15:00:00 | 2474.92 | 2024-07-23 11:15:00 | 2722.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-05 09:15:00 | 2470.54 | 2024-07-23 11:15:00 | 2717.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-08 09:15:00 | 2706.67 | 2024-08-08 09:15:00 | 2690.34 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-08-21 09:15:00 | 2719.16 | 2024-08-27 13:15:00 | 2733.77 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-09-12 14:00:00 | 2880.34 | 2024-09-16 09:15:00 | 2827.46 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-09-18 13:00:00 | 2824.56 | 2024-09-19 09:15:00 | 2853.33 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-09-18 15:15:00 | 2823.14 | 2024-09-19 09:15:00 | 2853.33 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-09-24 13:15:00 | 2932.23 | 2024-09-24 13:15:00 | 2917.17 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-09-30 09:15:00 | 2923.22 | 2024-09-30 11:15:00 | 2907.49 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-10-08 12:30:00 | 2788.71 | 2024-10-14 13:15:00 | 2746.85 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2024-10-08 13:30:00 | 2783.44 | 2024-10-14 13:15:00 | 2746.85 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2024-10-29 09:15:00 | 2518.49 | 2024-11-11 10:15:00 | 2462.33 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2024-11-19 14:45:00 | 2377.24 | 2024-11-22 14:15:00 | 2407.98 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-11-27 12:30:00 | 2436.21 | 2024-12-02 12:15:00 | 2424.75 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-11-29 09:15:00 | 2434.44 | 2024-12-02 12:15:00 | 2424.75 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-12-02 10:15:00 | 2435.13 | 2024-12-02 12:15:00 | 2424.75 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-12-02 10:45:00 | 2432.91 | 2024-12-02 12:15:00 | 2424.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-12-17 10:00:00 | 2321.27 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2024-12-17 12:00:00 | 2323.14 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2024-12-17 13:15:00 | 2324.51 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2024-12-17 14:45:00 | 2322.74 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-12-18 13:30:00 | 2320.18 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2024-12-18 14:00:00 | 2322.30 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2024-12-19 14:45:00 | 2321.81 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2024-12-20 09:15:00 | 2313.59 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-12-24 15:00:00 | 2294.95 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-12-26 13:30:00 | 2296.28 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-12-27 10:00:00 | 2297.07 | 2024-12-27 15:15:00 | 2305.72 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-01-07 09:45:00 | 2347.53 | 2025-01-14 11:15:00 | 2358.50 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-01-07 10:45:00 | 2350.48 | 2025-01-14 11:15:00 | 2358.50 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-01-07 14:00:00 | 2347.19 | 2025-01-14 11:15:00 | 2358.50 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-01-07 14:30:00 | 2350.88 | 2025-01-14 11:15:00 | 2358.50 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-01-09 10:15:00 | 2420.91 | 2025-01-14 11:15:00 | 2358.50 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-01-09 12:15:00 | 2410.73 | 2025-01-14 11:15:00 | 2358.50 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-01-10 13:00:00 | 2410.54 | 2025-01-14 11:15:00 | 2358.50 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-01-10 13:30:00 | 2411.47 | 2025-01-14 11:15:00 | 2358.50 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-01-17 13:15:00 | 2321.22 | 2025-01-24 11:15:00 | 2320.73 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-01-21 12:00:00 | 2321.76 | 2025-01-24 11:15:00 | 2320.73 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-01-21 13:30:00 | 2321.96 | 2025-01-24 11:15:00 | 2320.73 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-01-29 12:00:00 | 2351.91 | 2025-02-03 14:15:00 | 2400.26 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2025-01-29 13:15:00 | 2351.96 | 2025-02-03 14:15:00 | 2400.26 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2025-01-30 09:15:00 | 2353.83 | 2025-02-03 14:15:00 | 2400.26 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2025-01-30 10:00:00 | 2357.71 | 2025-02-03 14:15:00 | 2400.26 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2025-02-10 12:15:00 | 2328.94 | 2025-02-19 12:15:00 | 2212.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 14:00:00 | 2319.59 | 2025-02-20 09:15:00 | 2203.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:45:00 | 2325.89 | 2025-02-20 09:15:00 | 2209.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 12:15:00 | 2328.94 | 2025-02-21 14:15:00 | 2209.23 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2025-02-10 14:00:00 | 2319.59 | 2025-02-21 14:15:00 | 2209.23 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-02-11 09:45:00 | 2325.89 | 2025-02-21 14:15:00 | 2209.23 | STOP_HIT | 0.50 | 5.02% |
| BUY | retest2 | 2025-03-10 11:15:00 | 2218.18 | 2025-03-12 12:15:00 | 2178.49 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-03-11 09:45:00 | 2223.10 | 2025-03-12 12:15:00 | 2178.49 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-04-03 09:15:00 | 2190.98 | 2025-04-03 09:15:00 | 2205.54 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-04-03 11:15:00 | 2197.47 | 2025-04-03 11:15:00 | 2208.59 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-04-03 13:45:00 | 2197.47 | 2025-04-03 14:15:00 | 2207.75 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-04-17 12:00:00 | 2333.37 | 2025-04-21 14:15:00 | 2311.82 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-04-28 12:45:00 | 2282.12 | 2025-04-30 11:15:00 | 2308.58 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-04-28 14:45:00 | 2281.82 | 2025-04-30 11:15:00 | 2308.58 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-04-29 09:30:00 | 2281.63 | 2025-04-30 11:15:00 | 2308.58 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-04-29 12:45:00 | 2283.10 | 2025-04-30 11:15:00 | 2308.58 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-07 11:45:00 | 2339.07 | 2025-05-08 10:15:00 | 2316.74 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-05-22 09:15:00 | 2287.04 | 2025-05-26 09:15:00 | 2342.12 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-05-23 11:45:00 | 2320.09 | 2025-05-26 09:15:00 | 2342.12 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-27 15:15:00 | 2344.09 | 2025-05-28 14:15:00 | 2324.91 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-30 11:45:00 | 2316.64 | 2025-06-02 10:15:00 | 2332.87 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-05-30 12:30:00 | 2316.64 | 2025-06-02 10:15:00 | 2332.87 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-05-30 14:00:00 | 2315.07 | 2025-06-02 10:15:00 | 2332.87 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-05-30 15:00:00 | 2306.12 | 2025-06-02 10:15:00 | 2332.87 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-06-06 09:15:00 | 2342.22 | 2025-06-11 15:15:00 | 2333.46 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-06-06 13:30:00 | 2341.14 | 2025-06-11 15:15:00 | 2333.46 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-06-11 15:00:00 | 2339.46 | 2025-06-11 15:15:00 | 2333.46 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-18 10:30:00 | 2280.64 | 2025-06-27 09:15:00 | 2256.15 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest1 | 2025-07-08 15:15:00 | 2360.81 | 2025-07-10 12:15:00 | 2365.63 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-07-11 09:15:00 | 2467.54 | 2025-07-18 10:15:00 | 2448.46 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-07-23 10:30:00 | 2424.55 | 2025-07-29 13:15:00 | 2409.99 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-08-06 09:15:00 | 2499.02 | 2025-08-07 10:15:00 | 2477.97 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-08-06 11:15:00 | 2502.16 | 2025-08-07 10:15:00 | 2477.97 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-08-06 13:30:00 | 2501.48 | 2025-08-07 10:15:00 | 2477.97 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-08-06 15:15:00 | 2498.52 | 2025-08-07 10:15:00 | 2477.97 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-14 13:30:00 | 2446.29 | 2025-08-18 09:15:00 | 2520.56 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-08-25 11:45:00 | 2589.02 | 2025-09-01 09:15:00 | 2596.89 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-08-26 09:15:00 | 2602.11 | 2025-09-01 09:15:00 | 2596.89 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-09-04 12:45:00 | 2643.22 | 2025-09-04 14:15:00 | 2623.16 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-10 10:15:00 | 2587.35 | 2025-09-29 09:15:00 | 2457.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 11:30:00 | 2587.84 | 2025-09-29 09:15:00 | 2458.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 14:45:00 | 2583.02 | 2025-09-29 09:15:00 | 2453.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 10:15:00 | 2587.35 | 2025-09-29 12:15:00 | 2469.70 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2025-09-10 11:30:00 | 2587.84 | 2025-09-29 12:15:00 | 2469.70 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2025-09-11 14:45:00 | 2583.02 | 2025-09-29 12:15:00 | 2469.70 | STOP_HIT | 0.50 | 4.39% |
| BUY | retest2 | 2025-10-06 11:30:00 | 2495.48 | 2025-10-07 10:15:00 | 2482.39 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-10-06 12:30:00 | 2494.29 | 2025-10-07 10:15:00 | 2482.39 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-10-06 13:00:00 | 2495.38 | 2025-10-07 10:15:00 | 2482.39 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-10-08 09:15:00 | 2466.65 | 2025-10-10 10:15:00 | 2480.82 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-08 14:00:00 | 2470.59 | 2025-10-10 10:15:00 | 2480.82 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-09 10:45:00 | 2472.56 | 2025-10-10 10:15:00 | 2480.82 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-10-09 12:00:00 | 2472.16 | 2025-10-10 10:15:00 | 2480.82 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-04 09:15:00 | 2412.16 | 2025-11-11 13:15:00 | 2382.94 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2025-11-06 12:30:00 | 2404.68 | 2025-11-12 14:15:00 | 2386.68 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-11-06 13:00:00 | 2407.14 | 2025-11-12 14:15:00 | 2386.68 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2025-11-06 13:30:00 | 2401.53 | 2025-11-12 14:15:00 | 2386.68 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-12-02 09:15:00 | 2432.82 | 2025-12-03 09:15:00 | 2414.81 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-03 10:15:00 | 2427.80 | 2025-12-03 14:15:00 | 2409.80 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-03 10:45:00 | 2426.72 | 2025-12-03 14:15:00 | 2409.80 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-03 14:00:00 | 2425.73 | 2025-12-03 14:15:00 | 2409.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-12-12 09:15:00 | 2300.90 | 2025-12-16 11:15:00 | 2296.90 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-12-18 09:15:00 | 2267.50 | 2025-12-19 14:15:00 | 2280.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-12-18 11:00:00 | 2265.80 | 2025-12-19 14:15:00 | 2280.90 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-18 12:00:00 | 2268.10 | 2025-12-19 14:15:00 | 2280.90 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-18 13:30:00 | 2265.00 | 2025-12-19 14:15:00 | 2280.90 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-23 10:30:00 | 2300.20 | 2025-12-24 15:15:00 | 2283.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-12-23 11:00:00 | 2300.50 | 2025-12-24 15:15:00 | 2283.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-12-23 14:15:00 | 2300.10 | 2025-12-24 15:15:00 | 2283.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-31 12:15:00 | 2320.50 | 2026-01-09 09:15:00 | 2374.90 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-12-31 13:30:00 | 2318.50 | 2026-01-09 09:15:00 | 2374.90 | STOP_HIT | 1.00 | 2.43% |
| BUY | retest2 | 2026-01-01 14:45:00 | 2320.20 | 2026-01-09 09:15:00 | 2374.90 | STOP_HIT | 1.00 | 2.36% |
| SELL | retest2 | 2026-01-16 12:15:00 | 2358.00 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-16 13:15:00 | 2355.50 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-01-21 09:15:00 | 2388.20 | 2026-01-21 12:15:00 | 2385.70 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-01-21 10:45:00 | 2387.70 | 2026-01-21 12:15:00 | 2385.70 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2026-01-22 12:45:00 | 2367.80 | 2026-01-23 09:15:00 | 2407.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-29 09:15:00 | 2344.60 | 2026-02-03 11:15:00 | 2382.40 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-05 09:15:00 | 2412.50 | 2026-02-05 13:15:00 | 2353.90 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-02-19 11:00:00 | 2319.00 | 2026-02-20 11:15:00 | 2329.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-02-20 11:45:00 | 2321.10 | 2026-02-23 09:15:00 | 2334.40 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-02-20 13:30:00 | 2313.00 | 2026-02-23 09:15:00 | 2334.40 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-02-27 12:15:00 | 2354.30 | 2026-02-27 14:15:00 | 2339.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-02-27 13:00:00 | 2360.90 | 2026-02-27 14:15:00 | 2339.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-03-10 11:15:00 | 2195.10 | 2026-03-16 11:15:00 | 2166.70 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2026-03-10 11:45:00 | 2192.30 | 2026-03-16 11:15:00 | 2166.70 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2026-04-01 10:45:00 | 2068.90 | 2026-04-06 14:15:00 | 2083.30 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2026-04-10 09:15:00 | 2151.60 | 2026-04-13 15:15:00 | 2121.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2136.30 | 2026-04-13 15:15:00 | 2121.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2026-04-21 09:15:00 | 2239.40 | 2026-04-22 09:15:00 | 2351.37 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-21 09:15:00 | 2239.40 | 2026-04-23 15:15:00 | 2353.00 | STOP_HIT | 0.50 | 5.07% |
| BUY | retest2 | 2026-04-27 09:15:00 | 2343.70 | 2026-04-28 09:15:00 | 2310.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-27 12:15:00 | 2335.50 | 2026-04-28 09:15:00 | 2310.90 | STOP_HIT | 1.00 | -1.05% |
