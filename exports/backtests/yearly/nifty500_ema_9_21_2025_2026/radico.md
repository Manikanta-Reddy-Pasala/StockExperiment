# Radico Khaitan Ltd (RADICO)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 3481.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 55 |
| ALERT2 | 53 |
| ALERT2_SKIP | 31 |
| ALERT3 | 152 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 61 |
| PARTIAL | 12 |
| TARGET_HIT | 1 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 41
- **Target hits / Stop hits / Partials:** 1 / 61 / 12
- **Avg / median % per leg:** 1.17% / -0.38%
- **Sum % (uncompounded):** 86.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 3 | 16.7% | 1 | 17 | 0 | -0.53% | -9.5% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.34% | 1.3% |
| BUY @ 3rd Alert (retest2) | 17 | 2 | 11.8% | 1 | 16 | 0 | -0.64% | -10.8% |
| SELL (all) | 56 | 30 | 53.6% | 0 | 44 | 12 | 1.72% | 96.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 56 | 30 | 53.6% | 0 | 44 | 12 | 1.72% | 96.2% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.34% | 1.3% |
| retest2 (combined) | 73 | 32 | 43.8% | 1 | 60 | 12 | 1.17% | 85.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2506.00 | 2465.56 | 2462.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 2526.00 | 2491.11 | 2476.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 2558.70 | 2574.96 | 2547.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:00:00 | 2558.70 | 2574.96 | 2547.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 2572.80 | 2572.15 | 2550.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 2563.30 | 2572.15 | 2550.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 2571.80 | 2579.11 | 2572.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:30:00 | 2572.60 | 2579.11 | 2572.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 2572.10 | 2577.71 | 2572.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 2564.00 | 2577.71 | 2572.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 2580.00 | 2583.66 | 2577.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:45:00 | 2576.90 | 2583.66 | 2577.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 2590.10 | 2584.95 | 2579.07 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 2562.10 | 2576.08 | 2576.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 2484.50 | 2553.42 | 2566.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 11:15:00 | 2474.60 | 2473.71 | 2504.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:00:00 | 2474.60 | 2473.71 | 2504.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2484.50 | 2476.71 | 2494.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 2486.20 | 2476.71 | 2494.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2485.70 | 2478.50 | 2493.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 2485.70 | 2478.50 | 2493.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2479.70 | 2470.92 | 2482.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 2451.90 | 2472.21 | 2480.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:45:00 | 2448.50 | 2459.91 | 2467.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 2476.80 | 2463.91 | 2463.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 2476.80 | 2463.91 | 2463.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 2498.80 | 2473.03 | 2467.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 2706.00 | 2706.79 | 2673.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 14:30:00 | 2706.40 | 2706.79 | 2673.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 2669.40 | 2698.23 | 2674.98 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 13:15:00 | 2637.70 | 2661.30 | 2662.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 2628.10 | 2654.66 | 2659.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 14:15:00 | 2650.00 | 2644.98 | 2650.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 14:15:00 | 2650.00 | 2644.98 | 2650.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 2650.00 | 2644.98 | 2650.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:45:00 | 2650.00 | 2644.98 | 2650.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 2647.00 | 2645.38 | 2650.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 2656.10 | 2645.38 | 2650.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2658.30 | 2647.97 | 2650.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:45:00 | 2645.50 | 2648.97 | 2650.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 13:45:00 | 2646.60 | 2648.14 | 2650.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 2630.30 | 2649.16 | 2650.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 2625.10 | 2604.49 | 2603.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 2625.10 | 2604.49 | 2603.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 2633.20 | 2616.79 | 2610.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 2608.00 | 2615.24 | 2610.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 2608.00 | 2615.24 | 2610.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 2608.00 | 2615.24 | 2610.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 2608.00 | 2615.24 | 2610.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 2599.90 | 2612.17 | 2609.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 2599.40 | 2612.17 | 2609.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 2600.00 | 2609.74 | 2608.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 2594.20 | 2609.74 | 2608.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2610.00 | 2609.54 | 2608.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 2631.00 | 2609.54 | 2608.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 2638.60 | 2642.87 | 2643.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 2638.60 | 2642.87 | 2643.04 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 2646.60 | 2643.25 | 2643.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 2662.00 | 2649.10 | 2646.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 2689.40 | 2690.96 | 2676.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:00:00 | 2689.40 | 2690.96 | 2676.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 2677.40 | 2687.79 | 2677.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 2681.20 | 2687.79 | 2677.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 2677.00 | 2685.63 | 2677.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 2672.20 | 2685.63 | 2677.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 2680.30 | 2684.56 | 2677.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 2677.40 | 2684.56 | 2677.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 2678.60 | 2683.37 | 2677.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:30:00 | 2672.50 | 2683.37 | 2677.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 2680.20 | 2682.74 | 2677.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:45:00 | 2677.80 | 2682.74 | 2677.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 2673.30 | 2680.85 | 2677.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 2673.30 | 2680.85 | 2677.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 2677.20 | 2680.12 | 2677.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 2675.20 | 2680.12 | 2677.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 09:15:00 | 2653.00 | 2674.70 | 2675.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 11:15:00 | 2649.10 | 2665.95 | 2670.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 2617.00 | 2612.72 | 2632.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 15:00:00 | 2617.00 | 2612.72 | 2632.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 2613.10 | 2614.62 | 2629.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:15:00 | 2601.10 | 2614.62 | 2629.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 2634.80 | 2580.29 | 2574.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 2634.80 | 2580.29 | 2574.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 2679.00 | 2659.64 | 2641.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 15:15:00 | 2692.00 | 2696.98 | 2675.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:15:00 | 2702.10 | 2696.98 | 2675.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2743.80 | 2745.03 | 2735.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 2737.70 | 2745.03 | 2735.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 2740.00 | 2743.66 | 2738.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 2739.80 | 2743.66 | 2738.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2742.90 | 2743.51 | 2738.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 2738.40 | 2741.86 | 2738.80 | SL hit (close<ema400) qty=1.00 sl=2738.80 alert=retest1 |

### Cycle 10 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 2731.40 | 2736.28 | 2736.85 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 2751.20 | 2739.27 | 2738.16 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 2718.60 | 2738.18 | 2738.92 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 2751.10 | 2732.66 | 2732.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 2758.70 | 2737.87 | 2734.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 2747.00 | 2750.40 | 2743.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 2747.00 | 2750.40 | 2743.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 2747.00 | 2750.40 | 2743.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 2747.00 | 2750.40 | 2743.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 2746.90 | 2749.70 | 2743.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:45:00 | 2745.70 | 2749.70 | 2743.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 2748.60 | 2749.48 | 2744.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:30:00 | 2744.80 | 2749.48 | 2744.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 2763.80 | 2752.34 | 2746.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 2768.90 | 2752.97 | 2746.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 2737.70 | 2749.92 | 2746.13 | SL hit (close<static) qty=1.00 sl=2743.10 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 2730.60 | 2742.87 | 2743.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 2722.00 | 2738.69 | 2741.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2720.00 | 2711.81 | 2721.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2720.00 | 2711.81 | 2721.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2720.00 | 2711.81 | 2721.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 2701.00 | 2707.07 | 2717.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:45:00 | 2696.60 | 2703.67 | 2714.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 2700.10 | 2688.81 | 2703.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:45:00 | 2700.00 | 2693.55 | 2701.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2685.00 | 2691.84 | 2700.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 2682.40 | 2689.95 | 2698.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 2711.30 | 2699.22 | 2699.60 | SL hit (close>static) qty=1.00 sl=2702.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 2707.00 | 2700.78 | 2700.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 2722.50 | 2705.12 | 2702.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 2826.00 | 2828.49 | 2789.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:30:00 | 2817.70 | 2828.49 | 2789.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 2787.20 | 2820.23 | 2788.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 2787.20 | 2820.23 | 2788.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 2801.90 | 2816.56 | 2790.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 2836.50 | 2806.03 | 2792.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 2844.70 | 2849.59 | 2842.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 2819.00 | 2836.44 | 2838.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 2819.00 | 2836.44 | 2838.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 2810.40 | 2829.20 | 2834.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2843.70 | 2832.10 | 2834.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2843.70 | 2832.10 | 2834.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2843.70 | 2832.10 | 2834.93 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 2859.00 | 2837.48 | 2837.11 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 13:15:00 | 2833.10 | 2836.74 | 2836.90 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2850.60 | 2839.51 | 2838.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 2874.90 | 2849.10 | 2842.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 2868.70 | 2880.21 | 2866.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 2868.70 | 2880.21 | 2866.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 2868.70 | 2880.21 | 2866.25 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 2815.50 | 2854.16 | 2856.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 13:15:00 | 2795.70 | 2842.47 | 2851.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 13:15:00 | 2862.90 | 2830.03 | 2838.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 13:15:00 | 2862.90 | 2830.03 | 2838.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 2862.90 | 2830.03 | 2838.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 2862.90 | 2830.03 | 2838.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 2861.90 | 2836.41 | 2840.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 2869.50 | 2836.41 | 2840.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 2879.50 | 2847.36 | 2844.71 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 2825.80 | 2847.65 | 2849.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 14:15:00 | 2821.00 | 2839.35 | 2844.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 12:15:00 | 2832.70 | 2821.57 | 2832.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 12:15:00 | 2832.70 | 2821.57 | 2832.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 2832.70 | 2821.57 | 2832.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 2832.70 | 2821.57 | 2832.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 2830.00 | 2823.25 | 2832.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 2830.00 | 2823.25 | 2832.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 2829.90 | 2824.58 | 2831.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 2837.80 | 2824.58 | 2831.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 2825.90 | 2824.85 | 2831.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 2872.00 | 2824.85 | 2831.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 2847.40 | 2829.36 | 2832.80 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 2861.30 | 2839.00 | 2836.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 14:15:00 | 2890.20 | 2857.45 | 2846.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 2858.30 | 2893.61 | 2884.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 2858.30 | 2893.61 | 2884.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 2858.30 | 2893.61 | 2884.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 2862.90 | 2893.61 | 2884.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 2899.50 | 2894.79 | 2885.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:00:00 | 2911.90 | 2898.21 | 2887.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 14:45:00 | 2913.20 | 2901.06 | 2891.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 11:15:00 | 2865.90 | 2883.76 | 2885.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 2865.90 | 2883.76 | 2885.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 2853.90 | 2877.79 | 2883.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 2872.70 | 2860.43 | 2869.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 11:15:00 | 2872.70 | 2860.43 | 2869.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 2872.70 | 2860.43 | 2869.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 2873.10 | 2860.43 | 2869.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 2833.30 | 2855.01 | 2866.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:30:00 | 2868.90 | 2855.01 | 2866.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 2849.10 | 2848.35 | 2861.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:30:00 | 2858.70 | 2848.35 | 2861.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2855.50 | 2850.40 | 2859.83 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 2897.60 | 2866.06 | 2862.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 2940.60 | 2893.00 | 2879.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 2901.20 | 2905.53 | 2891.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 2901.20 | 2905.53 | 2891.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 2899.20 | 2904.26 | 2891.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 2872.70 | 2904.26 | 2891.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 2890.20 | 2901.45 | 2891.69 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 2840.90 | 2879.57 | 2883.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 2822.50 | 2868.15 | 2877.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 2756.70 | 2752.33 | 2779.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 15:00:00 | 2756.70 | 2752.33 | 2779.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2721.50 | 2747.39 | 2772.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 2712.30 | 2747.39 | 2772.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 2791.40 | 2755.52 | 2767.74 | SL hit (close>static) qty=1.00 sl=2790.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 2830.00 | 2785.70 | 2779.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 11:15:00 | 2852.40 | 2806.96 | 2791.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 2870.50 | 2871.67 | 2844.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:45:00 | 2874.90 | 2871.67 | 2844.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 3011.00 | 3002.32 | 2980.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 2998.90 | 3002.32 | 2980.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 3034.20 | 3037.04 | 3022.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:15:00 | 3020.00 | 3037.04 | 3022.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 3020.40 | 3033.71 | 3022.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 3018.10 | 3033.71 | 3022.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 3030.00 | 3032.97 | 3023.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:30:00 | 3023.80 | 3032.97 | 3023.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 3024.90 | 3031.36 | 3023.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 3011.10 | 3031.36 | 3023.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 2992.20 | 3023.53 | 3020.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 2992.90 | 3023.53 | 3020.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 2992.40 | 3017.30 | 3018.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 2986.40 | 3011.12 | 3015.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 2968.00 | 2965.38 | 2983.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 13:15:00 | 2968.00 | 2965.38 | 2983.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 2968.00 | 2965.38 | 2983.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 2985.00 | 2965.38 | 2983.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2999.80 | 2973.12 | 2982.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:45:00 | 2969.20 | 2975.32 | 2981.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:00:00 | 2956.00 | 2970.65 | 2978.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 2918.60 | 2903.21 | 2902.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 2918.60 | 2903.21 | 2902.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 2942.80 | 2911.13 | 2906.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 2917.70 | 2925.25 | 2916.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 13:15:00 | 2917.70 | 2925.25 | 2916.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 2917.70 | 2925.25 | 2916.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:45:00 | 2915.30 | 2925.25 | 2916.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 2961.20 | 2932.44 | 2920.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:30:00 | 2968.20 | 2945.79 | 2931.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:15:00 | 2965.90 | 2945.79 | 2931.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 2973.00 | 2953.12 | 2939.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:45:00 | 2966.90 | 2973.61 | 2970.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2972.60 | 2973.41 | 2970.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 2966.70 | 2973.41 | 2970.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 2971.60 | 2973.05 | 2971.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:15:00 | 2966.60 | 2973.05 | 2971.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 2988.50 | 2976.14 | 2972.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:30:00 | 2974.60 | 2976.14 | 2972.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3002.60 | 2981.09 | 2975.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:30:00 | 3026.40 | 2988.38 | 2982.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 2945.80 | 2984.70 | 2985.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 2945.80 | 2984.70 | 2985.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2938.70 | 2975.50 | 2980.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 2944.10 | 2939.42 | 2957.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 2944.10 | 2939.42 | 2957.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2950.00 | 2939.10 | 2949.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 2960.00 | 2939.10 | 2949.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 2946.00 | 2940.48 | 2949.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 2988.00 | 2940.48 | 2949.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 3001.10 | 2952.60 | 2954.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 3000.30 | 2952.60 | 2954.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 2994.10 | 2960.90 | 2957.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 3011.00 | 2970.92 | 2962.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 3272.90 | 3280.74 | 3211.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:00:00 | 3272.90 | 3280.74 | 3211.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 3239.70 | 3264.11 | 3240.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 3234.20 | 3264.11 | 3240.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3218.40 | 3254.97 | 3238.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 3218.40 | 3254.97 | 3238.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 3219.90 | 3247.96 | 3236.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 3203.10 | 3247.96 | 3236.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 3222.50 | 3236.74 | 3233.15 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 3220.40 | 3230.64 | 3230.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 3209.70 | 3221.27 | 3225.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 3210.00 | 3206.35 | 3215.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 3210.00 | 3206.35 | 3215.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 3210.00 | 3206.35 | 3215.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:15:00 | 3182.50 | 3203.56 | 3213.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 3166.20 | 3194.72 | 3205.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:00:00 | 3178.00 | 3152.43 | 3160.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:00:00 | 3179.20 | 3157.79 | 3162.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 3192.00 | 3169.16 | 3167.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 3192.00 | 3169.16 | 3167.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 3197.40 | 3181.72 | 3174.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 3177.10 | 3211.56 | 3201.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 14:15:00 | 3177.10 | 3211.56 | 3201.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 3177.10 | 3211.56 | 3201.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 3177.10 | 3211.56 | 3201.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 3178.00 | 3204.84 | 3199.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 3175.10 | 3204.84 | 3199.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 3276.10 | 3274.06 | 3260.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 3276.10 | 3274.06 | 3260.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 3314.00 | 3296.62 | 3280.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:30:00 | 3304.70 | 3296.62 | 3280.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 3269.20 | 3292.62 | 3282.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 3269.20 | 3292.62 | 3282.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 3281.00 | 3290.30 | 3282.61 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 3219.90 | 3270.57 | 3274.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 3201.00 | 3256.66 | 3267.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 3241.30 | 3228.42 | 3246.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 12:00:00 | 3241.30 | 3228.42 | 3246.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 3247.90 | 3232.32 | 3246.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 3248.20 | 3232.32 | 3246.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 3248.60 | 3235.57 | 3246.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 3257.60 | 3235.57 | 3246.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 3263.40 | 3241.14 | 3248.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 3263.40 | 3241.14 | 3248.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 3260.00 | 3244.91 | 3249.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 3287.00 | 3244.91 | 3249.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 3291.40 | 3254.21 | 3253.08 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 3246.70 | 3272.70 | 3274.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 3229.60 | 3264.08 | 3270.12 | Break + close below crossover candle low |

### Cycle 37 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 3413.40 | 3284.13 | 3276.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 3570.70 | 3341.44 | 3303.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 13:15:00 | 3356.60 | 3360.77 | 3323.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:00:00 | 3356.60 | 3360.77 | 3323.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 3319.80 | 3363.37 | 3340.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 3319.80 | 3363.37 | 3340.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 3306.20 | 3351.94 | 3337.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:45:00 | 3300.00 | 3351.94 | 3337.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 3279.30 | 3327.48 | 3328.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 09:15:00 | 3237.30 | 3281.53 | 3289.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 10:15:00 | 3200.40 | 3199.43 | 3232.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 10:15:00 | 3200.40 | 3199.43 | 3232.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 3200.40 | 3199.43 | 3232.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 3200.40 | 3199.43 | 3232.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 3238.50 | 3210.07 | 3231.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:00:00 | 3238.50 | 3210.07 | 3231.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 3227.00 | 3213.46 | 3231.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:30:00 | 3238.80 | 3213.46 | 3231.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 3213.00 | 3213.36 | 3229.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 3207.00 | 3213.36 | 3229.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:30:00 | 3207.80 | 3213.18 | 3223.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 3208.00 | 3216.41 | 3223.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:45:00 | 3209.40 | 3218.42 | 3222.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 3219.00 | 3218.54 | 3222.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 3247.00 | 3218.54 | 3222.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 3208.00 | 3216.43 | 3221.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 3235.50 | 3216.31 | 3218.57 | SL hit (close>static) qty=1.00 sl=3232.20 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 10:15:00 | 3249.90 | 3223.03 | 3221.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 13:15:00 | 3257.90 | 3238.70 | 3229.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 3225.00 | 3242.61 | 3236.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 11:15:00 | 3225.00 | 3242.61 | 3236.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 3225.00 | 3242.61 | 3236.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 3225.00 | 3242.61 | 3236.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 3239.60 | 3242.00 | 3236.47 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 3210.00 | 3230.68 | 3232.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 3194.80 | 3223.50 | 3228.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 3207.70 | 3206.78 | 3216.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 15:15:00 | 3207.70 | 3206.78 | 3216.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 3207.70 | 3206.78 | 3216.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 3227.20 | 3206.78 | 3216.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 3236.80 | 3212.78 | 3218.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 3243.20 | 3212.78 | 3218.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 3205.70 | 3211.37 | 3217.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:30:00 | 3199.50 | 3208.91 | 3215.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:45:00 | 3200.00 | 3181.64 | 3187.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 3240.00 | 3197.69 | 3194.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 3240.00 | 3197.69 | 3194.42 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 13:15:00 | 3199.60 | 3205.62 | 3205.86 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 3223.00 | 3207.94 | 3206.72 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 3184.30 | 3217.87 | 3217.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 3151.00 | 3204.50 | 3211.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 3177.00 | 3174.28 | 3190.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 09:15:00 | 3175.30 | 3174.28 | 3190.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3181.90 | 3175.81 | 3190.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 3195.00 | 3175.81 | 3190.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 3144.20 | 3117.53 | 3142.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 3144.20 | 3117.53 | 3142.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 3178.20 | 3129.66 | 3145.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 3178.20 | 3129.66 | 3145.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 3169.50 | 3137.63 | 3147.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 3154.00 | 3137.63 | 3147.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 3150.50 | 3146.30 | 3150.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 3180.90 | 3157.25 | 3154.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 3180.90 | 3157.25 | 3154.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 3202.90 | 3169.78 | 3160.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 3256.30 | 3260.34 | 3228.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 3256.30 | 3260.34 | 3228.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 3310.10 | 3272.87 | 3248.88 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 3228.80 | 3266.33 | 3270.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 3167.80 | 3231.19 | 3249.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 3278.90 | 3227.61 | 3242.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 3278.90 | 3227.61 | 3242.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 3278.90 | 3227.61 | 3242.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 3278.90 | 3227.61 | 3242.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 3378.80 | 3257.85 | 3254.78 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 3262.70 | 3273.10 | 3273.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 14:15:00 | 3247.00 | 3267.88 | 3271.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 3103.80 | 3092.10 | 3142.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 15:00:00 | 3103.80 | 3092.10 | 3142.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 3151.00 | 3111.89 | 3133.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 3151.00 | 3111.89 | 3133.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 3150.00 | 3119.51 | 3134.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:30:00 | 3162.00 | 3119.51 | 3134.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 3140.00 | 3124.87 | 3134.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 3146.10 | 3124.87 | 3134.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 3154.60 | 3130.82 | 3136.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 3164.70 | 3130.82 | 3136.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 3170.60 | 3138.77 | 3139.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 3167.00 | 3138.77 | 3139.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 3206.80 | 3152.38 | 3145.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 12:15:00 | 3231.10 | 3168.12 | 3153.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 3166.00 | 3184.45 | 3169.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 3166.00 | 3184.45 | 3169.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 3166.00 | 3184.45 | 3169.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 3166.00 | 3184.45 | 3169.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 3162.60 | 3180.08 | 3168.86 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 3096.40 | 3154.69 | 3158.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 3087.20 | 3141.19 | 3152.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 2899.90 | 2869.28 | 2942.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 10:15:00 | 2932.00 | 2881.83 | 2941.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 2932.00 | 2881.83 | 2941.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 2941.60 | 2881.83 | 2941.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 2951.80 | 2895.82 | 2942.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 2951.80 | 2895.82 | 2942.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 2940.00 | 2904.66 | 2942.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:15:00 | 2927.60 | 2904.66 | 2942.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:30:00 | 2937.10 | 2914.50 | 2940.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:00:00 | 2929.10 | 2914.50 | 2940.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 2909.40 | 2917.62 | 2937.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 2933.90 | 2920.14 | 2935.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 2933.90 | 2920.14 | 2935.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 2931.90 | 2922.49 | 2934.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 2936.50 | 2922.49 | 2934.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 2925.40 | 2923.07 | 2934.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 2903.30 | 2919.12 | 2931.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 2896.00 | 2914.19 | 2926.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 2790.24 | 2853.64 | 2880.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2781.22 | 2841.89 | 2872.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2782.64 | 2841.89 | 2872.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 2763.93 | 2802.44 | 2842.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 2758.14 | 2802.44 | 2842.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 2751.20 | 2791.95 | 2834.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 2792.10 | 2784.38 | 2819.69 | SL hit (close>ema200) qty=0.50 sl=2784.38 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 2874.00 | 2835.73 | 2830.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 2942.10 | 2862.71 | 2844.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 2958.40 | 2960.40 | 2917.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 2958.40 | 2960.40 | 2917.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2945.20 | 2957.61 | 2923.63 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 10:15:00 | 2820.20 | 2903.35 | 2912.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 11:15:00 | 2790.00 | 2880.68 | 2900.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 2759.60 | 2731.40 | 2780.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 2759.60 | 2731.40 | 2780.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 2806.70 | 2746.46 | 2782.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 2808.90 | 2746.46 | 2782.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 2807.50 | 2758.67 | 2784.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 2807.50 | 2758.67 | 2784.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 2795.00 | 2765.93 | 2785.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:30:00 | 2807.80 | 2765.93 | 2785.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 2748.70 | 2753.63 | 2769.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:30:00 | 2766.20 | 2753.63 | 2769.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 2757.00 | 2736.04 | 2749.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:45:00 | 2754.00 | 2736.04 | 2749.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2774.90 | 2743.81 | 2751.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2799.90 | 2743.81 | 2751.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2787.10 | 2752.47 | 2755.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:30:00 | 2776.20 | 2753.16 | 2755.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 2769.20 | 2757.79 | 2756.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 2769.20 | 2757.79 | 2756.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 2840.90 | 2776.36 | 2765.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 13:15:00 | 2773.20 | 2790.26 | 2777.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 13:15:00 | 2773.20 | 2790.26 | 2777.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 2773.20 | 2790.26 | 2777.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 2771.60 | 2790.26 | 2777.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 2755.00 | 2783.20 | 2775.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:45:00 | 2752.80 | 2783.20 | 2775.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 2754.60 | 2768.50 | 2769.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 11:15:00 | 2716.10 | 2758.02 | 2765.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 2773.30 | 2752.89 | 2760.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 2773.30 | 2752.89 | 2760.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 2773.30 | 2752.89 | 2760.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 2773.30 | 2752.89 | 2760.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 2769.00 | 2756.11 | 2761.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 2738.30 | 2756.11 | 2761.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 2782.20 | 2742.58 | 2744.81 | SL hit (close>static) qty=1.00 sl=2773.20 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 2811.40 | 2756.34 | 2750.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2822.90 | 2769.65 | 2757.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 2784.00 | 2795.39 | 2775.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:00:00 | 2784.00 | 2795.39 | 2775.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 2770.30 | 2788.64 | 2778.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 2770.30 | 2788.64 | 2778.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 2760.00 | 2782.91 | 2777.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 2760.00 | 2782.91 | 2777.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2794.30 | 2801.63 | 2791.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 2814.20 | 2803.03 | 2793.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:00:00 | 2815.90 | 2805.60 | 2795.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:30:00 | 2814.00 | 2812.48 | 2804.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 2787.40 | 2799.86 | 2799.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 2787.40 | 2799.86 | 2799.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 2782.80 | 2796.45 | 2798.31 | Break + close below crossover candle low |

### Cycle 57 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 2831.10 | 2799.80 | 2799.07 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 2801.00 | 2806.32 | 2806.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 14:15:00 | 2786.80 | 2800.77 | 2803.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 2734.90 | 2720.49 | 2741.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 2734.90 | 2720.49 | 2741.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2734.90 | 2720.49 | 2741.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:00:00 | 2707.30 | 2717.85 | 2738.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 2707.90 | 2714.79 | 2731.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 2704.40 | 2715.96 | 2728.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:45:00 | 2706.60 | 2717.26 | 2720.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 2720.00 | 2717.81 | 2720.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 2684.50 | 2717.81 | 2720.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2571.93 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2572.51 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2569.18 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2571.27 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2550.28 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 10:15:00 | 2549.10 | 2548.92 | 2581.24 | SL hit (close>ema200) qty=0.50 sl=2548.92 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 2705.50 | 2600.23 | 2586.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 14:15:00 | 2772.70 | 2634.73 | 2603.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 2818.50 | 2828.87 | 2782.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 2818.50 | 2828.87 | 2782.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2818.50 | 2828.87 | 2782.56 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 2756.10 | 2802.35 | 2808.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 12:15:00 | 2746.00 | 2791.08 | 2802.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 2766.60 | 2766.29 | 2784.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 2766.60 | 2766.29 | 2784.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 2766.60 | 2766.29 | 2784.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:30:00 | 2731.10 | 2756.14 | 2775.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:15:00 | 2594.54 | 2655.17 | 2706.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 10:15:00 | 2646.70 | 2635.46 | 2678.55 | SL hit (close>ema200) qty=0.50 sl=2635.46 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 2671.60 | 2636.10 | 2634.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 2711.90 | 2655.72 | 2644.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2687.90 | 2703.82 | 2676.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2687.90 | 2703.82 | 2676.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2687.90 | 2703.82 | 2676.32 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 2638.40 | 2664.64 | 2665.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2599.00 | 2651.52 | 2659.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 2641.60 | 2621.88 | 2638.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 2641.60 | 2621.88 | 2638.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 2641.60 | 2621.88 | 2638.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 2641.60 | 2621.88 | 2638.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 2604.00 | 2618.30 | 2635.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 2656.10 | 2618.30 | 2635.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2629.10 | 2620.46 | 2635.14 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 2676.10 | 2649.19 | 2645.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 2690.00 | 2662.07 | 2652.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 2608.40 | 2651.34 | 2648.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 2608.40 | 2651.34 | 2648.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2608.40 | 2651.34 | 2648.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 2608.40 | 2651.34 | 2648.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 2622.80 | 2645.63 | 2646.04 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 2686.00 | 2646.36 | 2642.62 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 2625.60 | 2638.90 | 2640.46 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 2727.60 | 2653.35 | 2645.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 12:15:00 | 2793.00 | 2753.11 | 2727.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 2770.80 | 2772.33 | 2746.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 2770.80 | 2772.33 | 2746.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2770.80 | 2772.33 | 2746.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2784.90 | 2772.33 | 2746.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 3063.39 | 3012.18 | 2934.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 3198.50 | 3222.39 | 3223.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 3173.30 | 3204.33 | 3213.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 3274.80 | 3210.25 | 3212.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 3274.80 | 3210.25 | 3212.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 3274.80 | 3210.25 | 3212.23 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 3278.60 | 3223.92 | 3218.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 3321.10 | 3253.45 | 3233.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 3383.60 | 3386.44 | 3346.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:45:00 | 3372.70 | 3386.44 | 3346.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 3327.30 | 3397.22 | 3363.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 3327.30 | 3397.22 | 3363.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 3294.30 | 3376.64 | 3357.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 3294.30 | 3376.64 | 3357.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 3430.80 | 3373.87 | 3360.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 3447.00 | 3397.88 | 3374.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 3473.80 | 3406.58 | 3380.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 3346.00 | 3374.44 | 3377.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 3346.00 | 3374.44 | 3377.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 15:15:00 | 3332.00 | 3355.78 | 3367.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 3376.00 | 3359.82 | 3367.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 3376.00 | 3359.82 | 3367.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3376.00 | 3359.82 | 3367.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:30:00 | 3375.00 | 3359.82 | 3367.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 3367.00 | 3361.26 | 3367.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:45:00 | 3374.00 | 3361.26 | 3367.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 3372.00 | 3363.41 | 3368.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:00:00 | 3372.00 | 3363.41 | 3368.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 3355.30 | 3361.79 | 3367.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:30:00 | 3348.60 | 3359.43 | 3365.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 15:00:00 | 3349.50 | 3357.44 | 3364.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 3350.60 | 3354.08 | 3361.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 3391.00 | 3361.47 | 3363.92 | SL hit (close>static) qty=1.00 sl=3372.90 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 3427.00 | 3374.57 | 3369.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 3458.20 | 3391.30 | 3377.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 13:15:00 | 3390.00 | 3391.04 | 3378.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 14:00:00 | 3390.00 | 3391.04 | 3378.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 3408.20 | 3394.47 | 3381.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:45:00 | 3391.60 | 3394.47 | 3381.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 13:15:00 | 2451.90 | 2025-05-29 10:15:00 | 2476.80 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-05-27 13:45:00 | 2448.50 | 2025-05-29 10:15:00 | 2476.80 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-10 11:45:00 | 2645.50 | 2025-06-16 11:15:00 | 2625.10 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-06-10 13:45:00 | 2646.60 | 2025-06-16 11:15:00 | 2625.10 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-06-11 09:15:00 | 2630.30 | 2025-06-16 11:15:00 | 2625.10 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-06-18 09:15:00 | 2631.00 | 2025-06-23 09:15:00 | 2638.60 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-07-01 10:15:00 | 2601.10 | 2025-07-04 09:15:00 | 2634.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2025-07-11 09:15:00 | 2702.10 | 2025-07-17 10:15:00 | 2738.40 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-07-24 09:15:00 | 2768.90 | 2025-07-24 09:15:00 | 2737.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-28 11:30:00 | 2701.00 | 2025-07-30 14:15:00 | 2711.30 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-07-28 12:45:00 | 2696.60 | 2025-07-30 15:15:00 | 2707.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-29 09:30:00 | 2700.10 | 2025-07-30 15:15:00 | 2707.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-07-29 12:45:00 | 2700.00 | 2025-07-30 15:15:00 | 2707.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-07-29 15:00:00 | 2682.40 | 2025-07-30 15:15:00 | 2707.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-08-05 09:15:00 | 2836.50 | 2025-08-08 10:15:00 | 2819.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-08-07 09:30:00 | 2844.70 | 2025-08-08 10:15:00 | 2819.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-26 12:00:00 | 2911.90 | 2025-08-28 11:15:00 | 2865.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-08-26 14:45:00 | 2913.20 | 2025-08-28 11:15:00 | 2865.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-09-10 10:15:00 | 2712.30 | 2025-09-10 13:15:00 | 2791.40 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-09-25 12:45:00 | 2969.20 | 2025-10-01 15:15:00 | 2918.60 | STOP_HIT | 1.00 | 1.70% |
| SELL | retest2 | 2025-09-25 15:00:00 | 2956.00 | 2025-10-01 15:15:00 | 2918.60 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2025-10-06 11:30:00 | 2968.20 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-06 12:15:00 | 2965.90 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-07 09:15:00 | 2973.00 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-09 11:45:00 | 2966.90 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-13 09:30:00 | 3026.40 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-10-29 11:15:00 | 3182.50 | 2025-11-03 13:15:00 | 3192.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-10-29 14:30:00 | 3166.20 | 2025-11-03 13:15:00 | 3192.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-03 11:00:00 | 3178.00 | 2025-11-03 13:15:00 | 3192.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-11-03 12:00:00 | 3179.20 | 2025-11-03 13:15:00 | 3192.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-28 15:15:00 | 3207.00 | 2025-12-03 09:15:00 | 3235.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-01 12:30:00 | 3207.80 | 2025-12-03 09:15:00 | 3235.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-01 14:30:00 | 3208.00 | 2025-12-03 09:15:00 | 3235.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-02 09:45:00 | 3209.40 | 2025-12-03 09:15:00 | 3235.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-08 11:30:00 | 3199.50 | 2025-12-10 09:15:00 | 3240.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-12-09 14:45:00 | 3200.00 | 2025-12-10 09:15:00 | 3240.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-12-18 14:15:00 | 3154.00 | 2025-12-19 11:15:00 | 3180.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-19 09:15:00 | 3150.50 | 2025-12-19 11:15:00 | 3180.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-01-13 13:15:00 | 2927.60 | 2026-01-20 09:15:00 | 2790.24 | PARTIAL | 0.50 | 4.69% |
| SELL | retest2 | 2026-01-13 14:30:00 | 2937.10 | 2026-01-20 10:15:00 | 2781.22 | PARTIAL | 0.50 | 5.31% |
| SELL | retest2 | 2026-01-13 15:00:00 | 2929.10 | 2026-01-20 10:15:00 | 2782.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2909.40 | 2026-01-20 14:15:00 | 2763.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 2903.30 | 2026-01-20 14:15:00 | 2758.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 10:15:00 | 2896.00 | 2026-01-20 15:15:00 | 2751.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:15:00 | 2927.60 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2026-01-13 14:30:00 | 2937.10 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2026-01-13 15:00:00 | 2929.10 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2909.40 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2026-01-14 15:00:00 | 2903.30 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2026-01-16 10:15:00 | 2896.00 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2026-02-03 10:30:00 | 2776.20 | 2026-02-03 14:15:00 | 2769.20 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-02-06 09:15:00 | 2738.30 | 2026-02-09 10:15:00 | 2782.20 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-12 10:45:00 | 2814.20 | 2026-02-13 12:15:00 | 2787.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-12 12:00:00 | 2815.90 | 2026-02-13 12:15:00 | 2787.40 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-02-13 09:30:00 | 2814.00 | 2026-02-13 12:15:00 | 2787.40 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-02-23 11:00:00 | 2707.30 | 2026-03-02 09:15:00 | 2571.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:45:00 | 2707.90 | 2026-03-02 09:15:00 | 2572.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:30:00 | 2704.40 | 2026-03-02 09:15:00 | 2569.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:45:00 | 2706.60 | 2026-03-02 09:15:00 | 2571.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 2684.50 | 2026-03-02 09:15:00 | 2550.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:00:00 | 2707.30 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.84% |
| SELL | retest2 | 2026-02-23 13:45:00 | 2707.90 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2026-02-24 09:30:00 | 2704.40 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2026-02-25 14:45:00 | 2706.60 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.82% |
| SELL | retest2 | 2026-02-26 09:15:00 | 2684.50 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2026-03-18 12:30:00 | 2731.10 | 2026-03-19 13:15:00 | 2594.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 12:30:00 | 2731.10 | 2026-03-20 10:15:00 | 2646.70 | STOP_HIT | 0.50 | 3.09% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2784.90 | 2026-04-17 09:15:00 | 3063.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-04 10:15:00 | 3447.00 | 2026-05-05 12:15:00 | 3346.00 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-05-04 11:15:00 | 3473.80 | 2026-05-05 12:15:00 | 3346.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2026-05-06 13:30:00 | 3348.60 | 2026-05-07 10:15:00 | 3391.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-05-06 15:00:00 | 3349.50 | 2026-05-07 10:15:00 | 3391.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-05-07 10:00:00 | 3350.60 | 2026-05-07 10:15:00 | 3391.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-05-07 10:45:00 | 3349.40 | 2026-05-07 11:15:00 | 3427.00 | STOP_HIT | 1.00 | -2.32% |
