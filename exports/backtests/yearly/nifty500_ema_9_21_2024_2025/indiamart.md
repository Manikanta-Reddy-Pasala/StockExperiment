# Indiamart Intermesh Ltd. (INDIAMART)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 2091.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 145 |
| ALERT1 | 103 |
| ALERT2 | 101 |
| ALERT2_SKIP | 53 |
| ALERT3 | 288 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 110 |
| PARTIAL | 15 |
| TARGET_HIT | 0 |
| STOP_HIT | 118 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 123 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 82
- **Target hits / Stop hits / Partials:** 0 / 112 / 11
- **Avg / median % per leg:** -0.05% / -0.84%
- **Sum % (uncompounded):** -6.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 18 | 35.3% | 0 | 50 | 1 | -0.19% | -9.9% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 4.00% | 12.0% |
| BUY @ 3rd Alert (retest2) | 48 | 15 | 31.2% | 0 | 48 | 0 | -0.46% | -21.9% |
| SELL (all) | 72 | 23 | 31.9% | 0 | 62 | 10 | 0.05% | 3.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 72 | 23 | 31.9% | 0 | 62 | 10 | 0.05% | 3.7% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 2 | 1 | 4.00% | 12.0% |
| retest2 (combined) | 120 | 38 | 31.7% | 0 | 110 | 10 | -0.15% | -18.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 2645.15 | 2627.61 | 2627.20 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 2597.25 | 2622.23 | 2624.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 12:15:00 | 2594.30 | 2611.10 | 2618.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 2606.70 | 2606.24 | 2613.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 2606.70 | 2606.24 | 2613.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 2606.70 | 2606.24 | 2613.54 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 11:15:00 | 2672.40 | 2625.54 | 2621.43 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 2593.20 | 2617.95 | 2619.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 11:15:00 | 2567.10 | 2586.40 | 2599.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 2572.05 | 2564.34 | 2577.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 13:15:00 | 2572.05 | 2564.34 | 2577.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 2572.05 | 2564.34 | 2577.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:45:00 | 2571.55 | 2564.34 | 2577.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2434.25 | 2431.99 | 2458.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 2384.30 | 2441.55 | 2452.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 2411.25 | 2422.00 | 2441.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 2265.09 | 2391.22 | 2422.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 2290.69 | 2391.22 | 2422.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 15:15:00 | 2387.30 | 2387.25 | 2413.00 | SL hit (close>ema200) qty=0.50 sl=2387.25 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 2481.30 | 2433.52 | 2428.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 2499.50 | 2454.15 | 2439.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 12:15:00 | 2537.70 | 2539.53 | 2509.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 13:00:00 | 2537.70 | 2539.53 | 2509.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 2537.90 | 2541.99 | 2520.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:30:00 | 2529.65 | 2541.99 | 2520.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 2550.90 | 2567.30 | 2554.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 2550.90 | 2567.30 | 2554.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 2550.15 | 2563.87 | 2553.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:30:00 | 2577.45 | 2569.37 | 2557.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 11:15:00 | 2567.95 | 2573.76 | 2574.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 11:15:00 | 2567.95 | 2573.76 | 2574.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 13:15:00 | 2558.05 | 2569.78 | 2572.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 14:15:00 | 2578.00 | 2571.43 | 2572.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 14:15:00 | 2578.00 | 2571.43 | 2572.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 2578.00 | 2571.43 | 2572.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 2578.00 | 2571.43 | 2572.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 2578.00 | 2572.74 | 2573.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 2567.85 | 2572.74 | 2573.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 2571.65 | 2565.11 | 2568.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:00:00 | 2571.65 | 2565.11 | 2568.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 2580.95 | 2568.28 | 2570.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:00:00 | 2580.95 | 2568.28 | 2570.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 2562.95 | 2567.21 | 2569.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 14:15:00 | 2560.00 | 2567.21 | 2569.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 15:00:00 | 2556.80 | 2565.13 | 2568.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 2557.00 | 2565.10 | 2567.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 12:15:00 | 2611.80 | 2573.81 | 2570.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 12:15:00 | 2611.80 | 2573.81 | 2570.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 2628.80 | 2595.32 | 2582.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 2652.70 | 2658.26 | 2633.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 15:00:00 | 2652.70 | 2658.26 | 2633.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 2681.70 | 2693.79 | 2678.50 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 2663.80 | 2683.22 | 2685.50 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 2712.10 | 2687.70 | 2685.57 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 2667.15 | 2696.64 | 2698.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 12:15:00 | 2657.20 | 2683.16 | 2691.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 2667.05 | 2667.00 | 2680.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:30:00 | 2665.00 | 2667.00 | 2680.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 2700.00 | 2673.60 | 2681.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:45:00 | 2701.25 | 2673.60 | 2681.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 2716.35 | 2682.15 | 2685.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:00:00 | 2716.35 | 2682.15 | 2685.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 2712.10 | 2688.14 | 2687.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 11:15:00 | 2739.70 | 2711.20 | 2700.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 2858.45 | 2877.58 | 2838.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 2858.45 | 2877.58 | 2838.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 2858.45 | 2877.58 | 2838.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 2850.85 | 2877.58 | 2838.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 2853.80 | 2880.74 | 2856.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 2853.80 | 2880.74 | 2856.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 2865.50 | 2877.69 | 2857.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 2869.00 | 2877.69 | 2857.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 2896.70 | 2881.49 | 2860.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 2853.05 | 2881.49 | 2860.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 2885.00 | 2882.19 | 2863.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:45:00 | 2861.60 | 2882.19 | 2863.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 2862.00 | 2896.33 | 2881.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 2862.00 | 2896.33 | 2881.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 2839.65 | 2885.00 | 2877.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 2839.65 | 2885.00 | 2877.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 2836.30 | 2868.70 | 2870.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 2809.00 | 2856.76 | 2865.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 2843.80 | 2838.31 | 2853.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 2843.80 | 2838.31 | 2853.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2831.15 | 2836.88 | 2851.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 2849.50 | 2836.88 | 2851.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 2841.70 | 2837.84 | 2850.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 2841.70 | 2837.84 | 2850.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 2861.95 | 2842.66 | 2851.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 2860.80 | 2842.66 | 2851.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 2846.20 | 2843.37 | 2850.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 2823.75 | 2843.84 | 2849.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 2828.40 | 2843.42 | 2848.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 2761.05 | 2843.74 | 2848.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 2875.70 | 2850.13 | 2851.05 | SL hit (close>static) qty=1.00 sl=2866.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 2897.10 | 2859.52 | 2855.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 2932.45 | 2882.25 | 2867.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 13:15:00 | 2972.00 | 2988.32 | 2952.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 13:45:00 | 2971.20 | 2988.32 | 2952.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 3050.00 | 3040.94 | 3022.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:45:00 | 3025.95 | 3040.94 | 3022.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 2982.95 | 3066.09 | 3045.13 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 11:15:00 | 2952.55 | 3027.75 | 3030.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 2893.35 | 2963.28 | 2994.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 15:15:00 | 2700.00 | 2683.80 | 2726.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 09:15:00 | 2664.75 | 2683.80 | 2726.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 2728.20 | 2690.24 | 2718.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 2728.20 | 2690.24 | 2718.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 2740.80 | 2700.35 | 2720.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 2732.40 | 2700.35 | 2720.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 2749.00 | 2731.62 | 2731.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 2772.95 | 2739.89 | 2734.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 2789.75 | 2802.56 | 2783.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 14:15:00 | 2789.75 | 2802.56 | 2783.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 2789.75 | 2802.56 | 2783.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:30:00 | 2791.00 | 2802.56 | 2783.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 2785.00 | 2799.05 | 2783.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 2756.25 | 2799.05 | 2783.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 2727.00 | 2784.64 | 2778.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 2721.90 | 2784.64 | 2778.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 2731.15 | 2773.94 | 2774.32 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 2800.05 | 2777.26 | 2775.65 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 15:15:00 | 2759.00 | 2775.13 | 2775.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 2728.10 | 2765.72 | 2771.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 13:15:00 | 2702.00 | 2694.55 | 2719.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 14:00:00 | 2702.00 | 2694.55 | 2719.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 2710.00 | 2690.99 | 2710.90 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 2750.30 | 2720.72 | 2719.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 10:15:00 | 2822.00 | 2755.85 | 2737.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 13:15:00 | 2897.90 | 2898.69 | 2862.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 13:45:00 | 2893.75 | 2898.69 | 2862.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2906.60 | 2920.13 | 2901.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 2965.15 | 2906.26 | 2901.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 09:15:00 | 2983.75 | 3002.13 | 3003.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 2983.75 | 3002.13 | 3003.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 2955.00 | 2989.17 | 2997.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 3050.00 | 2991.25 | 2993.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 3050.00 | 2991.25 | 2993.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 3050.00 | 2991.25 | 2993.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:45:00 | 3055.00 | 2991.25 | 2993.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 3030.00 | 2999.00 | 2996.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 3066.85 | 3032.27 | 3016.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 14:15:00 | 3058.05 | 3062.67 | 3040.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 15:00:00 | 3058.05 | 3062.67 | 3040.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 3035.50 | 3056.16 | 3044.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:45:00 | 3036.00 | 3056.16 | 3044.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 3036.50 | 3052.23 | 3043.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:30:00 | 3031.80 | 3052.23 | 3043.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 2984.95 | 3031.30 | 3035.86 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 3049.00 | 3021.61 | 3020.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 3107.00 | 3038.69 | 3027.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 3090.00 | 3101.92 | 3073.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 09:30:00 | 3103.95 | 3101.92 | 3073.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 3130.95 | 3132.02 | 3121.34 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 15:15:00 | 3110.10 | 3116.07 | 3116.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 11:15:00 | 3085.00 | 3108.44 | 3113.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 2988.00 | 2983.00 | 3014.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 2988.00 | 2983.00 | 3014.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 3015.15 | 2990.61 | 3012.79 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 3046.95 | 3024.86 | 3024.85 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 2982.60 | 3027.82 | 3030.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 12:15:00 | 2975.10 | 3005.92 | 3018.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 2941.00 | 2908.59 | 2933.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 2941.00 | 2908.59 | 2933.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 2941.00 | 2908.59 | 2933.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 2950.70 | 2908.59 | 2933.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 2940.00 | 2914.87 | 2933.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 2938.85 | 2914.87 | 2933.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 2930.95 | 2920.19 | 2933.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:30:00 | 2940.50 | 2920.19 | 2933.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 2916.50 | 2920.29 | 2930.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:30:00 | 2905.05 | 2923.15 | 2930.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 10:15:00 | 2940.30 | 2926.58 | 2931.30 | SL hit (close>static) qty=1.00 sl=2935.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 13:15:00 | 2958.80 | 2936.51 | 2934.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 09:15:00 | 3016.65 | 2954.73 | 2943.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 2947.00 | 2979.13 | 2966.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 2947.00 | 2979.13 | 2966.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 2947.00 | 2979.13 | 2966.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 2947.00 | 2979.13 | 2966.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 2936.30 | 2970.57 | 2963.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 2936.30 | 2970.57 | 2963.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 2918.95 | 2960.24 | 2959.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:30:00 | 2910.00 | 2960.24 | 2959.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 2921.50 | 2952.49 | 2956.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 2888.70 | 2939.74 | 2950.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2885.05 | 2818.31 | 2852.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 2885.05 | 2818.31 | 2852.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 2885.05 | 2818.31 | 2852.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 2885.05 | 2818.31 | 2852.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 2919.30 | 2838.51 | 2858.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 2919.30 | 2838.51 | 2858.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 2935.15 | 2882.13 | 2875.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 2964.75 | 2898.66 | 2883.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 2984.85 | 2993.11 | 2966.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 2984.85 | 2993.11 | 2966.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 2998.45 | 3005.70 | 2994.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:45:00 | 2996.05 | 3005.70 | 2994.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 3007.15 | 3005.99 | 2995.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 14:30:00 | 3012.20 | 3009.17 | 2997.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 15:00:00 | 3021.90 | 3009.17 | 2997.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 3036.30 | 3041.37 | 3025.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 10:00:00 | 3019.55 | 3042.30 | 3035.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 3035.70 | 3040.98 | 3035.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:00:00 | 3045.90 | 3041.96 | 3036.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 13:00:00 | 3048.15 | 3043.20 | 3037.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:15:00 | 3055.10 | 3041.70 | 3038.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 13:15:00 | 3025.55 | 3035.25 | 3036.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 13:15:00 | 3025.55 | 3035.25 | 3036.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 14:15:00 | 3005.00 | 3029.20 | 3033.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 13:15:00 | 2507.30 | 2486.86 | 2534.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 14:00:00 | 2507.30 | 2486.86 | 2534.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 2471.65 | 2458.47 | 2480.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 2473.85 | 2458.47 | 2480.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 2493.75 | 2465.53 | 2481.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:00:00 | 2493.75 | 2465.53 | 2481.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 2522.95 | 2477.01 | 2485.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:00:00 | 2522.95 | 2477.01 | 2485.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 2523.00 | 2492.54 | 2491.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 2530.45 | 2500.12 | 2495.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 2548.10 | 2560.52 | 2541.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 15:00:00 | 2548.10 | 2560.52 | 2541.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 2544.50 | 2557.31 | 2542.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 2545.65 | 2557.31 | 2542.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 2538.15 | 2553.48 | 2541.70 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 2498.05 | 2533.79 | 2534.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2449.00 | 2513.22 | 2523.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 12:15:00 | 2456.75 | 2440.78 | 2456.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 12:15:00 | 2456.75 | 2440.78 | 2456.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 2456.75 | 2440.78 | 2456.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 2456.75 | 2440.78 | 2456.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 2461.70 | 2444.96 | 2456.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:00:00 | 2461.70 | 2444.96 | 2456.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 2461.65 | 2448.30 | 2457.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:15:00 | 2438.90 | 2451.54 | 2457.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 14:15:00 | 2316.95 | 2343.94 | 2368.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 2315.60 | 2306.06 | 2328.63 | SL hit (close>ema200) qty=0.50 sl=2306.06 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 2281.45 | 2271.63 | 2271.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 2289.30 | 2277.07 | 2274.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 2361.40 | 2364.66 | 2344.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 10:45:00 | 2365.40 | 2364.66 | 2344.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 2348.80 | 2357.74 | 2345.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:45:00 | 2346.10 | 2357.74 | 2345.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 2344.05 | 2355.00 | 2345.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 2344.05 | 2355.00 | 2345.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 2335.05 | 2351.01 | 2344.69 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 12:15:00 | 2328.20 | 2340.76 | 2341.37 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 2346.60 | 2341.93 | 2341.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 2358.30 | 2347.36 | 2344.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 14:15:00 | 2360.25 | 2362.87 | 2356.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 14:15:00 | 2360.25 | 2362.87 | 2356.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 2360.25 | 2362.87 | 2356.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:45:00 | 2358.35 | 2362.87 | 2356.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 2360.00 | 2362.30 | 2356.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 2365.75 | 2362.30 | 2356.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 2360.95 | 2362.03 | 2357.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:15:00 | 2380.00 | 2363.50 | 2361.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 15:15:00 | 2360.00 | 2362.73 | 2363.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 15:15:00 | 2360.00 | 2362.73 | 2363.00 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 2368.60 | 2363.89 | 2363.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 2383.55 | 2369.63 | 2366.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 14:15:00 | 2377.00 | 2378.53 | 2373.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 15:00:00 | 2377.00 | 2378.53 | 2373.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 2383.85 | 2389.80 | 2384.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 2383.85 | 2389.80 | 2384.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 2363.05 | 2384.45 | 2382.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 2363.05 | 2384.45 | 2382.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 2365.00 | 2380.56 | 2380.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 2346.00 | 2370.38 | 2375.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 2364.40 | 2354.29 | 2364.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 12:00:00 | 2364.40 | 2354.29 | 2364.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 2375.75 | 2358.58 | 2365.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 2375.75 | 2358.58 | 2365.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 2371.90 | 2361.25 | 2366.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:30:00 | 2372.15 | 2361.25 | 2366.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 2383.00 | 2369.39 | 2369.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 2395.55 | 2374.62 | 2371.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 14:15:00 | 2377.10 | 2381.68 | 2377.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 14:15:00 | 2377.10 | 2381.68 | 2377.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 2377.10 | 2381.68 | 2377.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 2377.10 | 2381.68 | 2377.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 2371.80 | 2379.70 | 2376.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 2372.00 | 2379.70 | 2376.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 2387.50 | 2381.26 | 2377.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:30:00 | 2386.45 | 2381.26 | 2377.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 2394.90 | 2383.99 | 2379.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 2388.25 | 2383.99 | 2379.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 2380.15 | 2383.22 | 2379.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 2380.15 | 2383.22 | 2379.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 2376.90 | 2381.96 | 2379.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 2376.90 | 2381.96 | 2379.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 2376.20 | 2380.81 | 2378.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:30:00 | 2383.15 | 2380.81 | 2378.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 2367.55 | 2378.15 | 2377.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:00:00 | 2367.55 | 2378.15 | 2377.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 15:15:00 | 2368.10 | 2376.14 | 2376.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 2358.40 | 2371.37 | 2374.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 2363.65 | 2362.16 | 2367.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 11:15:00 | 2363.65 | 2362.16 | 2367.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 2363.65 | 2362.16 | 2367.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:45:00 | 2363.05 | 2362.16 | 2367.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 2360.70 | 2361.87 | 2366.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:30:00 | 2364.35 | 2361.87 | 2366.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 2360.10 | 2361.52 | 2366.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:45:00 | 2368.25 | 2361.52 | 2366.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 2365.45 | 2362.30 | 2366.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 2365.45 | 2362.30 | 2366.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2355.10 | 2360.86 | 2365.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 2367.00 | 2360.86 | 2365.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2330.00 | 2354.69 | 2361.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 2362.80 | 2354.69 | 2361.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 2239.40 | 2232.39 | 2247.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:45:00 | 2232.00 | 2233.11 | 2246.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 12:15:00 | 2258.95 | 2241.82 | 2248.33 | SL hit (close>static) qty=1.00 sl=2258.50 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 2265.25 | 2253.25 | 2252.39 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 2231.25 | 2251.75 | 2252.60 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 2264.85 | 2250.93 | 2250.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 2280.60 | 2264.98 | 2258.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 2289.65 | 2294.87 | 2282.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:00:00 | 2289.65 | 2294.87 | 2282.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 2290.80 | 2294.06 | 2283.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:45:00 | 2292.15 | 2294.06 | 2283.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 2267.65 | 2288.78 | 2282.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:45:00 | 2270.10 | 2288.78 | 2282.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 2259.55 | 2282.93 | 2280.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 2259.55 | 2282.93 | 2280.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 2253.70 | 2277.08 | 2277.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 2231.95 | 2264.04 | 2271.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 2237.70 | 2212.17 | 2234.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 2237.70 | 2212.17 | 2234.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 2237.70 | 2212.17 | 2234.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:15:00 | 2271.80 | 2212.17 | 2234.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 2273.70 | 2224.48 | 2238.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 2273.70 | 2224.48 | 2238.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 2254.80 | 2230.54 | 2239.97 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 2306.20 | 2250.79 | 2247.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 09:15:00 | 2358.25 | 2303.36 | 2282.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 2323.40 | 2324.90 | 2303.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 15:00:00 | 2323.40 | 2324.90 | 2303.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 2295.00 | 2317.59 | 2303.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 2295.00 | 2317.59 | 2303.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 2317.05 | 2317.49 | 2304.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:45:00 | 2319.40 | 2315.47 | 2304.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:30:00 | 2322.50 | 2314.97 | 2306.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 14:15:00 | 2324.30 | 2314.97 | 2306.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 15:00:00 | 2324.60 | 2316.89 | 2308.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 2298.50 | 2313.16 | 2308.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 2298.50 | 2313.16 | 2308.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 2277.15 | 2305.96 | 2305.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-13 10:15:00 | 2277.15 | 2305.96 | 2305.20 | SL hit (close<static) qty=1.00 sl=2285.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 2260.05 | 2296.78 | 2301.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 2256.60 | 2288.74 | 2297.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 2273.65 | 2268.73 | 2283.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 2273.65 | 2268.73 | 2283.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 2273.65 | 2268.73 | 2283.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:45:00 | 2244.75 | 2265.18 | 2278.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 14:30:00 | 2244.55 | 2259.14 | 2272.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 2227.80 | 2260.32 | 2272.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:45:00 | 2235.90 | 2225.65 | 2240.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 2238.45 | 2228.51 | 2238.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 2238.45 | 2228.51 | 2238.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 2244.00 | 2231.61 | 2238.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 2240.60 | 2231.61 | 2238.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 2243.05 | 2233.90 | 2238.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-17 11:15:00 | 2260.00 | 2245.36 | 2243.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 11:15:00 | 2260.00 | 2245.36 | 2243.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 09:15:00 | 2302.55 | 2274.90 | 2264.29 | Break + close above crossover candle high |

### Cycle 48 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 2142.50 | 2263.89 | 2267.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 2055.35 | 2081.58 | 2110.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 11:15:00 | 2097.50 | 2081.95 | 2105.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 12:00:00 | 2097.50 | 2081.95 | 2105.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 2085.95 | 2085.68 | 2094.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 2081.50 | 2084.75 | 2093.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:15:00 | 2082.30 | 2082.25 | 2089.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:45:00 | 2081.45 | 2082.57 | 2088.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:45:00 | 2077.00 | 2080.54 | 2085.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 2081.25 | 2080.68 | 2084.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:00:00 | 2081.25 | 2080.68 | 2084.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 2072.15 | 2078.97 | 2083.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:30:00 | 2082.60 | 2078.97 | 2083.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 2078.40 | 2074.89 | 2080.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:45:00 | 2078.00 | 2074.89 | 2080.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 2072.25 | 2075.17 | 2079.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 10:30:00 | 2063.25 | 2072.79 | 2078.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 11:30:00 | 2068.95 | 2070.82 | 2077.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 13:15:00 | 2144.00 | 2088.90 | 2081.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 2144.00 | 2088.90 | 2081.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 14:15:00 | 2171.45 | 2105.41 | 2090.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 10:15:00 | 2147.45 | 2164.58 | 2142.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 11:00:00 | 2147.45 | 2164.58 | 2142.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 2174.50 | 2166.56 | 2145.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 12:15:00 | 2176.60 | 2166.56 | 2145.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 14:30:00 | 2188.75 | 2171.00 | 2152.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:30:00 | 2187.90 | 2178.73 | 2159.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 15:00:00 | 2183.00 | 2184.92 | 2179.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 2171.50 | 2182.24 | 2178.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 2177.85 | 2182.24 | 2178.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 2176.60 | 2181.11 | 2178.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 2161.65 | 2181.11 | 2178.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2177.40 | 2180.37 | 2178.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 2169.15 | 2180.37 | 2178.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 2165.35 | 2175.58 | 2176.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 12:15:00 | 2165.35 | 2175.58 | 2176.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 2153.45 | 2171.15 | 2174.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 2174.65 | 2171.85 | 2174.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 14:15:00 | 2174.65 | 2171.85 | 2174.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 2174.65 | 2171.85 | 2174.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 2174.65 | 2171.85 | 2174.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 2173.90 | 2172.26 | 2174.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 2180.00 | 2172.26 | 2174.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 2185.05 | 2174.82 | 2175.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 2185.05 | 2174.82 | 2175.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 10:15:00 | 2192.90 | 2178.44 | 2177.02 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 2142.50 | 2171.26 | 2174.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 2135.70 | 2159.47 | 2168.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 2149.80 | 2143.95 | 2155.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 2149.80 | 2143.95 | 2155.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 2170.00 | 2149.16 | 2156.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 2170.00 | 2149.16 | 2156.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 2179.40 | 2155.21 | 2158.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 2179.40 | 2155.21 | 2158.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 15:15:00 | 2168.40 | 2161.56 | 2161.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 2213.00 | 2171.85 | 2165.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 15:15:00 | 2192.00 | 2194.78 | 2182.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 09:15:00 | 2178.10 | 2194.78 | 2182.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 2144.00 | 2184.62 | 2179.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 2144.00 | 2184.62 | 2179.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 2134.10 | 2174.52 | 2175.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 2111.30 | 2161.87 | 2169.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 2081.65 | 2072.17 | 2093.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 2037.00 | 2057.46 | 2068.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 2037.00 | 2057.46 | 2068.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 10:15:00 | 2033.45 | 2057.46 | 2068.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 11:45:00 | 2031.70 | 2048.38 | 2062.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 11:45:00 | 2026.45 | 2025.75 | 2041.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 12:45:00 | 2034.55 | 2027.13 | 2041.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 2002.95 | 2014.74 | 2030.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 10:45:00 | 1974.75 | 1992.81 | 2009.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 15:15:00 | 1931.78 | 1973.95 | 1992.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 15:15:00 | 1930.12 | 1973.95 | 1992.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 15:15:00 | 1925.13 | 1973.95 | 1992.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 15:15:00 | 1932.82 | 1973.95 | 1992.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 1956.05 | 1954.83 | 1972.33 | SL hit (close>ema200) qty=0.50 sl=1954.83 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 1970.40 | 1962.31 | 1962.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 1973.50 | 1964.55 | 1963.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 2004.00 | 2007.88 | 1997.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 09:15:00 | 1991.30 | 2007.88 | 1997.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2001.45 | 2006.60 | 1998.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 1995.00 | 2006.60 | 1998.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 2005.20 | 2006.32 | 1998.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 12:00:00 | 2011.05 | 2007.26 | 1999.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:15:00 | 2010.70 | 2011.29 | 2003.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 15:00:00 | 2012.00 | 2011.43 | 2004.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 1953.05 | 1998.41 | 1999.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1953.05 | 1998.41 | 1999.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 11:15:00 | 1942.70 | 1979.84 | 1990.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 10:15:00 | 1950.00 | 1938.83 | 1952.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 10:15:00 | 1950.00 | 1938.83 | 1952.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1950.00 | 1938.83 | 1952.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:45:00 | 1950.00 | 1938.83 | 1952.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1951.35 | 1941.34 | 1952.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:30:00 | 1950.00 | 1941.34 | 1952.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 1950.00 | 1943.07 | 1951.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:45:00 | 1950.05 | 1943.07 | 1951.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 1955.00 | 1945.46 | 1952.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:00:00 | 1955.00 | 1945.46 | 1952.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1955.00 | 1947.36 | 1952.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:45:00 | 1955.20 | 1947.36 | 1952.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1960.70 | 1953.29 | 1954.10 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 1977.90 | 1958.22 | 1956.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 2024.50 | 1975.60 | 1965.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 11:15:00 | 2101.00 | 2108.77 | 2083.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 12:00:00 | 2101.00 | 2108.77 | 2083.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 2101.05 | 2128.19 | 2115.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 2101.05 | 2128.19 | 2115.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 2096.30 | 2121.81 | 2113.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 2096.30 | 2121.81 | 2113.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 2101.75 | 2117.80 | 2112.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 2100.10 | 2117.80 | 2112.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 2091.05 | 2106.44 | 2108.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 2081.70 | 2097.18 | 2103.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 2079.00 | 2077.79 | 2087.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 2079.00 | 2077.79 | 2087.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 2079.00 | 2077.79 | 2087.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 2087.25 | 2077.79 | 2087.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2027.00 | 2065.53 | 2079.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 10:15:00 | 2021.20 | 2065.53 | 2079.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 2077.55 | 2066.64 | 2066.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 2077.55 | 2066.64 | 2066.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 2107.40 | 2080.62 | 2073.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 2094.75 | 2094.97 | 2084.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 13:00:00 | 2094.75 | 2094.97 | 2084.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 2055.75 | 2101.95 | 2092.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 2055.75 | 2101.95 | 2092.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 2065.60 | 2094.68 | 2089.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 2047.65 | 2094.68 | 2089.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 2065.95 | 2083.10 | 2085.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1950.00 | 2052.73 | 2070.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 12:15:00 | 2011.95 | 1976.18 | 2003.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 12:15:00 | 2011.95 | 1976.18 | 2003.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 2011.95 | 1976.18 | 2003.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:45:00 | 2005.20 | 1976.18 | 2003.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1980.80 | 1977.11 | 2001.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1967.80 | 1981.29 | 1999.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 1963.90 | 1977.81 | 1995.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2037.75 | 1994.18 | 1994.59 | SL hit (close>static) qty=1.00 sl=2014.70 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 2064.90 | 2008.33 | 2000.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 2069.30 | 2020.52 | 2007.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 2110.90 | 2112.84 | 2081.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:45:00 | 2142.80 | 2117.28 | 2089.06 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:15:00 | 2249.94 | 2183.19 | 2151.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 2258.80 | 2267.81 | 2236.98 | SL hit (close<ema200) qty=0.50 sl=2267.81 alert=retest1 |

### Cycle 62 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 2256.20 | 2268.09 | 2268.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 2245.20 | 2263.52 | 2266.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 15:15:00 | 2239.70 | 2239.35 | 2249.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 09:15:00 | 2270.30 | 2239.35 | 2249.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2254.20 | 2242.32 | 2249.91 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 2324.90 | 2267.43 | 2260.54 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 14:15:00 | 2260.10 | 2287.91 | 2291.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 09:15:00 | 2241.10 | 2274.10 | 2283.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 2219.40 | 2218.90 | 2237.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 10:45:00 | 2209.30 | 2218.90 | 2237.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 2244.70 | 2225.37 | 2237.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 2244.70 | 2225.37 | 2237.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 2258.90 | 2232.08 | 2239.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 2258.90 | 2232.08 | 2239.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 2276.00 | 2247.33 | 2245.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 2308.10 | 2259.48 | 2251.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 2290.50 | 2296.32 | 2276.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 2290.50 | 2296.32 | 2276.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 2273.00 | 2291.65 | 2275.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 2220.30 | 2291.65 | 2275.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2240.00 | 2281.32 | 2272.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:30:00 | 2247.20 | 2281.32 | 2272.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 12:15:00 | 2254.30 | 2265.63 | 2266.70 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2309.40 | 2271.27 | 2268.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 2353.30 | 2316.36 | 2299.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 2324.20 | 2325.07 | 2306.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2324.20 | 2325.07 | 2306.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2324.20 | 2325.07 | 2306.55 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 15:15:00 | 2288.00 | 2300.17 | 2300.68 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 2328.40 | 2305.82 | 2303.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 2335.10 | 2315.59 | 2308.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 15:15:00 | 2322.00 | 2323.97 | 2315.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 15:15:00 | 2322.00 | 2323.97 | 2315.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 2322.00 | 2323.97 | 2315.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 2360.10 | 2323.97 | 2315.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 2346.80 | 2325.92 | 2317.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 2367.70 | 2382.60 | 2383.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 2367.70 | 2382.60 | 2383.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 2330.40 | 2370.15 | 2377.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 2314.70 | 2313.25 | 2328.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:30:00 | 2315.30 | 2313.25 | 2328.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 2316.20 | 2313.84 | 2327.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 2316.20 | 2313.84 | 2327.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 2319.50 | 2316.57 | 2325.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 2319.50 | 2316.57 | 2325.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 2319.00 | 2317.61 | 2324.51 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 2345.70 | 2329.01 | 2326.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 2395.30 | 2342.04 | 2334.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 2355.40 | 2356.27 | 2345.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 2355.40 | 2356.27 | 2345.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 2355.40 | 2356.27 | 2345.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:30:00 | 2357.50 | 2356.27 | 2345.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 2395.50 | 2408.49 | 2389.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 2394.70 | 2408.49 | 2389.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 2431.00 | 2411.83 | 2398.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:45:00 | 2413.80 | 2411.83 | 2398.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 2411.50 | 2416.43 | 2403.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:00:00 | 2430.00 | 2418.28 | 2407.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 13:15:00 | 2450.00 | 2471.19 | 2473.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 13:15:00 | 2450.00 | 2471.19 | 2473.30 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 2492.90 | 2474.65 | 2473.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 2498.00 | 2486.85 | 2480.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 13:15:00 | 2479.40 | 2489.57 | 2484.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 13:15:00 | 2479.40 | 2489.57 | 2484.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 2479.40 | 2489.57 | 2484.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:00:00 | 2479.40 | 2489.57 | 2484.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 2480.20 | 2487.70 | 2484.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 2487.30 | 2485.25 | 2483.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 2470.90 | 2482.11 | 2482.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 2470.90 | 2482.11 | 2482.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 2454.00 | 2469.43 | 2475.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 2492.00 | 2460.19 | 2465.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 2492.00 | 2460.19 | 2465.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 2492.00 | 2460.19 | 2465.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 2488.40 | 2460.19 | 2465.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 2484.60 | 2465.07 | 2467.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 2487.40 | 2465.07 | 2467.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 2490.60 | 2470.18 | 2469.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 2495.80 | 2480.76 | 2474.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 2507.40 | 2518.47 | 2502.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:00:00 | 2507.40 | 2518.47 | 2502.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 2497.30 | 2514.23 | 2502.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 2495.30 | 2514.23 | 2502.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 2489.80 | 2509.35 | 2501.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 2489.80 | 2509.35 | 2501.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2605.70 | 2631.19 | 2611.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:45:00 | 2612.00 | 2631.19 | 2611.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 2594.60 | 2623.87 | 2609.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 2594.60 | 2623.87 | 2609.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 2605.00 | 2615.19 | 2608.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 2602.80 | 2615.19 | 2608.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 2590.80 | 2610.31 | 2607.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 2588.90 | 2610.31 | 2607.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 2565.40 | 2601.33 | 2603.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 2558.90 | 2578.76 | 2586.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 2581.00 | 2575.41 | 2583.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 2581.00 | 2575.41 | 2583.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 2581.00 | 2575.41 | 2583.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 2581.00 | 2575.41 | 2583.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 2588.20 | 2577.97 | 2583.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 2555.20 | 2577.97 | 2583.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2550.90 | 2572.55 | 2580.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:45:00 | 2536.70 | 2556.13 | 2567.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 2538.00 | 2547.52 | 2558.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 2536.40 | 2545.14 | 2552.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 2539.50 | 2542.50 | 2549.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2545.10 | 2543.02 | 2548.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 2545.10 | 2543.02 | 2548.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 2545.60 | 2543.54 | 2548.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 2545.60 | 2543.54 | 2548.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 2541.60 | 2543.15 | 2547.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 2586.20 | 2554.50 | 2552.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 2586.20 | 2554.50 | 2552.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 2591.10 | 2571.22 | 2562.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 2568.30 | 2574.78 | 2565.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:00:00 | 2568.30 | 2574.78 | 2565.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 2578.90 | 2575.60 | 2566.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 11:30:00 | 2580.80 | 2576.54 | 2567.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:00:00 | 2580.30 | 2576.54 | 2567.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 2593.00 | 2582.99 | 2574.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 2650.90 | 2673.34 | 2675.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 09:15:00 | 2650.90 | 2673.34 | 2675.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 2644.20 | 2663.55 | 2670.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 2644.40 | 2628.91 | 2641.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 2644.40 | 2628.91 | 2641.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 2644.40 | 2628.91 | 2641.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 2644.40 | 2628.91 | 2641.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 2655.00 | 2634.13 | 2642.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 2648.60 | 2634.13 | 2642.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 2613.00 | 2629.90 | 2640.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 10:45:00 | 2603.00 | 2622.72 | 2636.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 2626.90 | 2593.84 | 2592.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 2626.90 | 2593.84 | 2592.22 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 2566.20 | 2597.17 | 2598.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 2555.30 | 2583.54 | 2592.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 2579.50 | 2575.14 | 2584.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:00:00 | 2579.50 | 2575.14 | 2584.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 2588.50 | 2556.56 | 2564.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 2588.50 | 2556.56 | 2564.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 2585.00 | 2562.25 | 2566.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 2612.20 | 2562.25 | 2566.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 2620.00 | 2573.80 | 2570.91 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 2544.40 | 2584.51 | 2585.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 10:15:00 | 2508.60 | 2569.33 | 2578.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 2520.00 | 2498.65 | 2520.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 2520.00 | 2498.65 | 2520.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2520.00 | 2498.65 | 2520.34 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 2556.90 | 2520.86 | 2519.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 10:15:00 | 2562.90 | 2535.98 | 2527.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 2536.20 | 2548.39 | 2539.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2536.20 | 2548.39 | 2539.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2536.20 | 2548.39 | 2539.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 2536.60 | 2548.39 | 2539.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2544.40 | 2547.59 | 2539.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:15:00 | 2538.10 | 2547.59 | 2539.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2548.40 | 2547.75 | 2540.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:00:00 | 2562.90 | 2546.88 | 2542.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 13:15:00 | 2530.10 | 2553.15 | 2547.70 | SL hit (close<static) qty=1.00 sl=2531.30 alert=retest2 |

### Cycle 84 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 2525.00 | 2542.71 | 2543.60 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 2561.30 | 2546.43 | 2545.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 2575.30 | 2552.20 | 2547.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 2543.40 | 2550.44 | 2547.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 2543.40 | 2550.44 | 2547.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2543.40 | 2550.44 | 2547.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 2543.40 | 2550.44 | 2547.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 2542.80 | 2548.91 | 2547.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 2542.80 | 2548.91 | 2547.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 2544.50 | 2548.03 | 2546.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:30:00 | 2549.20 | 2549.12 | 2547.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:00:00 | 2548.80 | 2549.95 | 2548.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:30:00 | 2550.10 | 2549.52 | 2548.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:00:00 | 2548.80 | 2549.38 | 2548.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 2543.20 | 2548.14 | 2547.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:45:00 | 2536.40 | 2548.14 | 2547.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 2551.00 | 2548.71 | 2548.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 2542.90 | 2548.71 | 2548.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 2534.90 | 2545.95 | 2547.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 2534.90 | 2545.95 | 2547.04 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 14:15:00 | 2572.00 | 2550.41 | 2548.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 2584.80 | 2557.29 | 2551.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 2579.80 | 2585.83 | 2573.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:15:00 | 2607.10 | 2585.83 | 2573.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 2648.50 | 2661.33 | 2649.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 2648.50 | 2661.33 | 2649.92 | SL hit (close<ema400) qty=1.00 sl=2649.92 alert=retest1 |

### Cycle 88 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 2638.70 | 2645.27 | 2645.50 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 2656.90 | 2645.68 | 2645.47 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 2580.00 | 2633.23 | 2639.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 10:15:00 | 2568.20 | 2620.23 | 2633.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 2598.40 | 2589.71 | 2609.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:00:00 | 2598.40 | 2589.71 | 2609.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 2598.90 | 2592.73 | 2607.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:15:00 | 2588.00 | 2593.54 | 2606.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 2590.00 | 2592.51 | 2603.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 2618.00 | 2597.20 | 2603.83 | SL hit (close>static) qty=1.00 sl=2610.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 2578.20 | 2558.87 | 2557.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2588.30 | 2574.13 | 2569.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 2575.00 | 2577.44 | 2572.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 12:15:00 | 2575.00 | 2577.44 | 2572.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 2575.00 | 2577.44 | 2572.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 2563.90 | 2577.44 | 2572.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 2615.00 | 2584.67 | 2577.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 2626.00 | 2604.14 | 2600.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 2594.00 | 2600.49 | 2601.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 2594.00 | 2600.49 | 2601.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 09:15:00 | 2567.50 | 2593.89 | 2598.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 12:15:00 | 2387.10 | 2380.15 | 2401.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 13:15:00 | 2386.50 | 2380.15 | 2401.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2375.40 | 2367.94 | 2378.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 2344.60 | 2364.10 | 2375.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 2327.80 | 2356.45 | 2368.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 2336.50 | 2350.40 | 2355.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 2345.60 | 2350.72 | 2355.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 2350.40 | 2350.65 | 2355.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 2347.40 | 2350.65 | 2355.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 2351.60 | 2350.84 | 2354.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 2351.60 | 2350.84 | 2354.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 2370.40 | 2354.76 | 2356.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 2370.40 | 2354.76 | 2356.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 2380.10 | 2359.82 | 2358.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 2380.10 | 2359.82 | 2358.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 2400.80 | 2368.02 | 2362.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 2377.00 | 2382.74 | 2374.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 13:45:00 | 2377.00 | 2382.74 | 2374.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 2373.90 | 2380.97 | 2374.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 2373.90 | 2380.97 | 2374.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 2365.80 | 2377.94 | 2373.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 2362.80 | 2377.94 | 2373.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 2363.80 | 2375.11 | 2372.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:45:00 | 2347.60 | 2375.11 | 2372.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 2369.40 | 2373.97 | 2372.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:30:00 | 2364.00 | 2373.97 | 2372.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 2373.20 | 2373.82 | 2372.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 2370.10 | 2373.82 | 2372.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 2377.10 | 2374.47 | 2372.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:45:00 | 2385.00 | 2376.46 | 2373.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 2392.70 | 2375.93 | 2374.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 11:45:00 | 2380.00 | 2378.45 | 2375.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 2379.60 | 2378.68 | 2376.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 2375.80 | 2378.10 | 2376.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 2375.80 | 2378.10 | 2376.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 2359.60 | 2374.40 | 2374.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 2359.60 | 2374.40 | 2374.62 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 2450.00 | 2387.60 | 2379.55 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 2389.60 | 2399.21 | 2399.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 15:15:00 | 2380.00 | 2395.37 | 2397.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 2405.50 | 2397.40 | 2398.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 2405.50 | 2397.40 | 2398.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 2405.50 | 2397.40 | 2398.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 2401.40 | 2397.40 | 2398.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 2411.50 | 2400.22 | 2399.45 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 2363.70 | 2394.51 | 2397.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2339.40 | 2383.49 | 2392.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 2350.60 | 2347.63 | 2365.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 2350.60 | 2347.63 | 2365.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2348.00 | 2341.51 | 2354.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 2365.80 | 2341.51 | 2354.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 2349.10 | 2343.03 | 2354.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 2349.10 | 2343.03 | 2354.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 2356.10 | 2345.64 | 2354.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 2363.80 | 2345.64 | 2354.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 2366.60 | 2349.83 | 2355.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 2366.60 | 2349.83 | 2355.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 2359.20 | 2351.71 | 2355.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:45:00 | 2355.50 | 2352.29 | 2355.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 2343.20 | 2353.45 | 2355.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 2355.10 | 2345.63 | 2347.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 2375.00 | 2347.17 | 2347.66 | SL hit (close>static) qty=1.00 sl=2368.90 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 2379.00 | 2353.54 | 2350.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 2407.60 | 2372.06 | 2361.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 2441.50 | 2457.13 | 2433.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 10:00:00 | 2441.50 | 2457.13 | 2433.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 2467.00 | 2459.11 | 2436.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 2441.40 | 2459.11 | 2436.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2456.80 | 2470.31 | 2453.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 2454.10 | 2470.31 | 2453.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 2470.00 | 2471.58 | 2460.47 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 2451.20 | 2460.20 | 2460.70 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 2479.00 | 2461.85 | 2461.13 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 2435.50 | 2456.90 | 2459.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 2431.40 | 2441.69 | 2446.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 2447.20 | 2436.78 | 2441.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 14:15:00 | 2447.20 | 2436.78 | 2441.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 2447.20 | 2436.78 | 2441.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 2462.20 | 2436.78 | 2441.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 2449.50 | 2439.33 | 2442.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 2450.40 | 2439.33 | 2442.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2484.20 | 2448.30 | 2446.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 2500.00 | 2465.47 | 2454.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 2473.90 | 2488.47 | 2476.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 2473.90 | 2488.47 | 2476.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 2473.90 | 2488.47 | 2476.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 2473.10 | 2488.47 | 2476.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 2486.20 | 2488.02 | 2477.50 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 2451.00 | 2471.08 | 2472.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 2431.30 | 2457.49 | 2465.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 11:15:00 | 2466.60 | 2458.95 | 2464.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 11:15:00 | 2466.60 | 2458.95 | 2464.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 2466.60 | 2458.95 | 2464.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 2466.60 | 2458.95 | 2464.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 2474.30 | 2462.02 | 2465.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 2474.30 | 2462.02 | 2465.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 2475.60 | 2464.74 | 2466.45 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 2477.90 | 2469.25 | 2468.32 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2445.00 | 2464.40 | 2466.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 2441.10 | 2459.74 | 2463.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 2320.90 | 2318.70 | 2340.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 2320.90 | 2318.70 | 2340.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 2336.40 | 2324.53 | 2339.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 2336.40 | 2324.53 | 2339.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2353.90 | 2330.40 | 2340.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 2353.90 | 2330.40 | 2340.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 2361.90 | 2336.70 | 2342.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 2335.20 | 2336.70 | 2342.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 2336.60 | 2321.46 | 2320.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 2336.60 | 2321.46 | 2320.65 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 2313.40 | 2321.74 | 2321.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 2302.30 | 2316.09 | 2319.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 2310.30 | 2303.32 | 2308.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 2310.30 | 2303.32 | 2308.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 2310.30 | 2303.32 | 2308.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 2310.30 | 2303.32 | 2308.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 2319.00 | 2306.45 | 2309.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 2290.00 | 2306.45 | 2309.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 2272.10 | 2245.57 | 2243.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 2272.10 | 2245.57 | 2243.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 2280.20 | 2256.24 | 2249.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2247.20 | 2256.10 | 2250.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 2247.20 | 2256.10 | 2250.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2247.20 | 2256.10 | 2250.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 2246.40 | 2256.10 | 2250.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2275.30 | 2259.94 | 2253.18 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 2234.50 | 2249.09 | 2249.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 2222.90 | 2243.85 | 2247.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 2240.00 | 2232.63 | 2238.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 2240.00 | 2232.63 | 2238.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 2240.00 | 2232.63 | 2238.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 2240.00 | 2232.63 | 2238.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 2238.70 | 2233.85 | 2238.28 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 2259.10 | 2241.65 | 2240.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 13:15:00 | 2271.30 | 2247.58 | 2243.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 10:15:00 | 2245.00 | 2252.89 | 2247.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 2245.00 | 2252.89 | 2247.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2245.00 | 2252.89 | 2247.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 2245.00 | 2252.89 | 2247.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2238.50 | 2250.01 | 2247.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 2238.50 | 2250.01 | 2247.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 2224.00 | 2244.81 | 2244.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 14:15:00 | 2216.60 | 2235.26 | 2240.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 15:15:00 | 2218.00 | 2217.40 | 2226.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 2219.90 | 2217.40 | 2226.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2242.00 | 2222.32 | 2227.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:00:00 | 2242.00 | 2222.32 | 2227.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2233.30 | 2224.52 | 2227.98 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 13:15:00 | 2259.00 | 2235.54 | 2232.52 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 2217.30 | 2235.17 | 2236.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 2206.00 | 2222.57 | 2229.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 2214.20 | 2212.62 | 2221.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 2214.20 | 2212.62 | 2221.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2214.20 | 2212.62 | 2221.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 2214.20 | 2212.62 | 2221.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2197.90 | 2209.68 | 2219.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 2220.00 | 2209.68 | 2219.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 2219.50 | 2208.63 | 2216.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:45:00 | 2222.40 | 2208.63 | 2216.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 2224.00 | 2211.70 | 2216.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 2224.00 | 2211.70 | 2216.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 2246.00 | 2218.56 | 2219.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 2235.10 | 2218.56 | 2219.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 2249.10 | 2224.67 | 2222.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 2256.80 | 2231.10 | 2225.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 15:15:00 | 2212.20 | 2227.32 | 2224.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 2212.20 | 2227.32 | 2224.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 2212.20 | 2227.32 | 2224.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 2203.50 | 2227.32 | 2224.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2219.90 | 2225.83 | 2223.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 2212.10 | 2225.83 | 2223.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2212.00 | 2223.07 | 2222.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 2212.00 | 2223.07 | 2222.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 11:15:00 | 2210.30 | 2220.51 | 2221.56 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 2234.00 | 2221.79 | 2221.52 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 2212.40 | 2219.91 | 2220.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 2198.00 | 2213.66 | 2217.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 2182.50 | 2180.61 | 2192.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:45:00 | 2182.10 | 2180.61 | 2192.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2195.60 | 2183.60 | 2192.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 2196.10 | 2183.60 | 2192.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 2190.60 | 2185.00 | 2192.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:00:00 | 2190.60 | 2185.00 | 2192.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 2191.80 | 2186.36 | 2192.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 2191.80 | 2186.36 | 2192.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 2193.70 | 2187.83 | 2192.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 2196.70 | 2187.83 | 2192.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2201.60 | 2190.58 | 2193.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 2203.70 | 2190.58 | 2193.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 2177.00 | 2187.87 | 2191.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 2174.40 | 2187.87 | 2191.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 2174.00 | 2185.99 | 2190.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 2159.30 | 2150.07 | 2149.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 2159.30 | 2150.07 | 2149.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 2162.60 | 2154.50 | 2151.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 15:15:00 | 2150.00 | 2153.60 | 2151.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 15:15:00 | 2150.00 | 2153.60 | 2151.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 2150.00 | 2153.60 | 2151.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 2138.70 | 2153.60 | 2151.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 2135.50 | 2149.98 | 2150.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 12:15:00 | 2119.30 | 2138.99 | 2144.75 | Break + close below crossover candle low |

### Cycle 121 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 2282.80 | 2158.90 | 2150.87 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 2167.70 | 2196.22 | 2199.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 2157.50 | 2188.48 | 2195.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 2223.20 | 2175.51 | 2184.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 2223.20 | 2175.51 | 2184.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 2223.20 | 2175.51 | 2184.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 2233.00 | 2175.51 | 2184.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 2219.30 | 2184.27 | 2187.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:00:00 | 2219.30 | 2184.27 | 2187.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 11:15:00 | 2233.80 | 2194.17 | 2191.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 14:15:00 | 2256.70 | 2217.88 | 2204.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 11:15:00 | 2219.30 | 2229.84 | 2215.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-22 12:00:00 | 2219.30 | 2229.84 | 2215.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 2195.50 | 2222.97 | 2213.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:00:00 | 2195.50 | 2222.97 | 2213.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 2198.90 | 2218.16 | 2212.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 2187.10 | 2218.16 | 2212.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 2200.00 | 2213.49 | 2211.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 2218.60 | 2213.49 | 2211.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2235.40 | 2217.87 | 2213.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 10:15:00 | 2244.00 | 2217.87 | 2213.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 2196.10 | 2209.79 | 2210.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2196.10 | 2209.79 | 2210.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 2190.20 | 2205.87 | 2209.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 2187.80 | 2185.69 | 2194.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 2187.80 | 2185.69 | 2194.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 2187.80 | 2185.69 | 2194.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 2173.30 | 2185.69 | 2194.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 2176.10 | 2183.77 | 2192.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 2168.60 | 2180.72 | 2186.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 2170.00 | 2178.90 | 2185.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:00:00 | 2168.10 | 2176.74 | 2183.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:30:00 | 2169.80 | 2176.31 | 2182.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 2178.50 | 2176.25 | 2181.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:30:00 | 2179.90 | 2176.25 | 2181.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 2175.30 | 2175.84 | 2179.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 2178.30 | 2175.84 | 2179.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 2179.30 | 2176.53 | 2179.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 2179.30 | 2176.53 | 2179.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 2190.00 | 2179.22 | 2180.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 2190.00 | 2179.22 | 2180.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 2200.90 | 2183.56 | 2182.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 2200.90 | 2183.56 | 2182.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 2230.50 | 2197.75 | 2189.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 2201.20 | 2208.88 | 2197.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 2201.20 | 2208.88 | 2197.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2201.20 | 2208.88 | 2197.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2216.80 | 2208.88 | 2197.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2175.00 | 2200.27 | 2197.27 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 2174.70 | 2195.15 | 2195.22 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 2206.50 | 2197.43 | 2196.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 2251.80 | 2208.88 | 2201.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 2251.60 | 2251.62 | 2231.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 2251.60 | 2251.62 | 2231.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2251.60 | 2251.62 | 2231.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 2269.10 | 2256.79 | 2239.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 2270.20 | 2263.62 | 2247.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:45:00 | 2271.00 | 2264.00 | 2251.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 2212.00 | 2247.07 | 2250.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 2212.00 | 2247.07 | 2250.91 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 2271.90 | 2253.01 | 2250.57 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 2226.40 | 2248.44 | 2249.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 2202.00 | 2234.17 | 2241.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 2209.80 | 2203.30 | 2211.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 2209.80 | 2203.30 | 2211.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 2209.80 | 2203.30 | 2211.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 2204.50 | 2203.30 | 2211.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 2203.80 | 2203.40 | 2210.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 2203.30 | 2203.40 | 2210.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 2196.40 | 2202.00 | 2209.26 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 2225.10 | 2211.43 | 2210.61 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 2189.70 | 2210.33 | 2211.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 12:15:00 | 2167.00 | 2198.33 | 2205.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 2179.40 | 2168.21 | 2177.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 2179.40 | 2168.21 | 2177.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 2179.40 | 2168.21 | 2177.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 2179.40 | 2168.21 | 2177.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 2175.20 | 2169.61 | 2177.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 2184.00 | 2169.61 | 2177.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 2188.70 | 2173.43 | 2178.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 2188.70 | 2173.43 | 2178.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 2187.20 | 2176.18 | 2179.15 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 2200.40 | 2182.60 | 2181.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 2210.10 | 2191.84 | 2186.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 10:15:00 | 2202.10 | 2204.47 | 2196.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 11:00:00 | 2202.10 | 2204.47 | 2196.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 2202.70 | 2204.11 | 2197.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:45:00 | 2197.10 | 2204.11 | 2197.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 2211.60 | 2205.61 | 2198.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:30:00 | 2200.40 | 2205.61 | 2198.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 2200.80 | 2210.50 | 2204.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 2200.10 | 2210.50 | 2204.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 2198.20 | 2208.04 | 2204.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:30:00 | 2202.10 | 2208.04 | 2204.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 2200.00 | 2206.43 | 2203.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 2200.00 | 2206.43 | 2203.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 2191.00 | 2203.35 | 2202.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 2191.00 | 2203.35 | 2202.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 2196.00 | 2201.88 | 2201.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 09:15:00 | 2180.90 | 2197.68 | 2200.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 2086.60 | 2085.93 | 2110.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 15:00:00 | 2086.60 | 2085.93 | 2110.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 2087.00 | 2074.33 | 2086.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2085.60 | 2074.33 | 2086.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2096.00 | 2078.66 | 2087.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 2096.00 | 2078.66 | 2087.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 2099.40 | 2082.81 | 2088.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:45:00 | 2100.00 | 2082.81 | 2088.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 2122.20 | 2096.74 | 2094.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 12:15:00 | 2159.60 | 2119.75 | 2107.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 2167.80 | 2170.28 | 2153.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 2167.80 | 2170.28 | 2153.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 2148.00 | 2165.83 | 2152.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 2147.00 | 2165.83 | 2152.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 2136.10 | 2159.88 | 2151.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 2136.10 | 2159.88 | 2151.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 2123.00 | 2144.34 | 2145.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2100.00 | 2135.48 | 2141.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 2111.40 | 2103.81 | 2109.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 2111.40 | 2103.81 | 2109.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 2111.40 | 2103.81 | 2109.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 2111.40 | 2103.81 | 2109.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 2101.00 | 2103.25 | 2108.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 10:15:00 | 2096.00 | 2102.84 | 2108.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 15:00:00 | 2094.90 | 2088.60 | 2091.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 2088.00 | 2090.88 | 2092.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 15:15:00 | 1991.20 | 2009.10 | 2034.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 15:15:00 | 1990.15 | 2009.10 | 2034.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 15:15:00 | 1983.60 | 2009.10 | 2034.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1971.00 | 1969.42 | 1991.87 | SL hit (close>ema200) qty=0.50 sl=1969.42 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 2032.00 | 1999.90 | 1996.53 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1970.10 | 1997.39 | 2000.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 1960.00 | 1989.91 | 1996.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 13:15:00 | 1967.70 | 1962.34 | 1978.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 14:00:00 | 1967.70 | 1962.34 | 1978.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 1996.90 | 1969.25 | 1979.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 1996.90 | 1969.25 | 1979.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 1961.20 | 1967.64 | 1978.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 2027.90 | 1967.64 | 1978.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 2064.30 | 1986.97 | 1986.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 2067.70 | 2036.78 | 2023.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 2082.10 | 2097.31 | 2084.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 15:15:00 | 2082.10 | 2097.31 | 2084.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 2082.10 | 2097.31 | 2084.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 2108.50 | 2097.85 | 2086.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 2105.00 | 2097.85 | 2086.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:45:00 | 2105.90 | 2100.22 | 2089.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:45:00 | 2106.20 | 2101.24 | 2090.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2088.90 | 2098.68 | 2093.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2099.90 | 2098.68 | 2093.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 2157.90 | 2165.14 | 2165.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 2157.90 | 2165.14 | 2165.37 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 2175.00 | 2167.11 | 2166.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 2184.50 | 2170.59 | 2167.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 13:15:00 | 2166.00 | 2169.67 | 2167.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 13:15:00 | 2166.00 | 2169.67 | 2167.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 2166.00 | 2169.67 | 2167.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 2166.00 | 2169.67 | 2167.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 2161.10 | 2167.96 | 2167.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:30:00 | 2158.30 | 2167.96 | 2167.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 2160.00 | 2166.37 | 2166.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 2140.50 | 2161.19 | 2164.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 2138.80 | 2123.10 | 2134.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 2138.80 | 2123.10 | 2134.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 2138.80 | 2123.10 | 2134.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 2138.80 | 2123.10 | 2134.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 2119.50 | 2122.38 | 2133.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:30:00 | 2106.30 | 2121.78 | 2130.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 2113.70 | 2120.91 | 2129.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:45:00 | 2114.40 | 2121.28 | 2127.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:15:00 | 2113.00 | 2121.54 | 2126.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 2113.00 | 2119.84 | 2125.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 2146.70 | 2126.03 | 2126.74 | SL hit (close>static) qty=1.00 sl=2141.30 alert=retest2 |

### Cycle 143 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 2135.80 | 2127.98 | 2127.56 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 2124.90 | 2127.91 | 2127.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 13:15:00 | 2121.60 | 2126.20 | 2127.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 2097.80 | 2087.56 | 2102.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 14:15:00 | 2097.80 | 2087.56 | 2102.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 2097.80 | 2087.56 | 2102.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 2097.80 | 2087.56 | 2102.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 2098.00 | 2089.64 | 2101.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 2078.30 | 2089.64 | 2101.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:00:00 | 2089.20 | 2064.48 | 2068.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 12:00:00 | 2078.80 | 2067.34 | 2069.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 12:15:00 | 2101.60 | 2074.20 | 2072.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 2101.60 | 2074.20 | 2072.36 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-04 09:15:00 | 2384.30 | 2024-06-04 12:15:00 | 2265.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:30:00 | 2411.25 | 2024-06-04 12:15:00 | 2290.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 2384.30 | 2024-06-04 15:15:00 | 2387.30 | STOP_HIT | 0.50 | -0.13% |
| SELL | retest2 | 2024-06-04 10:30:00 | 2411.25 | 2024-06-04 15:15:00 | 2387.30 | STOP_HIT | 0.50 | 0.99% |
| BUY | retest2 | 2024-06-12 09:30:00 | 2577.45 | 2024-06-14 11:15:00 | 2567.95 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-06-18 14:15:00 | 2560.00 | 2024-06-19 12:15:00 | 2611.80 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-06-18 15:00:00 | 2556.80 | 2024-06-19 12:15:00 | 2611.80 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-06-19 09:15:00 | 2557.00 | 2024-06-19 12:15:00 | 2611.80 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-07-23 09:30:00 | 2823.75 | 2024-07-23 12:15:00 | 2875.70 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-07-23 11:15:00 | 2828.40 | 2024-07-23 12:15:00 | 2875.70 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-07-23 12:15:00 | 2761.05 | 2024-07-23 12:15:00 | 2875.70 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2024-08-26 09:15:00 | 2965.15 | 2024-09-02 09:15:00 | 2983.75 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2024-09-30 09:30:00 | 2905.05 | 2024-09-30 10:15:00 | 2940.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-10-14 14:30:00 | 3012.20 | 2024-10-18 13:15:00 | 3025.55 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-10-14 15:00:00 | 3021.90 | 2024-10-18 13:15:00 | 3025.55 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2024-10-16 09:30:00 | 3036.30 | 2024-10-18 13:15:00 | 3025.55 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-10-17 10:00:00 | 3019.55 | 2024-10-18 13:15:00 | 3025.55 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2024-10-17 12:00:00 | 3045.90 | 2024-10-18 13:15:00 | 3025.55 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-10-17 13:00:00 | 3048.15 | 2024-10-18 13:15:00 | 3025.55 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-10-18 10:15:00 | 3055.10 | 2024-10-18 13:15:00 | 3025.55 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-11-07 09:15:00 | 2438.90 | 2024-11-12 14:15:00 | 2316.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 09:15:00 | 2438.90 | 2024-11-14 09:15:00 | 2315.60 | STOP_HIT | 0.50 | 5.06% |
| BUY | retest2 | 2024-12-05 12:15:00 | 2380.00 | 2024-12-06 15:15:00 | 2360.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-12-27 10:45:00 | 2232.00 | 2024-12-27 12:15:00 | 2258.95 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-01-10 11:45:00 | 2319.40 | 2025-01-13 10:15:00 | 2277.15 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-01-10 13:30:00 | 2322.50 | 2025-01-13 10:15:00 | 2277.15 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-01-10 14:15:00 | 2324.30 | 2025-01-13 10:15:00 | 2277.15 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-01-10 15:00:00 | 2324.60 | 2025-01-13 10:15:00 | 2277.15 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-01-14 12:45:00 | 2244.75 | 2025-01-17 11:15:00 | 2260.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-01-14 14:30:00 | 2244.55 | 2025-01-17 11:15:00 | 2260.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-01-15 09:15:00 | 2227.80 | 2025-01-17 11:15:00 | 2260.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-01-16 11:45:00 | 2235.90 | 2025-01-17 11:15:00 | 2260.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-01-28 14:45:00 | 2081.50 | 2025-02-01 13:15:00 | 2144.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-01-29 12:15:00 | 2082.30 | 2025-02-01 13:15:00 | 2144.00 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-01-29 13:45:00 | 2081.45 | 2025-02-01 13:15:00 | 2144.00 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-01-30 10:45:00 | 2077.00 | 2025-02-01 13:15:00 | 2144.00 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-01-31 10:30:00 | 2063.25 | 2025-02-01 13:15:00 | 2144.00 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-01-31 11:30:00 | 2068.95 | 2025-02-01 13:15:00 | 2144.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2025-02-04 12:15:00 | 2176.60 | 2025-02-07 12:15:00 | 2165.35 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-02-04 14:30:00 | 2188.75 | 2025-02-07 12:15:00 | 2165.35 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-02-05 09:30:00 | 2187.90 | 2025-02-07 12:15:00 | 2165.35 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-02-06 15:00:00 | 2183.00 | 2025-02-07 12:15:00 | 2165.35 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-02-21 10:15:00 | 2033.45 | 2025-02-27 15:15:00 | 1931.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 11:45:00 | 2031.70 | 2025-02-27 15:15:00 | 1930.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 11:45:00 | 2026.45 | 2025-02-27 15:15:00 | 1925.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 12:45:00 | 2034.55 | 2025-02-27 15:15:00 | 1932.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 10:15:00 | 2033.45 | 2025-02-28 14:15:00 | 1956.05 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2025-02-21 11:45:00 | 2031.70 | 2025-02-28 14:15:00 | 1956.05 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2025-02-24 11:45:00 | 2026.45 | 2025-02-28 14:15:00 | 1956.05 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2025-02-24 12:45:00 | 2034.55 | 2025-02-28 14:15:00 | 1956.05 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2025-02-27 10:45:00 | 1974.75 | 2025-03-05 12:15:00 | 1970.40 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-03-10 12:00:00 | 2011.05 | 2025-03-11 09:15:00 | 1953.05 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-03-10 14:15:00 | 2010.70 | 2025-03-11 09:15:00 | 1953.05 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-03-10 15:00:00 | 2012.00 | 2025-03-11 09:15:00 | 1953.05 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-03-28 10:15:00 | 2021.20 | 2025-04-02 10:15:00 | 2077.55 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1967.80 | 2025-04-11 09:15:00 | 2037.75 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-04-09 10:00:00 | 1963.90 | 2025-04-11 09:15:00 | 2037.75 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest1 | 2025-04-16 11:45:00 | 2142.80 | 2025-04-21 09:15:00 | 2249.94 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-16 11:45:00 | 2142.80 | 2025-04-23 09:15:00 | 2258.80 | STOP_HIT | 0.50 | 5.41% |
| BUY | retest2 | 2025-04-24 15:00:00 | 2308.50 | 2025-04-25 13:15:00 | 2256.20 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-05-16 09:15:00 | 2360.10 | 2025-05-23 14:15:00 | 2367.70 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-05-16 10:15:00 | 2346.80 | 2025-05-23 14:15:00 | 2367.70 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-06-09 14:00:00 | 2430.00 | 2025-06-13 13:15:00 | 2450.00 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-06-18 09:45:00 | 2487.30 | 2025-06-18 11:15:00 | 2470.90 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-04 11:45:00 | 2536.70 | 2025-07-09 09:15:00 | 2586.20 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-07-07 09:45:00 | 2538.00 | 2025-07-09 09:15:00 | 2586.20 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-07-07 15:15:00 | 2536.40 | 2025-07-09 09:15:00 | 2586.20 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2539.50 | 2025-07-09 09:15:00 | 2586.20 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-10 11:30:00 | 2580.80 | 2025-07-17 09:15:00 | 2650.90 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest2 | 2025-07-10 12:00:00 | 2580.30 | 2025-07-17 09:15:00 | 2650.90 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2025-07-11 09:15:00 | 2593.00 | 2025-07-17 09:15:00 | 2650.90 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-07-21 10:45:00 | 2603.00 | 2025-07-24 09:15:00 | 2626.90 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-08-12 10:00:00 | 2562.90 | 2025-08-12 13:15:00 | 2530.10 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-08-13 14:30:00 | 2549.20 | 2025-08-18 09:15:00 | 2534.90 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-14 11:00:00 | 2548.80 | 2025-08-18 09:15:00 | 2534.90 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-08-14 12:30:00 | 2550.10 | 2025-08-18 09:15:00 | 2534.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-08-14 14:00:00 | 2548.80 | 2025-08-18 09:15:00 | 2534.90 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-08-20 09:15:00 | 2607.10 | 2025-08-25 11:15:00 | 2648.50 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-08-29 13:15:00 | 2588.00 | 2025-09-01 09:15:00 | 2618.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-08-29 15:15:00 | 2590.00 | 2025-09-01 09:15:00 | 2618.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-02 09:15:00 | 2565.20 | 2025-09-08 09:15:00 | 2578.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-09-15 15:15:00 | 2626.00 | 2025-09-16 15:15:00 | 2594.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-09-29 12:00:00 | 2344.60 | 2025-10-01 13:15:00 | 2380.10 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-29 15:00:00 | 2327.80 | 2025-10-01 13:15:00 | 2380.10 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-10-01 09:15:00 | 2336.50 | 2025-10-01 13:15:00 | 2380.10 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-10-01 10:15:00 | 2345.60 | 2025-10-01 13:15:00 | 2380.10 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-06 13:45:00 | 2385.00 | 2025-10-07 14:15:00 | 2359.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-07 09:15:00 | 2392.70 | 2025-10-07 14:15:00 | 2359.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-10-07 11:45:00 | 2380.00 | 2025-10-07 14:15:00 | 2359.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-07 13:00:00 | 2379.60 | 2025-10-07 14:15:00 | 2359.60 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-16 14:45:00 | 2355.50 | 2025-10-21 13:15:00 | 2375.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-17 09:15:00 | 2343.20 | 2025-10-21 13:15:00 | 2375.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-20 14:15:00 | 2355.10 | 2025-10-21 13:15:00 | 2375.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-26 09:15:00 | 2335.20 | 2025-12-03 11:15:00 | 2336.60 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-12-08 09:15:00 | 2290.00 | 2025-12-15 11:15:00 | 2272.10 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2026-01-06 09:15:00 | 2174.40 | 2026-01-13 10:15:00 | 2159.30 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-01-06 10:15:00 | 2174.00 | 2026-01-13 10:15:00 | 2159.30 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2026-01-23 10:15:00 | 2244.00 | 2026-01-23 13:15:00 | 2196.10 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-01-29 10:15:00 | 2168.60 | 2026-01-30 13:15:00 | 2200.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-29 11:15:00 | 2170.00 | 2026-01-30 13:15:00 | 2200.90 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-01-29 12:00:00 | 2168.10 | 2026-01-30 13:15:00 | 2200.90 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-01-29 12:30:00 | 2169.80 | 2026-01-30 13:15:00 | 2200.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-02-04 13:15:00 | 2269.10 | 2026-02-06 12:15:00 | 2212.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-02-05 09:15:00 | 2270.20 | 2026-02-06 12:15:00 | 2212.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-02-05 11:45:00 | 2271.00 | 2026-02-06 12:15:00 | 2212.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-03-17 10:15:00 | 2096.00 | 2026-03-20 15:15:00 | 1991.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 15:00:00 | 2094.90 | 2026-03-20 15:15:00 | 1990.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 2088.00 | 2026-03-20 15:15:00 | 1983.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 10:15:00 | 2096.00 | 2026-03-24 11:15:00 | 1971.00 | STOP_HIT | 0.50 | 5.96% |
| SELL | retest2 | 2026-03-18 15:00:00 | 2094.90 | 2026-03-24 11:15:00 | 1971.00 | STOP_HIT | 0.50 | 5.91% |
| SELL | retest2 | 2026-03-19 09:15:00 | 2088.00 | 2026-03-24 11:15:00 | 1971.00 | STOP_HIT | 0.50 | 5.60% |
| BUY | retest2 | 2026-04-10 09:30:00 | 2108.50 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2026-04-10 10:15:00 | 2105.00 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.51% |
| BUY | retest2 | 2026-04-10 11:45:00 | 2105.90 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2026-04-10 12:45:00 | 2106.20 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2099.90 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.76% |
| SELL | retest2 | 2026-04-28 09:30:00 | 2106.30 | 2026-04-29 12:15:00 | 2146.70 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-28 10:30:00 | 2113.70 | 2026-04-29 12:15:00 | 2146.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-04-28 13:45:00 | 2114.40 | 2026-04-29 12:15:00 | 2146.70 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-04-28 15:15:00 | 2113.00 | 2026-04-29 12:15:00 | 2146.70 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-05-05 09:15:00 | 2078.30 | 2026-05-07 12:15:00 | 2101.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-05-07 11:00:00 | 2089.20 | 2026-05-07 12:15:00 | 2101.60 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-05-07 12:00:00 | 2078.80 | 2026-05-07 12:15:00 | 2101.60 | STOP_HIT | 1.00 | -1.10% |
