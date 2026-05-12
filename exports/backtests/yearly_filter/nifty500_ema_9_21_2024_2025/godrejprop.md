# Godrej Properties Ltd. (GODREJPROP)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1874.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 143 |
| ALERT1 | 109 |
| ALERT2 | 110 |
| ALERT2_SKIP | 59 |
| ALERT3 | 299 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 112 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 108 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 123 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 86
- **Target hits / Stop hits / Partials:** 5 / 108 / 10
- **Avg / median % per leg:** 0.48% / -0.95%
- **Sum % (uncompounded):** 59.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 15 | 36.6% | 2 | 39 | 0 | 0.66% | 26.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.12% | -2.1% |
| BUY @ 3rd Alert (retest2) | 40 | 15 | 37.5% | 2 | 38 | 0 | 0.72% | 29.0% |
| SELL (all) | 82 | 22 | 26.8% | 3 | 69 | 10 | 0.39% | 32.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 82 | 22 | 26.8% | 3 | 69 | 10 | 0.39% | 32.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.12% | -2.1% |
| retest2 (combined) | 122 | 37 | 30.3% | 5 | 107 | 10 | 0.50% | 61.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 2805.50 | 2740.83 | 2736.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 2822.80 | 2757.23 | 2744.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 2792.00 | 2806.73 | 2785.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 12:15:00 | 2792.00 | 2806.73 | 2785.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 2792.00 | 2806.73 | 2785.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:45:00 | 2786.70 | 2806.73 | 2785.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 2822.05 | 2809.80 | 2789.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:45:00 | 2809.50 | 2809.80 | 2789.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 2859.10 | 2848.93 | 2830.79 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 2797.25 | 2833.24 | 2834.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 12:15:00 | 2783.45 | 2816.37 | 2826.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 2805.60 | 2803.69 | 2816.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 2805.60 | 2803.69 | 2816.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 2805.60 | 2803.69 | 2816.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:15:00 | 2777.25 | 2808.04 | 2813.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 09:15:00 | 2762.55 | 2803.19 | 2806.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 10:15:00 | 2849.95 | 2783.98 | 2785.42 | SL hit (close>static) qty=1.00 sl=2829.40 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 11:15:00 | 2844.75 | 2796.13 | 2790.81 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 2793.05 | 2801.22 | 2802.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 2780.00 | 2796.97 | 2800.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 14:15:00 | 2761.75 | 2758.26 | 2774.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 15:00:00 | 2761.75 | 2758.26 | 2774.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 2733.10 | 2753.51 | 2769.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:15:00 | 2725.55 | 2743.98 | 2760.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 14:00:00 | 2727.80 | 2740.75 | 2757.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 14:15:00 | 2783.75 | 2763.01 | 2760.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 2783.75 | 2763.01 | 2760.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 2895.50 | 2792.06 | 2774.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 2878.90 | 2912.86 | 2859.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 2878.90 | 2912.86 | 2859.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 2878.90 | 2912.86 | 2859.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 2834.25 | 2912.86 | 2859.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 2723.95 | 2875.07 | 2847.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 2723.95 | 2875.07 | 2847.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 2668.30 | 2833.72 | 2831.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 2668.30 | 2833.72 | 2831.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 12:15:00 | 2646.95 | 2796.37 | 2814.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 2522.10 | 2667.60 | 2741.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 2647.75 | 2639.63 | 2702.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 2647.75 | 2639.63 | 2702.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 2790.00 | 2677.94 | 2704.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 2792.50 | 2677.94 | 2704.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 2825.00 | 2707.35 | 2715.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 2825.00 | 2707.35 | 2715.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 2753.45 | 2726.89 | 2723.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 2776.55 | 2736.82 | 2728.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 2834.45 | 2844.52 | 2807.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 11:00:00 | 2834.45 | 2844.52 | 2807.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 2848.15 | 2844.20 | 2819.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 2822.65 | 2844.20 | 2819.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 2877.85 | 2868.14 | 2849.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 11:15:00 | 2896.90 | 2870.29 | 2852.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 15:15:00 | 2999.60 | 3011.64 | 3013.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 2999.60 | 3011.64 | 3013.13 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 3054.70 | 3019.93 | 3016.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 11:15:00 | 3059.15 | 3027.78 | 3020.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 09:15:00 | 3058.45 | 3064.27 | 3043.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 3058.45 | 3064.27 | 3043.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 3058.45 | 3064.27 | 3043.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 3058.45 | 3064.27 | 3043.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 3070.00 | 3062.45 | 3046.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 3047.45 | 3062.45 | 3046.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 3085.80 | 3072.35 | 3057.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:45:00 | 3053.50 | 3072.35 | 3057.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 3062.25 | 3069.53 | 3058.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 11:45:00 | 3085.45 | 3068.59 | 3062.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 13:30:00 | 3071.10 | 3072.78 | 3065.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 11:15:00 | 3241.25 | 3280.27 | 3282.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 11:15:00 | 3241.25 | 3280.27 | 3282.34 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 12:15:00 | 3291.20 | 3281.62 | 3280.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 14:15:00 | 3319.20 | 3290.36 | 3284.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 09:15:00 | 3270.35 | 3288.75 | 3285.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 3270.35 | 3288.75 | 3285.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 3270.35 | 3288.75 | 3285.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 3270.35 | 3288.75 | 3285.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 10:15:00 | 3256.55 | 3282.31 | 3282.73 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 3300.00 | 3279.71 | 3278.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 14:15:00 | 3316.45 | 3289.01 | 3283.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 11:15:00 | 3289.50 | 3296.18 | 3289.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 11:15:00 | 3289.50 | 3296.18 | 3289.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 3289.50 | 3296.18 | 3289.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 3289.50 | 3296.18 | 3289.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 3277.00 | 3292.35 | 3287.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:45:00 | 3273.45 | 3292.35 | 3287.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 3277.00 | 3289.28 | 3286.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:30:00 | 3270.40 | 3289.28 | 3286.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 3300.00 | 3292.15 | 3288.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 3304.05 | 3292.15 | 3288.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 3270.00 | 3287.72 | 3286.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 3270.00 | 3287.72 | 3286.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 3318.60 | 3293.90 | 3289.86 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 3260.25 | 3286.07 | 3286.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 3234.95 | 3267.04 | 3276.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 11:15:00 | 3271.95 | 3264.33 | 3273.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 12:00:00 | 3271.95 | 3264.33 | 3273.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 3295.00 | 3270.47 | 3275.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:00:00 | 3295.00 | 3270.47 | 3275.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 3307.60 | 3277.89 | 3278.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:45:00 | 3306.75 | 3277.89 | 3278.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 14:15:00 | 3300.35 | 3282.39 | 3280.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 3348.40 | 3298.09 | 3288.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 3355.45 | 3359.54 | 3331.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 3355.45 | 3359.54 | 3331.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 3355.45 | 3359.54 | 3331.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 3329.00 | 3359.54 | 3331.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 3346.10 | 3356.86 | 3332.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:30:00 | 3332.20 | 3356.86 | 3332.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 3303.50 | 3346.30 | 3338.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 3303.50 | 3346.30 | 3338.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 3269.45 | 3330.93 | 3332.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 3242.80 | 3291.45 | 3311.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 3286.40 | 3277.08 | 3298.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:00:00 | 3286.40 | 3277.08 | 3298.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 3299.65 | 3281.59 | 3298.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 3297.55 | 3281.59 | 3298.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 3278.55 | 3280.98 | 3296.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:00:00 | 3241.65 | 3272.97 | 3290.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 3252.00 | 3268.86 | 3285.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 3133.50 | 3272.39 | 3284.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 13:15:00 | 3079.57 | 3234.44 | 3264.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 13:15:00 | 3089.40 | 3234.44 | 3264.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 3127.90 | 3114.48 | 3146.01 | SL hit (close>ema200) qty=0.50 sl=3114.48 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 13:15:00 | 3148.00 | 3144.36 | 3144.12 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 14:15:00 | 3131.25 | 3141.74 | 3142.95 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 10:15:00 | 3198.45 | 3149.68 | 3145.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 11:15:00 | 3262.75 | 3191.00 | 3171.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 3200.20 | 3209.95 | 3190.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 3200.20 | 3209.95 | 3190.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 3200.20 | 3209.95 | 3190.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 3200.20 | 3209.95 | 3190.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 3175.50 | 3203.06 | 3189.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 3175.50 | 3203.06 | 3189.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 3147.90 | 3192.03 | 3185.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 3147.90 | 3192.03 | 3185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 3140.00 | 3181.62 | 3181.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:15:00 | 3126.00 | 3181.62 | 3181.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 3145.20 | 3174.34 | 3178.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 3048.85 | 3135.54 | 3158.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2977.00 | 2894.11 | 2967.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 2977.00 | 2894.11 | 2967.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 2977.00 | 2894.11 | 2967.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 2977.00 | 2894.11 | 2967.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 2927.20 | 2900.72 | 2964.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:15:00 | 2911.45 | 2942.69 | 2953.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 2918.00 | 2914.66 | 2934.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 2915.05 | 2914.74 | 2932.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 2908.45 | 2927.82 | 2933.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 2925.30 | 2911.64 | 2922.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:00:00 | 2925.30 | 2911.64 | 2922.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 2934.00 | 2916.12 | 2923.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:30:00 | 2938.40 | 2916.12 | 2923.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 2933.80 | 2919.65 | 2924.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:45:00 | 2948.75 | 2919.65 | 2924.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 2944.15 | 2929.92 | 2928.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 2944.15 | 2929.92 | 2928.28 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 2914.45 | 2926.97 | 2927.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 2896.00 | 2920.77 | 2924.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 2915.05 | 2891.81 | 2902.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 2915.05 | 2891.81 | 2902.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 2915.05 | 2891.81 | 2902.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 2929.55 | 2891.81 | 2902.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 2924.70 | 2898.38 | 2904.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:30:00 | 2929.05 | 2898.38 | 2904.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 2928.00 | 2904.31 | 2906.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:15:00 | 2930.00 | 2904.31 | 2906.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 2931.50 | 2909.75 | 2908.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 2957.00 | 2925.53 | 2919.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 2944.50 | 2956.97 | 2941.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 2944.50 | 2956.97 | 2941.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 2944.50 | 2956.97 | 2941.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 2941.85 | 2956.97 | 2941.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 2928.85 | 2951.34 | 2940.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 2928.85 | 2951.34 | 2940.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 2919.60 | 2945.00 | 2938.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:30:00 | 2921.30 | 2945.00 | 2938.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 14:15:00 | 2922.35 | 2933.13 | 2934.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 09:15:00 | 2911.10 | 2927.63 | 2930.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 12:15:00 | 2891.95 | 2889.46 | 2902.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 13:00:00 | 2891.95 | 2889.46 | 2902.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 2893.60 | 2890.29 | 2901.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:00:00 | 2893.60 | 2890.29 | 2901.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 2900.00 | 2890.94 | 2899.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 2900.00 | 2890.94 | 2899.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 2913.60 | 2895.47 | 2900.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:00:00 | 2913.60 | 2895.47 | 2900.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 2906.05 | 2897.59 | 2901.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 12:45:00 | 2896.95 | 2898.21 | 2901.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:45:00 | 2896.05 | 2898.58 | 2900.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:30:00 | 2885.00 | 2896.05 | 2899.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:15:00 | 2897.65 | 2897.00 | 2899.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 2891.00 | 2895.80 | 2898.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:00:00 | 2882.55 | 2893.15 | 2897.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 14:45:00 | 2883.70 | 2890.55 | 2895.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 15:15:00 | 2884.00 | 2890.55 | 2895.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 09:30:00 | 2879.85 | 2888.35 | 2893.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 2871.00 | 2869.41 | 2879.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 2898.90 | 2869.41 | 2879.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 2883.55 | 2872.24 | 2879.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:15:00 | 2877.70 | 2872.24 | 2879.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 14:15:00 | 2912.30 | 2886.74 | 2884.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 2912.30 | 2886.74 | 2884.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 15:15:00 | 2917.80 | 2892.96 | 2887.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 2888.90 | 2897.24 | 2891.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 2888.90 | 2897.24 | 2891.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 2888.90 | 2897.24 | 2891.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 2888.90 | 2897.24 | 2891.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 2893.45 | 2896.49 | 2891.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:30:00 | 2890.85 | 2896.49 | 2891.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 2902.70 | 2897.73 | 2892.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:30:00 | 2894.95 | 2897.73 | 2892.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 2888.75 | 2895.93 | 2892.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 2888.75 | 2895.93 | 2892.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 2887.90 | 2894.33 | 2891.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 2922.55 | 2894.33 | 2891.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:15:00 | 2890.70 | 2893.52 | 2892.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:00:00 | 2895.65 | 2896.30 | 2894.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:00:00 | 2894.30 | 2896.40 | 2895.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 2888.95 | 2894.91 | 2894.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:30:00 | 2882.55 | 2894.91 | 2894.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 2907.10 | 2897.34 | 2895.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 2922.55 | 2899.48 | 2896.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 2887.00 | 2897.54 | 2896.37 | SL hit (close<static) qty=1.00 sl=2887.35 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 11:15:00 | 2883.95 | 2894.83 | 2895.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 2870.40 | 2886.26 | 2890.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 2876.50 | 2874.50 | 2882.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 11:45:00 | 2877.35 | 2874.50 | 2882.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 2866.50 | 2872.90 | 2881.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:30:00 | 2890.75 | 2872.90 | 2881.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 2880.25 | 2869.47 | 2876.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:45:00 | 2889.60 | 2869.47 | 2876.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 2882.65 | 2872.10 | 2877.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:45:00 | 2878.75 | 2872.10 | 2877.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 2867.95 | 2871.27 | 2876.44 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 2896.50 | 2879.05 | 2877.57 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 2829.65 | 2871.66 | 2876.40 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 2967.90 | 2887.60 | 2878.45 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 2855.55 | 2899.92 | 2902.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 2837.80 | 2887.49 | 2896.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 14:15:00 | 2859.40 | 2859.32 | 2874.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 14:30:00 | 2855.45 | 2859.32 | 2874.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 2867.50 | 2861.28 | 2872.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:15:00 | 2833.00 | 2869.22 | 2871.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:45:00 | 2848.00 | 2857.70 | 2865.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 14:15:00 | 2913.00 | 2871.03 | 2870.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 2913.00 | 2871.03 | 2870.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 2919.15 | 2897.66 | 2884.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 11:15:00 | 3268.60 | 3291.30 | 3227.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 12:00:00 | 3268.60 | 3291.30 | 3227.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 3246.15 | 3283.96 | 3248.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 3237.00 | 3283.96 | 3248.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 3225.40 | 3272.25 | 3246.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 3225.40 | 3272.25 | 3246.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 3220.90 | 3261.98 | 3244.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:45:00 | 3211.90 | 3261.98 | 3244.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 3190.00 | 3229.28 | 3232.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 3140.60 | 3206.06 | 3220.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 15:15:00 | 3165.15 | 3164.65 | 3188.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:15:00 | 3190.30 | 3164.65 | 3188.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 3205.45 | 3172.81 | 3190.27 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 14:15:00 | 3233.00 | 3199.79 | 3197.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 15:15:00 | 3260.00 | 3211.83 | 3203.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 3191.80 | 3207.82 | 3202.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 3191.80 | 3207.82 | 3202.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 3191.80 | 3207.82 | 3202.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:15:00 | 3181.60 | 3207.82 | 3202.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 3180.45 | 3202.35 | 3200.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:45:00 | 3191.35 | 3202.35 | 3200.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 3135.25 | 3188.93 | 3194.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 3101.25 | 3171.39 | 3185.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2890.90 | 2889.53 | 2957.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 2899.10 | 2889.53 | 2957.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 2962.50 | 2916.86 | 2949.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 2962.50 | 2916.86 | 2949.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 2956.40 | 2924.77 | 2950.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 2932.70 | 2944.04 | 2952.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 13:15:00 | 2996.50 | 2957.49 | 2957.58 | SL hit (close>static) qty=1.00 sl=2962.50 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 14:15:00 | 2979.00 | 2961.79 | 2959.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 3025.35 | 2979.34 | 2968.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 11:15:00 | 2981.60 | 3002.77 | 2991.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 11:15:00 | 2981.60 | 3002.77 | 2991.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 2981.60 | 3002.77 | 2991.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:00:00 | 2981.60 | 3002.77 | 2991.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 3003.60 | 3002.94 | 2992.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:30:00 | 3012.20 | 3004.52 | 2995.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 15:15:00 | 3080.00 | 3118.65 | 3119.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 3080.00 | 3118.65 | 3119.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 3039.30 | 3102.78 | 3112.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 3119.55 | 3105.05 | 3111.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 3119.55 | 3105.05 | 3111.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 3119.55 | 3105.05 | 3111.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 3119.55 | 3105.05 | 3111.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 3111.30 | 3106.30 | 3111.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:45:00 | 3114.00 | 3106.30 | 3111.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 3130.65 | 3111.17 | 3113.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:45:00 | 3133.00 | 3111.17 | 3113.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 3132.05 | 3115.35 | 3115.00 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 13:15:00 | 3065.40 | 3109.97 | 3114.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 3021.60 | 3092.30 | 3105.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 13:15:00 | 2985.10 | 2962.44 | 2999.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 13:15:00 | 2985.10 | 2962.44 | 2999.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 2985.10 | 2962.44 | 2999.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 2985.10 | 2962.44 | 2999.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 3025.90 | 2974.72 | 2995.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:15:00 | 2952.05 | 2977.22 | 2994.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:00:00 | 2954.85 | 2972.75 | 2991.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:45:00 | 2974.40 | 2957.06 | 2957.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:15:00 | 2968.00 | 2957.06 | 2957.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 12:15:00 | 2977.75 | 2961.20 | 2959.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 2977.75 | 2961.20 | 2959.12 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 2918.80 | 2958.49 | 2959.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 2875.80 | 2921.53 | 2935.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 15:15:00 | 2883.50 | 2878.67 | 2899.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 2887.60 | 2880.46 | 2897.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 2887.60 | 2880.46 | 2897.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 2887.60 | 2880.46 | 2897.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 2808.70 | 2866.35 | 2888.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 09:30:00 | 2784.70 | 2807.65 | 2842.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 2901.25 | 2844.28 | 2837.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 2901.25 | 2844.28 | 2837.58 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 12:15:00 | 2829.35 | 2843.47 | 2843.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 13:15:00 | 2818.05 | 2838.38 | 2841.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 2696.40 | 2693.81 | 2727.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:30:00 | 2696.15 | 2693.81 | 2727.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 2614.20 | 2603.31 | 2623.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 2614.20 | 2603.31 | 2623.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 2611.85 | 2605.02 | 2622.41 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 2693.25 | 2634.28 | 2630.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 2712.00 | 2658.71 | 2642.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 2667.45 | 2681.53 | 2661.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 09:15:00 | 2670.00 | 2681.53 | 2661.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 2662.20 | 2677.66 | 2661.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 2662.20 | 2677.66 | 2661.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 2701.50 | 2682.43 | 2664.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 11:15:00 | 2716.95 | 2682.43 | 2664.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 15:15:00 | 2711.00 | 2700.83 | 2680.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-25 09:15:00 | 2988.64 | 2850.20 | 2780.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 2814.65 | 2874.13 | 2875.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 2787.00 | 2829.72 | 2847.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 2865.40 | 2803.90 | 2816.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 2865.40 | 2803.90 | 2816.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2865.40 | 2803.90 | 2816.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 2865.40 | 2803.90 | 2816.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 2845.65 | 2812.25 | 2818.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:30:00 | 2849.75 | 2812.25 | 2818.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 12:15:00 | 2862.70 | 2827.92 | 2825.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 13:15:00 | 2873.00 | 2836.93 | 2829.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 10:15:00 | 2867.55 | 2871.39 | 2850.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 11:00:00 | 2867.55 | 2871.39 | 2850.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 2845.15 | 2866.14 | 2850.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:00:00 | 2845.15 | 2866.14 | 2850.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 2845.45 | 2862.01 | 2849.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 13:30:00 | 2861.50 | 2861.82 | 2850.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 09:15:00 | 2836.95 | 2854.98 | 2850.39 | SL hit (close<static) qty=1.00 sl=2841.60 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 11:15:00 | 2823.20 | 2845.63 | 2846.75 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 2871.35 | 2849.29 | 2848.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 2904.10 | 2860.25 | 2853.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 10:15:00 | 2857.55 | 2865.43 | 2857.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 10:15:00 | 2857.55 | 2865.43 | 2857.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2857.55 | 2865.43 | 2857.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 2857.55 | 2865.43 | 2857.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 2864.95 | 2865.33 | 2858.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 2858.00 | 2865.33 | 2858.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 2858.90 | 2864.05 | 2858.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:45:00 | 2858.35 | 2864.05 | 2858.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 2861.75 | 2863.59 | 2858.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:30:00 | 2856.15 | 2863.59 | 2858.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 2861.70 | 2863.21 | 2859.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 2844.10 | 2863.21 | 2859.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 2863.00 | 2863.17 | 2859.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 2838.30 | 2863.17 | 2859.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 2813.35 | 2853.20 | 2855.33 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 2863.85 | 2849.97 | 2848.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 2906.90 | 2867.06 | 2857.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 2866.00 | 2894.99 | 2882.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 2866.00 | 2894.99 | 2882.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 2866.00 | 2894.99 | 2882.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 2866.00 | 2894.99 | 2882.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 2858.70 | 2887.74 | 2880.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 2858.70 | 2887.74 | 2880.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 2849.50 | 2874.56 | 2875.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 13:15:00 | 2841.25 | 2867.90 | 2872.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 14:15:00 | 2882.90 | 2870.90 | 2873.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 14:15:00 | 2882.90 | 2870.90 | 2873.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 2882.90 | 2870.90 | 2873.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 2882.90 | 2870.90 | 2873.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 2880.00 | 2872.72 | 2873.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 2866.55 | 2872.72 | 2873.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 14:15:00 | 2914.10 | 2861.37 | 2863.85 | SL hit (close>static) qty=1.00 sl=2891.70 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 2929.00 | 2874.89 | 2869.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 2949.60 | 2889.83 | 2877.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 2980.55 | 2984.73 | 2960.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 11:00:00 | 2980.55 | 2984.73 | 2960.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 2966.70 | 2987.88 | 2973.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:00:00 | 2990.05 | 2988.31 | 2975.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 12:00:00 | 2996.95 | 2990.04 | 2977.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:00:00 | 2988.70 | 2989.77 | 2978.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 2992.15 | 2982.07 | 2977.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 2998.50 | 2985.35 | 2979.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-20 12:15:00 | 2928.10 | 2973.22 | 2975.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 2928.10 | 2973.22 | 2975.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 2911.85 | 2960.95 | 2969.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 13:15:00 | 2853.15 | 2849.50 | 2874.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 14:00:00 | 2853.15 | 2849.50 | 2874.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 2868.20 | 2855.90 | 2871.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:00:00 | 2852.95 | 2855.31 | 2869.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 09:15:00 | 2710.30 | 2744.48 | 2762.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-07 09:15:00 | 2672.50 | 2663.70 | 2704.08 | SL hit (close>ema200) qty=0.50 sl=2663.70 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 2376.20 | 2347.07 | 2346.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 12:15:00 | 2381.00 | 2353.86 | 2349.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 2362.35 | 2364.33 | 2357.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 10:15:00 | 2362.35 | 2364.33 | 2357.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 2362.35 | 2364.33 | 2357.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 2362.35 | 2364.33 | 2357.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 2393.00 | 2370.07 | 2360.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 2372.80 | 2370.07 | 2360.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 2396.80 | 2415.89 | 2402.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 2407.25 | 2415.89 | 2402.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 2431.15 | 2418.94 | 2405.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 2424.55 | 2418.94 | 2405.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 2365.90 | 2410.50 | 2405.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 2365.90 | 2410.50 | 2405.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 2371.00 | 2402.60 | 2402.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 2308.25 | 2402.60 | 2402.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 2296.30 | 2381.34 | 2392.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 2243.00 | 2337.76 | 2369.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 2131.05 | 2121.39 | 2165.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 2181.00 | 2138.23 | 2166.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 2181.00 | 2138.23 | 2166.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 2181.00 | 2138.23 | 2166.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 2211.40 | 2152.86 | 2170.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 2211.40 | 2152.86 | 2170.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 2190.40 | 2160.37 | 2172.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:15:00 | 2185.75 | 2160.37 | 2172.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 10:15:00 | 2217.00 | 2180.57 | 2178.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 2217.00 | 2180.57 | 2178.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 2229.55 | 2197.25 | 2187.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 2312.25 | 2318.87 | 2293.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 2312.25 | 2318.87 | 2293.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 2312.25 | 2318.87 | 2293.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 2287.00 | 2318.87 | 2293.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 2334.50 | 2322.00 | 2297.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 2307.85 | 2322.00 | 2297.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 2306.00 | 2334.57 | 2313.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 2385.00 | 2321.70 | 2314.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 14:15:00 | 2304.00 | 2338.51 | 2342.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 2304.00 | 2338.51 | 2342.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 15:15:00 | 2294.70 | 2329.75 | 2338.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 2273.85 | 2272.56 | 2297.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 10:00:00 | 2273.85 | 2272.56 | 2297.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 2232.30 | 2268.02 | 2291.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 2231.60 | 2268.02 | 2291.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 13:15:00 | 2120.02 | 2163.29 | 2214.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 2008.44 | 2031.55 | 2100.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 2056.15 | 1983.07 | 1978.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 2073.90 | 2043.43 | 2018.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 2044.80 | 2047.14 | 2024.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 13:00:00 | 2044.80 | 2047.14 | 2024.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 2022.60 | 2054.29 | 2036.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 2022.60 | 2054.29 | 2036.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 2021.50 | 2047.73 | 2035.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 2021.50 | 2047.73 | 2035.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 2017.05 | 2041.60 | 2033.62 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 2003.00 | 2028.73 | 2028.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 1983.00 | 2014.25 | 2021.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 2011.40 | 2008.13 | 2015.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 12:15:00 | 2011.40 | 2008.13 | 2015.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 2011.40 | 2008.13 | 2015.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:30:00 | 2013.60 | 2008.13 | 2015.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 2006.50 | 2007.80 | 2015.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:45:00 | 1994.20 | 2012.15 | 2015.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:00:00 | 1995.00 | 2008.09 | 2013.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 14:15:00 | 2005.10 | 1966.63 | 1964.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 2005.10 | 1966.63 | 1964.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 2045.00 | 1987.64 | 1975.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 12:15:00 | 1995.00 | 2000.67 | 1985.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-04 13:00:00 | 1995.00 | 2000.67 | 1985.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 2068.05 | 2068.76 | 2053.04 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 13:15:00 | 2002.60 | 2038.92 | 2042.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 1987.20 | 2007.97 | 2022.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 2013.05 | 1998.11 | 2013.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 2013.05 | 1998.11 | 2013.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 2013.05 | 1998.11 | 2013.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:00:00 | 2013.05 | 1998.11 | 2013.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 2032.00 | 2004.89 | 2015.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 2032.00 | 2004.89 | 2015.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 2039.35 | 2011.78 | 2017.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 2039.35 | 2011.78 | 2017.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 2048.70 | 2021.90 | 2021.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 15:15:00 | 2050.00 | 2030.94 | 2025.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 2018.50 | 2028.45 | 2025.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 2018.50 | 2028.45 | 2025.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2018.50 | 2028.45 | 2025.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 2018.50 | 2028.45 | 2025.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 2008.90 | 2024.54 | 2023.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 2007.10 | 2024.54 | 2023.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 1997.50 | 2019.13 | 2021.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 12:15:00 | 1985.10 | 2005.58 | 2012.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 2000.40 | 1991.41 | 2002.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 2000.40 | 1991.41 | 2002.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 2000.40 | 1991.41 | 2002.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 2008.65 | 1991.41 | 2002.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1998.35 | 1992.80 | 2002.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:15:00 | 2005.85 | 1992.80 | 2002.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 2004.10 | 1995.06 | 2002.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 2008.45 | 1995.06 | 2002.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 1983.55 | 1992.76 | 2000.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:15:00 | 1979.00 | 1992.33 | 1999.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 2012.00 | 1994.13 | 1998.73 | SL hit (close>static) qty=1.00 sl=2004.10 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 2026.65 | 2004.61 | 2002.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 15:15:00 | 2033.90 | 2014.16 | 2007.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 2155.65 | 2172.81 | 2149.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 2155.65 | 2172.81 | 2149.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 2149.10 | 2168.07 | 2149.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 2149.00 | 2168.07 | 2149.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 2145.70 | 2163.60 | 2149.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 2131.00 | 2163.60 | 2149.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 2141.40 | 2159.16 | 2148.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 2141.40 | 2159.16 | 2148.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 2150.85 | 2157.49 | 2148.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 2163.05 | 2149.60 | 2146.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:00:00 | 2153.75 | 2150.43 | 2147.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:30:00 | 2152.55 | 2151.50 | 2148.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 12:00:00 | 2158.60 | 2152.92 | 2149.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 2142.15 | 2150.77 | 2148.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 2142.15 | 2150.77 | 2148.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 2140.25 | 2148.66 | 2147.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 2129.10 | 2144.75 | 2145.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 2129.10 | 2144.75 | 2145.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 2124.10 | 2138.26 | 2142.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 2130.20 | 2127.88 | 2135.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 15:00:00 | 2130.20 | 2127.88 | 2135.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 2140.00 | 2130.31 | 2135.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 2142.00 | 2130.31 | 2135.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2143.10 | 2132.86 | 2136.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 2119.70 | 2133.10 | 2135.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 2105.05 | 2130.31 | 2133.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 13:15:00 | 2141.85 | 2108.17 | 2106.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 2141.85 | 2108.17 | 2106.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 2147.85 | 2116.11 | 2109.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 2084.50 | 2134.02 | 2128.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 2084.50 | 2134.02 | 2128.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 2084.50 | 2134.02 | 2128.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 2084.50 | 2134.02 | 2128.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 2079.00 | 2123.02 | 2123.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1922.55 | 2054.05 | 2087.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 1966.90 | 1959.02 | 2006.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 1966.90 | 1959.02 | 2006.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 1958.65 | 1968.48 | 1991.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:30:00 | 1945.65 | 1961.60 | 1986.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:45:00 | 1950.65 | 1961.88 | 1984.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 14:00:00 | 1941.95 | 1972.07 | 1979.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 2005.90 | 1971.52 | 1976.95 | SL hit (close>static) qty=1.00 sl=1995.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 11:15:00 | 1995.30 | 1981.51 | 1980.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 2015.00 | 1994.32 | 1987.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 2004.50 | 2005.96 | 1996.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:30:00 | 2003.10 | 2005.96 | 1996.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 1999.40 | 2003.81 | 1996.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:45:00 | 1996.50 | 2003.81 | 1996.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1979.20 | 1997.59 | 1995.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 1979.20 | 1997.59 | 1995.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1988.10 | 1995.69 | 1994.97 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 11:15:00 | 1985.90 | 1993.73 | 1994.15 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 1999.40 | 1994.32 | 1994.31 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 14:15:00 | 1990.10 | 1993.47 | 1993.92 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 2012.10 | 1996.64 | 1995.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 2028.30 | 2002.97 | 1998.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 2101.80 | 2136.40 | 2120.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 2101.80 | 2136.40 | 2120.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2101.80 | 2136.40 | 2120.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2101.80 | 2136.40 | 2120.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 2066.60 | 2122.44 | 2115.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 2066.60 | 2122.44 | 2115.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 2096.50 | 2109.41 | 2110.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 2085.10 | 2104.55 | 2107.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 2114.80 | 2103.47 | 2106.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 2114.80 | 2103.47 | 2106.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 2114.80 | 2103.47 | 2106.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:45:00 | 2110.00 | 2103.47 | 2106.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 2103.10 | 2103.40 | 2106.37 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 2118.90 | 2107.92 | 2107.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 2146.00 | 2118.75 | 2113.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 2112.90 | 2119.46 | 2114.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 11:15:00 | 2112.90 | 2119.46 | 2114.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 2112.90 | 2119.46 | 2114.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 2112.90 | 2119.46 | 2114.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 2108.90 | 2117.35 | 2114.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:45:00 | 2112.30 | 2117.35 | 2114.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 2108.80 | 2115.64 | 2113.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:45:00 | 2108.40 | 2115.64 | 2113.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 2098.30 | 2112.17 | 2112.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 2098.30 | 2112.17 | 2112.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 15:15:00 | 2104.90 | 2110.72 | 2111.48 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 2196.30 | 2127.83 | 2119.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 10:15:00 | 2227.70 | 2147.81 | 2129.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 2142.20 | 2156.52 | 2141.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 15:15:00 | 2142.20 | 2156.52 | 2141.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 2142.20 | 2156.52 | 2141.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 2207.10 | 2156.52 | 2141.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 13:15:00 | 2182.60 | 2181.42 | 2160.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 2134.00 | 2181.71 | 2186.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 2134.00 | 2181.71 | 2186.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 2092.50 | 2155.42 | 2173.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 2129.90 | 2124.44 | 2150.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 2129.90 | 2124.44 | 2150.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2097.80 | 2025.60 | 2052.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 2083.00 | 2025.60 | 2052.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 2111.00 | 2068.91 | 2067.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 2135.50 | 2082.23 | 2074.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 2104.40 | 2109.01 | 2094.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:45:00 | 2102.60 | 2109.01 | 2094.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 2096.60 | 2106.53 | 2095.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 2096.60 | 2106.53 | 2095.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 2102.00 | 2105.62 | 2095.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:30:00 | 2110.00 | 2106.56 | 2097.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:00:00 | 2112.70 | 2107.79 | 2099.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 2112.30 | 2108.69 | 2100.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:15:00 | 2111.00 | 2107.85 | 2100.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 2116.00 | 2109.48 | 2102.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 2126.00 | 2111.51 | 2103.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 10:15:00 | 2097.10 | 2109.39 | 2104.85 | SL hit (close<static) qty=1.00 sl=2101.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 14:15:00 | 2180.40 | 2191.17 | 2191.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 2155.60 | 2182.35 | 2187.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 15:15:00 | 2169.00 | 2167.52 | 2176.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 09:15:00 | 2179.90 | 2167.52 | 2176.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2172.10 | 2168.44 | 2175.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 2179.00 | 2168.44 | 2175.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2175.00 | 2169.75 | 2175.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 2172.60 | 2169.75 | 2175.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 2173.30 | 2170.46 | 2175.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 2168.70 | 2170.46 | 2175.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 2174.50 | 2171.27 | 2175.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 2167.90 | 2172.13 | 2175.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 2182.80 | 2174.27 | 2176.17 | SL hit (close>static) qty=1.00 sl=2177.90 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 2210.10 | 2182.35 | 2179.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 12:15:00 | 2222.50 | 2196.59 | 2187.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 2255.90 | 2256.94 | 2238.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 12:45:00 | 2256.30 | 2256.94 | 2238.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 2234.50 | 2250.97 | 2239.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:45:00 | 2237.10 | 2250.97 | 2239.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 2230.00 | 2246.78 | 2238.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 2191.20 | 2246.78 | 2238.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 2200.00 | 2226.10 | 2229.61 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 2245.00 | 2231.00 | 2230.26 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 2215.00 | 2227.80 | 2228.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 2206.90 | 2223.62 | 2226.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 11:15:00 | 2234.00 | 2225.70 | 2227.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 11:15:00 | 2234.00 | 2225.70 | 2227.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 2234.00 | 2225.70 | 2227.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 2234.00 | 2225.70 | 2227.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 2248.00 | 2230.16 | 2229.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 11:15:00 | 2265.00 | 2242.59 | 2236.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 13:15:00 | 2275.00 | 2277.38 | 2263.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:00:00 | 2275.00 | 2277.38 | 2263.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 2267.40 | 2277.95 | 2267.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 2260.00 | 2277.95 | 2267.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 2259.30 | 2274.22 | 2266.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 2259.30 | 2274.22 | 2266.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 2265.00 | 2272.38 | 2266.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 2258.70 | 2272.38 | 2266.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 2266.00 | 2271.10 | 2266.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:45:00 | 2266.90 | 2271.10 | 2266.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 2266.90 | 2270.26 | 2266.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:30:00 | 2262.10 | 2270.26 | 2266.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 2262.20 | 2268.65 | 2266.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:45:00 | 2264.90 | 2268.65 | 2266.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 2262.60 | 2267.44 | 2265.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 2262.00 | 2267.44 | 2265.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 2269.00 | 2267.75 | 2266.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 2291.90 | 2271.16 | 2267.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 2421.60 | 2442.01 | 2442.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 2421.60 | 2442.01 | 2442.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 2412.40 | 2436.09 | 2439.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 2397.40 | 2388.72 | 2407.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 13:15:00 | 2397.40 | 2388.72 | 2407.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 2397.40 | 2388.72 | 2407.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 2397.40 | 2388.72 | 2407.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 2406.10 | 2392.20 | 2406.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 2406.10 | 2392.20 | 2406.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 2413.00 | 2396.36 | 2407.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 2411.50 | 2396.36 | 2407.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 2401.70 | 2397.43 | 2406.99 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 2429.20 | 2411.73 | 2411.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 2442.00 | 2420.88 | 2415.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 2452.00 | 2453.38 | 2442.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 13:45:00 | 2451.60 | 2453.38 | 2442.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 2453.90 | 2455.83 | 2447.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 2446.00 | 2455.83 | 2447.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 2435.00 | 2451.67 | 2446.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 2435.00 | 2451.67 | 2446.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 2385.10 | 2438.35 | 2440.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 14:15:00 | 2379.70 | 2418.04 | 2430.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 2421.10 | 2412.08 | 2425.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:00:00 | 2421.10 | 2412.08 | 2425.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 2429.90 | 2415.65 | 2425.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 2429.90 | 2415.65 | 2425.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 2428.40 | 2418.20 | 2425.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 2425.50 | 2418.20 | 2425.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 2414.60 | 2417.48 | 2424.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 2404.20 | 2419.25 | 2423.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 2408.10 | 2414.98 | 2420.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 2409.20 | 2413.19 | 2418.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 2432.70 | 2414.04 | 2417.29 | SL hit (close>static) qty=1.00 sl=2429.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 2436.50 | 2419.96 | 2419.51 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 2399.90 | 2415.92 | 2417.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 09:15:00 | 2396.60 | 2409.48 | 2414.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 2388.00 | 2382.35 | 2392.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 2388.00 | 2382.35 | 2392.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 2388.00 | 2382.35 | 2392.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 2388.00 | 2382.35 | 2392.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 2388.00 | 2383.48 | 2392.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 2392.20 | 2383.48 | 2392.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2374.50 | 2381.68 | 2390.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 15:00:00 | 2359.20 | 2373.63 | 2383.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 12:15:00 | 2309.00 | 2300.85 | 2300.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 2309.00 | 2300.85 | 2300.09 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 2276.70 | 2297.20 | 2298.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 2269.00 | 2288.03 | 2294.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 2299.20 | 2289.06 | 2293.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 2299.20 | 2289.06 | 2293.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 2299.20 | 2289.06 | 2293.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 2299.20 | 2289.06 | 2293.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 2298.10 | 2290.87 | 2293.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 2298.00 | 2290.87 | 2293.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 15:15:00 | 2308.00 | 2297.40 | 2296.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 2308.00 | 2297.40 | 2296.48 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 2271.10 | 2292.14 | 2294.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 2244.10 | 2260.00 | 2272.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 14:15:00 | 2257.50 | 2256.25 | 2268.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 15:00:00 | 2257.50 | 2256.25 | 2268.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 2256.00 | 2228.96 | 2239.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 2256.00 | 2228.96 | 2239.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 2255.20 | 2234.21 | 2240.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:30:00 | 2252.90 | 2238.83 | 2242.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 2249.10 | 2243.63 | 2244.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 2282.70 | 2251.45 | 2247.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 2282.70 | 2251.45 | 2247.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 2283.90 | 2257.94 | 2250.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 2368.10 | 2380.31 | 2358.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 2368.10 | 2380.31 | 2358.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 2368.10 | 2380.31 | 2358.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 2362.20 | 2380.31 | 2358.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 2362.70 | 2374.88 | 2359.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 2359.00 | 2374.88 | 2359.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 2352.60 | 2370.43 | 2359.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 2352.60 | 2370.43 | 2359.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 2366.70 | 2369.68 | 2359.91 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 2309.00 | 2355.55 | 2355.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 2259.80 | 2305.37 | 2320.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 2155.90 | 2155.16 | 2192.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 2155.90 | 2155.16 | 2192.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 2096.80 | 2084.14 | 2096.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 2096.80 | 2084.14 | 2096.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 2105.80 | 2088.47 | 2097.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 2114.00 | 2088.47 | 2097.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 2081.00 | 2086.98 | 2096.03 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 2113.00 | 2101.96 | 2100.53 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 2081.10 | 2097.79 | 2098.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 2068.70 | 2091.97 | 2096.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 12:15:00 | 2088.80 | 2088.24 | 2093.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 12:15:00 | 2088.80 | 2088.24 | 2093.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 2088.80 | 2088.24 | 2093.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:00:00 | 2088.80 | 2088.24 | 2093.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 2074.10 | 2085.41 | 2091.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:30:00 | 2080.90 | 2085.41 | 2091.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 2024.20 | 2042.83 | 2059.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 2013.40 | 2042.83 | 2059.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 2014.10 | 1971.38 | 1967.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 2014.10 | 1971.38 | 1967.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 2038.30 | 2016.90 | 2003.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 2054.70 | 2055.24 | 2035.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 2054.70 | 2055.24 | 2035.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 2038.00 | 2064.69 | 2059.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 2037.70 | 2064.69 | 2059.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 2020.00 | 2055.75 | 2055.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 2020.00 | 2055.75 | 2055.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 2021.70 | 2048.94 | 2052.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 2009.00 | 2036.40 | 2045.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1989.80 | 1984.31 | 2004.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1989.80 | 1984.31 | 2004.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1992.70 | 1962.29 | 1971.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 1992.70 | 1962.29 | 1971.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1998.00 | 1969.43 | 1973.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 1998.00 | 1969.43 | 1973.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 1981.30 | 1976.82 | 1976.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 2010.70 | 1984.10 | 1979.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 2014.40 | 2015.66 | 2003.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:15:00 | 2016.90 | 2015.66 | 2003.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 2003.00 | 2013.13 | 2003.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 2003.00 | 2013.13 | 2003.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 2001.70 | 2010.84 | 2003.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:15:00 | 2002.50 | 2010.84 | 2003.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 2002.50 | 2009.17 | 2003.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 2006.00 | 2009.17 | 2003.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 1967.80 | 1999.64 | 1999.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1967.80 | 1999.64 | 1999.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1950.70 | 1989.85 | 1995.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1977.20 | 1974.00 | 1983.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1977.20 | 1974.00 | 1983.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1977.20 | 1974.00 | 1983.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 1962.20 | 1972.61 | 1979.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1958.90 | 1971.79 | 1978.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 1963.60 | 1969.99 | 1977.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 1983.00 | 1970.99 | 1970.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 1983.00 | 1970.99 | 1970.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 2004.30 | 1994.15 | 1986.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 13:15:00 | 1994.00 | 1996.81 | 1990.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 13:15:00 | 1994.00 | 1996.81 | 1990.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 1994.00 | 1996.81 | 1990.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:45:00 | 1994.30 | 1996.81 | 1990.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1997.00 | 1996.85 | 1991.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 2026.60 | 1997.30 | 1991.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 2084.70 | 2102.16 | 2103.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 2084.70 | 2102.16 | 2103.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 2063.70 | 2094.47 | 2100.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 11:15:00 | 1983.00 | 1982.39 | 2013.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 12:00:00 | 1983.00 | 1982.39 | 2013.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1994.40 | 1975.66 | 1997.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 1994.40 | 1975.66 | 1997.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1999.10 | 1980.35 | 1997.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1989.40 | 1983.18 | 1997.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 2007.90 | 1988.12 | 1998.24 | SL hit (close>static) qty=1.00 sl=2004.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 2020.40 | 1993.95 | 1993.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 2030.50 | 2005.11 | 1998.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 2029.00 | 2030.99 | 2021.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 2030.80 | 2030.99 | 2021.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 2048.00 | 2046.47 | 2036.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 2053.80 | 2048.74 | 2038.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 2029.50 | 2044.01 | 2043.37 | SL hit (close<static) qty=1.00 sl=2033.40 alert=retest2 |

### Cycle 104 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 2024.30 | 2040.07 | 2041.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 2013.70 | 2033.43 | 2038.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 2029.90 | 2029.08 | 2034.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 2029.90 | 2029.08 | 2034.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 2034.00 | 2030.06 | 2034.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 2056.00 | 2030.06 | 2034.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 2053.30 | 2034.71 | 2035.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 2073.80 | 2034.71 | 2035.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 2067.00 | 2041.17 | 2038.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 2078.00 | 2048.53 | 2042.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 2077.60 | 2085.05 | 2073.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 2077.60 | 2085.05 | 2073.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 2077.60 | 2085.05 | 2073.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 2081.00 | 2085.05 | 2073.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 2070.90 | 2082.22 | 2073.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 2070.90 | 2082.22 | 2073.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 2054.20 | 2076.62 | 2071.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 2054.20 | 2076.62 | 2071.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 2056.40 | 2066.83 | 2067.98 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 2129.80 | 2079.00 | 2073.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 2133.00 | 2097.94 | 2083.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 2292.10 | 2294.71 | 2265.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:00:00 | 2292.10 | 2294.71 | 2265.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 2271.00 | 2284.86 | 2269.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 2274.90 | 2284.86 | 2269.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 2277.30 | 2283.35 | 2270.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 2271.10 | 2283.35 | 2270.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 2290.80 | 2288.69 | 2277.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 2277.90 | 2288.69 | 2277.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 2306.00 | 2317.15 | 2303.78 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 2294.70 | 2298.85 | 2299.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 11:15:00 | 2282.00 | 2295.48 | 2297.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 2300.10 | 2293.07 | 2295.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 14:15:00 | 2300.10 | 2293.07 | 2295.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 2300.10 | 2293.07 | 2295.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 2300.10 | 2293.07 | 2295.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 2314.00 | 2297.26 | 2297.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 2289.00 | 2297.26 | 2297.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 2298.90 | 2293.88 | 2294.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 2302.00 | 2295.50 | 2295.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 15:15:00 | 2302.00 | 2295.50 | 2295.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 2333.00 | 2303.00 | 2298.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 11:15:00 | 2303.50 | 2305.02 | 2300.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 12:00:00 | 2303.50 | 2305.02 | 2300.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 2299.40 | 2303.90 | 2300.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 2299.40 | 2303.90 | 2300.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 2304.30 | 2303.98 | 2300.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:00:00 | 2304.30 | 2303.98 | 2300.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 2286.70 | 2300.52 | 2299.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 2286.70 | 2300.52 | 2299.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 2285.90 | 2297.60 | 2298.33 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 2311.00 | 2300.66 | 2299.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 2325.30 | 2306.92 | 2302.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 14:15:00 | 2304.80 | 2307.41 | 2303.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 2304.80 | 2307.41 | 2303.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 2304.80 | 2307.41 | 2303.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 2304.80 | 2307.41 | 2303.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 2313.90 | 2308.71 | 2304.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 2309.00 | 2308.71 | 2304.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2319.00 | 2310.76 | 2305.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 2298.00 | 2310.76 | 2305.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 2302.00 | 2310.12 | 2306.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 2302.70 | 2310.12 | 2306.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 2309.50 | 2310.00 | 2306.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:30:00 | 2309.90 | 2310.00 | 2306.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 2301.60 | 2308.32 | 2306.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 2301.60 | 2308.32 | 2306.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 2290.60 | 2304.77 | 2304.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 2285.90 | 2299.44 | 2302.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 2303.00 | 2300.15 | 2302.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 2303.00 | 2300.15 | 2302.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 2303.00 | 2300.15 | 2302.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 2303.00 | 2300.15 | 2302.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 2229.70 | 2286.06 | 2295.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:15:00 | 2197.00 | 2286.06 | 2295.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:30:00 | 2223.90 | 2264.44 | 2283.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 2199.60 | 2177.70 | 2177.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 2199.60 | 2177.70 | 2177.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 2215.40 | 2189.54 | 2183.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 2202.40 | 2204.79 | 2194.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 2202.40 | 2204.79 | 2194.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 2192.50 | 2202.49 | 2195.48 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 2187.00 | 2193.68 | 2194.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 2156.30 | 2183.00 | 2188.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 2066.20 | 2062.90 | 2084.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 2066.20 | 2062.90 | 2084.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 2084.00 | 2070.40 | 2081.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 2084.00 | 2070.40 | 2081.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2094.30 | 2075.18 | 2082.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 2090.00 | 2075.18 | 2082.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 2092.00 | 2078.55 | 2083.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 2083.10 | 2078.55 | 2083.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 2107.00 | 2084.24 | 2085.86 | SL hit (close>static) qty=1.00 sl=2096.70 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 2111.90 | 2091.64 | 2089.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 2117.90 | 2099.33 | 2093.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 2092.20 | 2104.76 | 2099.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 12:15:00 | 2092.20 | 2104.76 | 2099.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 2092.20 | 2104.76 | 2099.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 2092.20 | 2104.76 | 2099.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 2095.60 | 2102.93 | 2099.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 2089.70 | 2102.93 | 2099.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 2099.00 | 2101.15 | 2099.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 2111.90 | 2101.15 | 2099.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:30:00 | 2106.80 | 2104.45 | 2100.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 2082.70 | 2103.48 | 2102.94 | SL hit (close<static) qty=1.00 sl=2094.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 2096.50 | 2102.08 | 2102.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 2065.10 | 2090.28 | 2095.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 2085.90 | 2076.83 | 2083.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 10:15:00 | 2085.90 | 2076.83 | 2083.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 2085.90 | 2076.83 | 2083.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 2085.90 | 2076.83 | 2083.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 2089.10 | 2079.29 | 2083.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 2091.50 | 2079.29 | 2083.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 2096.90 | 2082.81 | 2084.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 2098.80 | 2082.81 | 2084.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 2112.10 | 2087.20 | 2086.15 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 15:15:00 | 2082.60 | 2085.94 | 2086.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 2022.30 | 2073.21 | 2080.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 2011.90 | 1998.99 | 2028.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 2011.90 | 1998.99 | 2028.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 2040.70 | 2007.33 | 2029.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 2040.70 | 2007.33 | 2029.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2041.70 | 2014.21 | 2030.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:30:00 | 2046.30 | 2014.21 | 2030.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 2019.40 | 2027.05 | 2032.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 2009.80 | 2021.42 | 2028.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 15:15:00 | 2009.90 | 2020.65 | 2027.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 2004.80 | 2018.64 | 2025.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 15:15:00 | 2036.80 | 2029.02 | 2028.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 2036.80 | 2029.02 | 2028.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 2067.00 | 2036.61 | 2031.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 2048.20 | 2056.71 | 2047.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 2048.20 | 2056.71 | 2047.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2048.20 | 2056.71 | 2047.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 2048.20 | 2056.71 | 2047.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 2065.30 | 2058.43 | 2048.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 2049.10 | 2058.43 | 2048.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2043.20 | 2059.24 | 2053.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 2043.20 | 2059.24 | 2053.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2040.40 | 2055.47 | 2052.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 2040.40 | 2055.47 | 2052.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 2031.90 | 2047.66 | 2049.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 2025.50 | 2043.23 | 2047.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 2011.20 | 2006.69 | 2016.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 2011.20 | 2006.69 | 2016.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2011.20 | 2006.69 | 2016.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 2022.30 | 2006.69 | 2016.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2006.00 | 2006.55 | 2015.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 2010.00 | 2006.55 | 2015.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 2013.10 | 2007.82 | 2014.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 2016.10 | 2007.82 | 2014.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 2035.10 | 2013.28 | 2016.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 2035.10 | 2013.28 | 2016.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2045.90 | 2019.80 | 2019.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2050.00 | 2029.54 | 2023.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 11:15:00 | 2032.00 | 2032.58 | 2026.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 2032.00 | 2032.58 | 2026.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 2030.30 | 2032.13 | 2026.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 12:45:00 | 2026.00 | 2032.13 | 2026.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 2029.40 | 2031.92 | 2028.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 2032.50 | 2031.92 | 2028.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2039.50 | 2033.43 | 2029.07 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 14:15:00 | 2019.80 | 2026.45 | 2026.89 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 2032.00 | 2027.80 | 2027.34 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 2017.90 | 2025.85 | 2026.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 2003.80 | 2021.44 | 2024.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 11:15:00 | 2015.10 | 2012.78 | 2018.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 11:15:00 | 2015.10 | 2012.78 | 2018.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2015.10 | 2012.78 | 2018.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 2016.40 | 2012.78 | 2018.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 2006.10 | 2006.66 | 2013.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 2011.50 | 2006.66 | 2013.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1997.80 | 1986.52 | 1994.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 1997.80 | 1986.52 | 1994.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2007.00 | 1990.61 | 1995.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 2007.00 | 1990.61 | 1995.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 2012.10 | 1998.33 | 1998.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 2024.50 | 2009.17 | 2003.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 2112.20 | 2130.25 | 2117.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 2112.20 | 2130.25 | 2117.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2112.20 | 2130.25 | 2117.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 2112.20 | 2130.25 | 2117.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2108.80 | 2125.96 | 2116.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 2108.80 | 2125.96 | 2116.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 2106.80 | 2122.13 | 2115.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 2105.20 | 2122.13 | 2115.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 2090.10 | 2108.82 | 2110.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 2024.60 | 2091.98 | 2102.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 14:15:00 | 1872.20 | 1871.24 | 1906.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 15:00:00 | 1872.20 | 1871.24 | 1906.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1926.10 | 1882.30 | 1905.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:15:00 | 1877.60 | 1894.58 | 1905.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1860.90 | 1891.95 | 1901.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 1783.72 | 1812.85 | 1848.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 1767.86 | 1812.85 | 1848.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 1689.84 | 1738.26 | 1790.56 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 1563.70 | 1555.61 | 1555.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1584.30 | 1561.35 | 1557.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 1561.90 | 1564.69 | 1560.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 1561.90 | 1564.69 | 1560.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1561.90 | 1564.69 | 1560.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:15:00 | 1564.90 | 1564.69 | 1560.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 1575.30 | 1566.81 | 1561.80 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 1535.00 | 1556.35 | 1559.12 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1663.60 | 1569.27 | 1557.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 1694.90 | 1594.40 | 1569.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1678.90 | 1696.39 | 1663.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1678.90 | 1696.39 | 1663.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 1669.00 | 1685.04 | 1666.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:45:00 | 1695.70 | 1685.04 | 1666.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 1679.90 | 1684.01 | 1667.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 1698.90 | 1685.47 | 1671.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 1654.90 | 1677.12 | 1669.58 | SL hit (close<static) qty=1.00 sl=1666.10 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1770.30 | 1801.12 | 1804.94 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 1818.00 | 1807.12 | 1806.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 1844.10 | 1830.63 | 1822.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 1859.60 | 1866.09 | 1850.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:30:00 | 1863.50 | 1866.09 | 1850.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1840.00 | 1860.87 | 1849.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1840.00 | 1860.87 | 1849.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1842.90 | 1857.28 | 1849.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 1842.90 | 1857.28 | 1849.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 1832.50 | 1852.32 | 1847.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:30:00 | 1832.30 | 1852.32 | 1847.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 1806.00 | 1837.89 | 1841.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 1793.00 | 1828.91 | 1837.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 1824.80 | 1819.34 | 1828.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 13:15:00 | 1824.80 | 1819.34 | 1828.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1824.80 | 1819.34 | 1828.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 1824.80 | 1819.34 | 1828.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1825.40 | 1820.56 | 1828.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:30:00 | 1831.40 | 1820.56 | 1828.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1813.10 | 1819.06 | 1827.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 1846.60 | 1819.06 | 1827.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1831.30 | 1821.51 | 1827.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1841.40 | 1821.51 | 1827.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1825.30 | 1822.27 | 1827.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1815.20 | 1822.27 | 1827.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 1814.20 | 1816.68 | 1823.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1806.50 | 1822.02 | 1824.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1724.44 | 1744.97 | 1767.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1723.49 | 1744.97 | 1767.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1716.17 | 1744.97 | 1767.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 1736.00 | 1729.13 | 1749.43 | SL hit (close>ema200) qty=0.50 sl=1729.13 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1683.30 | 1672.53 | 1672.07 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 11:15:00 | 1663.50 | 1670.16 | 1671.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 1655.90 | 1667.31 | 1669.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1567.90 | 1567.11 | 1586.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 1567.90 | 1567.11 | 1586.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1589.20 | 1567.97 | 1580.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 1589.20 | 1567.97 | 1580.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1584.70 | 1571.31 | 1580.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 1593.00 | 1571.31 | 1580.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1581.00 | 1573.25 | 1580.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1600.20 | 1573.25 | 1580.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1631.70 | 1593.70 | 1588.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 1661.00 | 1607.16 | 1595.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1577.00 | 1615.05 | 1604.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1577.00 | 1615.05 | 1604.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1577.00 | 1615.05 | 1604.46 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1567.80 | 1595.44 | 1597.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1561.70 | 1588.69 | 1594.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1598.10 | 1581.00 | 1588.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1598.10 | 1581.00 | 1588.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1598.10 | 1581.00 | 1588.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 1594.60 | 1581.00 | 1588.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1585.40 | 1581.88 | 1587.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 1593.60 | 1581.88 | 1587.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1568.70 | 1579.24 | 1586.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 1562.40 | 1579.24 | 1586.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 1564.90 | 1574.41 | 1582.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1486.65 | 1518.10 | 1545.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 1527.00 | 1514.54 | 1536.84 | SL hit (close>ema200) qty=0.50 sl=1514.54 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1585.50 | 1544.23 | 1542.45 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1511.80 | 1544.06 | 1547.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 1502.40 | 1535.73 | 1543.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1532.00 | 1491.93 | 1505.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1532.00 | 1491.93 | 1505.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1532.00 | 1491.93 | 1505.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1532.00 | 1491.93 | 1505.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 1507.60 | 1507.62 | 1509.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 15:00:00 | 1507.60 | 1507.62 | 1509.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 1514.00 | 1487.01 | 1494.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 1514.00 | 1487.01 | 1494.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1508.60 | 1491.33 | 1495.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 1507.00 | 1491.33 | 1495.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 1559.80 | 1505.02 | 1501.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 1573.00 | 1518.62 | 1507.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1693.90 | 1695.01 | 1665.83 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1713.70 | 1695.01 | 1665.83 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1677.40 | 1708.87 | 1692.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1677.40 | 1708.87 | 1692.65 | SL hit (close<ema400) qty=1.00 sl=1692.65 alert=retest1 |

### Cycle 140 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1783.20 | 1801.38 | 1802.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1778.60 | 1796.82 | 1799.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1800.90 | 1784.59 | 1791.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1800.90 | 1784.59 | 1791.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1800.90 | 1784.59 | 1791.19 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1830.10 | 1801.25 | 1797.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 1831.50 | 1807.30 | 1800.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 1822.20 | 1825.95 | 1815.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 1822.20 | 1825.95 | 1815.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1832.50 | 1852.55 | 1840.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1876.00 | 1839.29 | 1837.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 1815.40 | 1863.20 | 1859.05 | SL hit (close<static) qty=1.00 sl=1815.50 alert=retest2 |

### Cycle 142 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 1821.90 | 1850.40 | 1853.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 13:15:00 | 1807.40 | 1841.80 | 1849.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 10:15:00 | 1837.20 | 1831.92 | 1841.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 10:15:00 | 1837.20 | 1831.92 | 1841.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1837.20 | 1831.92 | 1841.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:30:00 | 1840.00 | 1831.92 | 1841.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1830.50 | 1831.63 | 1840.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 1826.90 | 1831.63 | 1840.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 1864.20 | 1837.55 | 1840.67 | SL hit (close>static) qty=1.00 sl=1841.60 alert=retest2 |

### Cycle 143 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 1868.00 | 1843.64 | 1843.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 14:15:00 | 1876.00 | 1855.13 | 1849.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 1858.00 | 1859.44 | 1852.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 1858.00 | 1859.44 | 1852.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1858.00 | 1859.44 | 1852.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 1863.30 | 1859.44 | 1852.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1853.20 | 1858.19 | 1852.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 1851.40 | 1858.19 | 1852.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 1870.40 | 1860.63 | 1854.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 1855.90 | 1860.63 | 1854.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-23 10:15:00 | 2777.25 | 2024-05-27 10:15:00 | 2849.95 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-05-24 09:15:00 | 2762.55 | 2024-05-27 10:15:00 | 2849.95 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-05-30 13:15:00 | 2725.55 | 2024-05-31 14:15:00 | 2783.75 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-05-30 14:00:00 | 2727.80 | 2024-05-31 14:15:00 | 2783.75 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-06-12 11:15:00 | 2896.90 | 2024-06-21 15:15:00 | 2999.60 | STOP_HIT | 1.00 | 3.55% |
| BUY | retest2 | 2024-06-27 11:45:00 | 3085.45 | 2024-07-05 11:15:00 | 3241.25 | STOP_HIT | 1.00 | 5.05% |
| BUY | retest2 | 2024-06-27 13:30:00 | 3071.10 | 2024-07-05 11:15:00 | 3241.25 | STOP_HIT | 1.00 | 5.54% |
| SELL | retest2 | 2024-07-22 15:00:00 | 3241.65 | 2024-07-23 13:15:00 | 3079.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 09:30:00 | 3252.00 | 2024-07-23 13:15:00 | 3089.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 15:00:00 | 3241.65 | 2024-07-26 09:15:00 | 3127.90 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2024-07-23 09:30:00 | 3252.00 | 2024-07-26 09:15:00 | 3127.90 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2024-07-23 12:15:00 | 3133.50 | 2024-07-29 13:15:00 | 3148.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-08-08 13:15:00 | 2911.45 | 2024-08-13 10:15:00 | 2944.15 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-08-09 10:15:00 | 2918.00 | 2024-08-13 10:15:00 | 2944.15 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-08-09 11:00:00 | 2915.05 | 2024-08-13 10:15:00 | 2944.15 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-08-12 09:15:00 | 2908.45 | 2024-08-13 10:15:00 | 2944.15 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-08-27 12:45:00 | 2896.95 | 2024-08-30 14:15:00 | 2912.30 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-08-27 14:45:00 | 2896.05 | 2024-08-30 14:15:00 | 2912.30 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-08-28 09:30:00 | 2885.00 | 2024-08-30 14:15:00 | 2912.30 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-08-28 11:15:00 | 2897.65 | 2024-08-30 14:15:00 | 2912.30 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-08-28 13:00:00 | 2882.55 | 2024-08-30 14:15:00 | 2912.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-08-28 14:45:00 | 2883.70 | 2024-08-30 14:15:00 | 2912.30 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-08-28 15:15:00 | 2884.00 | 2024-08-30 14:15:00 | 2912.30 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-08-29 09:30:00 | 2879.85 | 2024-08-30 14:15:00 | 2912.30 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-08-30 10:15:00 | 2877.70 | 2024-08-30 14:15:00 | 2912.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-09-03 09:15:00 | 2922.55 | 2024-09-05 10:15:00 | 2887.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-09-03 12:15:00 | 2890.70 | 2024-09-05 11:15:00 | 2883.95 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-09-04 10:00:00 | 2895.65 | 2024-09-05 11:15:00 | 2883.95 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-09-04 13:00:00 | 2894.30 | 2024-09-05 11:15:00 | 2883.95 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-09-05 09:15:00 | 2922.55 | 2024-09-05 11:15:00 | 2883.95 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-09-19 11:15:00 | 2833.00 | 2024-09-19 14:15:00 | 2913.00 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2024-09-19 12:45:00 | 2848.00 | 2024-09-19 14:15:00 | 2913.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-10-09 11:45:00 | 2932.70 | 2024-10-09 13:15:00 | 2996.50 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-10-11 14:30:00 | 3012.20 | 2024-10-17 15:15:00 | 3080.00 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2024-10-24 11:15:00 | 2952.05 | 2024-10-28 12:15:00 | 2977.75 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-10-24 12:00:00 | 2954.85 | 2024-10-28 12:15:00 | 2977.75 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-10-28 11:45:00 | 2974.40 | 2024-10-28 12:15:00 | 2977.75 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-10-28 12:15:00 | 2968.00 | 2024-10-28 12:15:00 | 2977.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-11-05 09:30:00 | 2784.70 | 2024-11-06 11:15:00 | 2901.25 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-11-21 11:15:00 | 2716.95 | 2024-11-25 09:15:00 | 2988.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-21 15:15:00 | 2711.00 | 2024-11-25 09:15:00 | 2982.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-03 13:30:00 | 2861.50 | 2024-12-04 09:15:00 | 2836.95 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-13 09:15:00 | 2866.55 | 2024-12-13 14:15:00 | 2914.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-12-19 11:00:00 | 2990.05 | 2024-12-20 12:15:00 | 2928.10 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-12-19 12:00:00 | 2996.95 | 2024-12-20 12:15:00 | 2928.10 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-12-19 13:00:00 | 2988.70 | 2024-12-20 12:15:00 | 2928.10 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-12-20 09:30:00 | 2992.15 | 2024-12-20 12:15:00 | 2928.10 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-12-27 11:00:00 | 2852.95 | 2025-01-06 09:15:00 | 2710.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-27 11:00:00 | 2852.95 | 2025-01-07 09:15:00 | 2672.50 | STOP_HIT | 0.50 | 6.33% |
| SELL | retest2 | 2025-01-28 14:15:00 | 2185.75 | 2025-01-29 10:15:00 | 2217.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-02-04 09:15:00 | 2385.00 | 2025-02-05 14:15:00 | 2304.00 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-02-07 12:15:00 | 2231.60 | 2025-02-10 13:15:00 | 2120.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:15:00 | 2231.60 | 2025-02-12 09:15:00 | 2008.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-25 11:45:00 | 1994.20 | 2025-03-03 14:15:00 | 2005.10 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-02-25 14:00:00 | 1995.00 | 2025-03-03 14:15:00 | 2005.10 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-03-17 15:15:00 | 1979.00 | 2025-03-18 09:15:00 | 2012.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-03-26 09:15:00 | 2163.05 | 2025-03-26 14:15:00 | 2129.10 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-03-26 10:00:00 | 2153.75 | 2025-03-26 14:15:00 | 2129.10 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-03-26 10:30:00 | 2152.55 | 2025-03-26 14:15:00 | 2129.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-03-26 12:00:00 | 2158.60 | 2025-03-26 14:15:00 | 2129.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-03-28 13:00:00 | 2119.70 | 2025-04-02 13:15:00 | 2141.85 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-04-01 09:15:00 | 2105.05 | 2025-04-02 13:15:00 | 2141.85 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-04-09 10:30:00 | 1945.65 | 2025-04-15 09:15:00 | 2005.90 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-04-09 11:45:00 | 1950.65 | 2025-04-15 09:15:00 | 2005.90 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-04-11 14:00:00 | 1941.95 | 2025-04-15 09:15:00 | 2005.90 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-05-02 09:15:00 | 2207.10 | 2025-05-06 11:15:00 | 2134.00 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2025-05-02 13:15:00 | 2182.60 | 2025-05-06 11:15:00 | 2134.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-05-14 09:30:00 | 2110.00 | 2025-05-15 10:15:00 | 2097.10 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-05-14 11:00:00 | 2112.70 | 2025-05-21 14:15:00 | 2180.40 | STOP_HIT | 1.00 | 3.20% |
| BUY | retest2 | 2025-05-14 12:00:00 | 2112.30 | 2025-05-21 14:15:00 | 2180.40 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-05-14 13:15:00 | 2111.00 | 2025-05-21 14:15:00 | 2180.40 | STOP_HIT | 1.00 | 3.29% |
| BUY | retest2 | 2025-05-14 15:15:00 | 2126.00 | 2025-05-21 14:15:00 | 2180.40 | STOP_HIT | 1.00 | 2.56% |
| BUY | retest2 | 2025-05-15 13:45:00 | 2128.00 | 2025-05-21 14:15:00 | 2180.40 | STOP_HIT | 1.00 | 2.46% |
| SELL | retest2 | 2025-05-23 13:30:00 | 2167.90 | 2025-05-23 14:15:00 | 2182.80 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-06-05 10:30:00 | 2291.90 | 2025-06-12 11:15:00 | 2421.60 | STOP_HIT | 1.00 | 5.66% |
| SELL | retest2 | 2025-06-23 09:15:00 | 2404.20 | 2025-06-24 09:15:00 | 2432.70 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-06-23 11:00:00 | 2408.10 | 2025-06-24 09:15:00 | 2432.70 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-06-23 14:15:00 | 2409.20 | 2025-06-24 09:15:00 | 2432.70 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-06-27 15:00:00 | 2359.20 | 2025-07-07 12:15:00 | 2309.00 | STOP_HIT | 1.00 | 2.13% |
| SELL | retest2 | 2025-07-08 14:15:00 | 2298.00 | 2025-07-08 15:15:00 | 2308.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-14 14:30:00 | 2252.90 | 2025-07-15 10:15:00 | 2282.70 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-15 09:30:00 | 2249.10 | 2025-07-15 10:15:00 | 2282.70 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-08-08 10:15:00 | 2013.40 | 2025-08-18 10:15:00 | 2014.10 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-09-05 09:15:00 | 2006.00 | 2025-09-05 10:15:00 | 1967.80 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-09-08 15:00:00 | 1962.20 | 2025-09-10 12:15:00 | 1983.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1958.90 | 2025-09-10 12:15:00 | 1983.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-09-09 09:45:00 | 1963.60 | 2025-09-10 12:15:00 | 1983.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-15 09:15:00 | 2026.60 | 2025-09-24 09:15:00 | 2084.70 | STOP_HIT | 1.00 | 2.87% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1989.40 | 2025-09-29 12:15:00 | 2007.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1992.60 | 2025-10-01 11:15:00 | 2020.40 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-30 10:00:00 | 1992.50 | 2025-10-01 11:15:00 | 2020.40 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-30 11:30:00 | 1987.80 | 2025-10-01 11:15:00 | 2020.40 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-10-07 12:45:00 | 2053.80 | 2025-10-08 14:15:00 | 2029.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-30 09:15:00 | 2289.00 | 2025-10-30 15:15:00 | 2302.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-30 14:45:00 | 2298.90 | 2025-10-30 15:15:00 | 2302.00 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-11-06 12:15:00 | 2197.00 | 2025-11-12 13:15:00 | 2199.60 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-06 13:30:00 | 2223.90 | 2025-11-12 13:15:00 | 2199.60 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2025-11-26 09:15:00 | 2083.10 | 2025-11-26 09:15:00 | 2107.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-11-28 09:15:00 | 2111.90 | 2025-12-01 11:15:00 | 2082.70 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-11-28 10:30:00 | 2106.80 | 2025-12-01 11:15:00 | 2082.70 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-10 14:00:00 | 2009.80 | 2025-12-11 15:15:00 | 2036.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-12-10 15:15:00 | 2009.90 | 2025-12-11 15:15:00 | 2036.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-12-11 09:30:00 | 2004.80 | 2025-12-11 15:15:00 | 2036.80 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-01-16 14:15:00 | 1877.60 | 2026-01-20 09:15:00 | 1783.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1860.90 | 2026-01-20 09:15:00 | 1767.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 14:15:00 | 1877.60 | 2026-01-20 15:15:00 | 1689.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1860.90 | 2026-01-21 09:15:00 | 1674.81 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-06 09:15:00 | 1698.90 | 2026-02-06 10:15:00 | 1654.90 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2026-02-06 14:30:00 | 1696.40 | 2026-02-13 10:15:00 | 1770.30 | STOP_HIT | 1.00 | 4.36% |
| BUY | retest2 | 2026-02-09 09:15:00 | 1781.90 | 2026-02-13 10:15:00 | 1770.30 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1815.20 | 2026-03-02 09:15:00 | 1724.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:30:00 | 1814.20 | 2026-03-02 09:15:00 | 1723.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1806.50 | 2026-03-02 09:15:00 | 1716.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1815.20 | 2026-03-02 14:15:00 | 1736.00 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2026-02-23 13:30:00 | 1814.20 | 2026-03-02 14:15:00 | 1736.00 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1806.50 | 2026-03-02 14:15:00 | 1736.00 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1562.40 | 2026-03-23 13:15:00 | 1486.65 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1562.40 | 2026-03-24 09:15:00 | 1527.00 | STOP_HIT | 0.50 | 2.27% |
| SELL | retest2 | 2026-03-20 13:30:00 | 1564.90 | 2026-03-25 09:15:00 | 1585.50 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1713.70 | 2026-04-13 09:15:00 | 1677.40 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-04-13 10:45:00 | 1702.90 | 2026-04-24 10:15:00 | 1783.20 | STOP_HIT | 1.00 | 4.72% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1876.00 | 2026-05-05 10:15:00 | 1815.40 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2026-05-06 12:15:00 | 1826.90 | 2026-05-06 14:15:00 | 1864.20 | STOP_HIT | 1.00 | -2.04% |
