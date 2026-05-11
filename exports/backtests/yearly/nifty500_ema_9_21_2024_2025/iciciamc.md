# ICICI Prudential Asset Management Company Ltd. (ICICIAMC)

## Backtest Summary

- **Window:** 2025-12-19 09:15:00 → 2026-05-08 15:15:00 (651 bars)
- **Last close:** 3240.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 39 |
| ALERT1 | 23 |
| ALERT2 | 22 |
| ALERT2_SKIP | 16 |
| ALERT3 | 57 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 0 / 18 / 3
- **Avg / median % per leg:** 0.04% / -0.47%
- **Sum % (uncompounded):** 0.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 7 | 0 | -1.10% | -7.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 0 | 7 | 0 | -1.10% | -7.7% |
| SELL (all) | 14 | 6 | 42.9% | 0 | 11 | 3 | 0.61% | 8.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 0 | 11 | 3 | 0.61% | 8.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 8 | 38.1% | 0 | 18 | 3 | 0.04% | 0.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 2629.90 | 2647.01 | 2648.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 15:15:00 | 2624.90 | 2642.59 | 2646.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 11:15:00 | 2642.00 | 2640.89 | 2644.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 2642.00 | 2640.89 | 2644.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 2642.00 | 2640.89 | 2644.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 2642.00 | 2640.89 | 2644.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 2655.00 | 2643.71 | 2645.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:45:00 | 2655.50 | 2643.71 | 2645.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 2655.90 | 2646.15 | 2646.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 2654.60 | 2646.15 | 2646.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 2669.00 | 2650.72 | 2648.33 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 2640.10 | 2646.63 | 2647.01 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 2649.90 | 2647.28 | 2647.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 12:15:00 | 2655.00 | 2648.83 | 2647.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 2696.30 | 2698.19 | 2688.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 11:15:00 | 2696.30 | 2698.19 | 2688.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2696.30 | 2698.19 | 2688.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:45:00 | 2686.10 | 2698.19 | 2688.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 2700.00 | 2698.55 | 2689.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:00:00 | 2708.40 | 2700.52 | 2691.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 2708.00 | 2703.97 | 2695.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 2663.00 | 2695.77 | 2692.80 | SL hit (close<static) qty=1.00 sl=2688.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 2651.90 | 2687.00 | 2689.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 2590.80 | 2655.50 | 2671.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 2674.90 | 2634.68 | 2642.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 2674.90 | 2634.68 | 2642.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 2674.90 | 2634.68 | 2642.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 2670.10 | 2634.68 | 2642.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 2673.00 | 2642.35 | 2645.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 2664.00 | 2642.35 | 2645.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 2671.10 | 2648.10 | 2647.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 2687.00 | 2655.88 | 2651.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 11:15:00 | 2870.30 | 2897.86 | 2828.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 12:00:00 | 2870.30 | 2897.86 | 2828.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 2844.40 | 2897.28 | 2855.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:00:00 | 2844.40 | 2897.28 | 2855.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 2817.60 | 2881.34 | 2852.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 2817.60 | 2881.34 | 2852.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 2812.00 | 2867.48 | 2848.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 2812.00 | 2867.48 | 2848.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 2824.00 | 2839.53 | 2839.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 2797.40 | 2839.53 | 2839.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 2787.10 | 2829.04 | 2834.49 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 2860.00 | 2828.66 | 2827.71 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 12:15:00 | 2791.40 | 2822.17 | 2825.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 09:15:00 | 2764.00 | 2794.85 | 2809.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 11:15:00 | 2800.00 | 2794.46 | 2807.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 11:15:00 | 2800.00 | 2794.46 | 2807.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 2800.00 | 2794.46 | 2807.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 2800.00 | 2794.46 | 2807.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2745.60 | 2777.39 | 2793.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 15:00:00 | 2702.50 | 2743.62 | 2768.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 2833.00 | 2782.77 | 2778.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 2833.00 | 2782.77 | 2778.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 2863.40 | 2812.54 | 2794.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2911.90 | 2928.56 | 2893.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 2911.90 | 2928.56 | 2893.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2911.90 | 2928.56 | 2893.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 2905.00 | 2928.56 | 2893.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2917.20 | 2920.70 | 2897.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2910.80 | 2920.70 | 2897.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2906.00 | 2919.01 | 2904.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:45:00 | 2916.00 | 2915.11 | 2907.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 2995.10 | 3049.61 | 3052.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 09:15:00 | 2995.10 | 3049.61 | 3052.08 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 3059.50 | 3041.69 | 3040.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 3089.00 | 3053.58 | 3047.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 3047.30 | 3052.32 | 3047.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 3047.30 | 3052.32 | 3047.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 3047.30 | 3052.32 | 3047.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 3047.30 | 3052.32 | 3047.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 3081.00 | 3058.06 | 3050.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:30:00 | 3095.00 | 3064.97 | 3054.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 2989.20 | 3065.63 | 3060.09 | SL hit (close<static) qty=1.00 sl=3045.70 alert=retest2 |

### Cycle 13 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 3003.10 | 3053.12 | 3054.91 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 3050.00 | 3028.33 | 3026.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 3062.40 | 3035.14 | 3030.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 3046.10 | 3050.50 | 3040.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 09:45:00 | 3046.30 | 3050.50 | 3040.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 3038.00 | 3048.00 | 3040.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:15:00 | 3036.00 | 3048.00 | 3040.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 3012.00 | 3040.80 | 3037.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 3012.00 | 3040.80 | 3037.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 3031.60 | 3038.96 | 3036.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:30:00 | 3035.70 | 3037.73 | 3036.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 3027.00 | 3035.58 | 3035.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 3027.00 | 3035.58 | 3035.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 3014.00 | 3031.27 | 3033.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 3039.00 | 3032.81 | 3034.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 3039.00 | 3032.81 | 3034.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 3039.00 | 3032.81 | 3034.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 3046.20 | 3032.81 | 3034.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 3051.50 | 3036.55 | 3035.80 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 11:15:00 | 3022.60 | 3033.76 | 3034.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 12:15:00 | 3018.20 | 3030.65 | 3033.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 3030.30 | 3027.67 | 3030.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 3030.30 | 3027.67 | 3030.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 3030.30 | 3027.67 | 3030.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 3044.50 | 3027.67 | 3030.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 3059.50 | 3034.04 | 3033.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 3102.90 | 3047.81 | 3039.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 10:15:00 | 3136.00 | 3143.04 | 3114.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 11:00:00 | 3136.00 | 3143.04 | 3114.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 3105.50 | 3139.47 | 3122.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:45:00 | 3121.20 | 3139.47 | 3122.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 3095.00 | 3130.58 | 3119.69 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 3082.80 | 3109.93 | 3112.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 3068.90 | 3101.73 | 3108.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 14:15:00 | 3110.50 | 3098.57 | 3105.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 3110.50 | 3098.57 | 3105.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 3110.50 | 3098.57 | 3105.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 3110.50 | 3098.57 | 3105.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 3120.00 | 3102.85 | 3106.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 3075.00 | 3102.85 | 3106.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 3115.30 | 3105.16 | 3106.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:30:00 | 3112.20 | 3105.16 | 3106.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 3099.70 | 3104.07 | 3106.33 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 3118.50 | 3109.70 | 3108.55 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 3092.00 | 3105.82 | 3106.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 2991.90 | 3056.85 | 3079.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 3026.00 | 3012.49 | 3040.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 3026.00 | 3012.49 | 3040.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3026.00 | 3012.49 | 3040.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:30:00 | 3005.80 | 3011.67 | 3037.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 3005.00 | 3007.05 | 3026.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:15:00 | 3009.50 | 3006.03 | 3021.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:00:00 | 3005.50 | 3005.92 | 3019.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 3001.30 | 3005.00 | 3018.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 3001.30 | 3005.00 | 3018.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 2855.51 | 2988.62 | 3006.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 2859.03 | 2988.62 | 3006.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 2855.22 | 2988.62 | 3006.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2956.50 | 2933.95 | 2962.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 2956.50 | 2933.95 | 2962.53 | SL hit (close>ema200) qty=0.50 sl=2933.95 alert=retest2 |

### Cycle 22 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 3060.40 | 2980.34 | 2976.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 3088.20 | 3001.91 | 2987.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 3022.50 | 3023.35 | 3002.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 3022.50 | 3023.35 | 3002.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 3000.00 | 3017.09 | 3003.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 3000.00 | 3017.09 | 3003.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 3000.00 | 3013.67 | 3002.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:15:00 | 3001.30 | 3013.67 | 3002.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 3000.90 | 3011.12 | 3002.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 3000.90 | 3011.12 | 3002.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 3003.00 | 3009.49 | 3002.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 2939.00 | 3009.49 | 3002.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 2936.40 | 2994.87 | 2996.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 10:15:00 | 2920.10 | 2979.92 | 2989.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 2884.60 | 2874.07 | 2901.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 13:15:00 | 2884.60 | 2874.07 | 2901.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 2884.60 | 2874.07 | 2901.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:30:00 | 2896.00 | 2874.07 | 2901.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 2883.90 | 2876.03 | 2899.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 2896.10 | 2876.03 | 2899.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 2907.10 | 2883.82 | 2897.60 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 2956.10 | 2913.13 | 2908.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 2970.00 | 2924.51 | 2913.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2933.10 | 2973.71 | 2950.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2933.10 | 2973.71 | 2950.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2933.10 | 2973.71 | 2950.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 2933.10 | 2973.71 | 2950.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 2913.20 | 2961.61 | 2947.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 2913.20 | 2961.61 | 2947.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 2855.00 | 2925.31 | 2933.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 2829.90 | 2906.23 | 2923.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 2738.40 | 2716.07 | 2757.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 2738.40 | 2716.07 | 2757.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 2782.60 | 2729.38 | 2759.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 2782.60 | 2729.38 | 2759.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 2799.30 | 2743.36 | 2763.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 2800.40 | 2743.36 | 2763.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 2875.10 | 2778.45 | 2776.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 2900.20 | 2802.80 | 2787.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2857.10 | 2868.97 | 2834.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2857.10 | 2868.97 | 2834.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2857.10 | 2868.97 | 2834.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 10:45:00 | 2892.10 | 2874.47 | 2840.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 2772.00 | 2849.30 | 2843.71 | SL hit (close<static) qty=1.00 sl=2830.30 alert=retest2 |

### Cycle 27 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 2748.40 | 2829.12 | 2835.05 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 2850.80 | 2823.21 | 2821.61 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 2768.50 | 2817.16 | 2819.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 10:15:00 | 2761.10 | 2805.95 | 2814.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 12:15:00 | 2859.80 | 2814.18 | 2816.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 12:15:00 | 2859.80 | 2814.18 | 2816.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 2859.80 | 2814.18 | 2816.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 2859.80 | 2814.18 | 2816.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 2869.40 | 2825.23 | 2821.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 2903.70 | 2858.92 | 2841.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 10:15:00 | 3199.40 | 3213.75 | 3138.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 10:45:00 | 3202.70 | 3213.75 | 3138.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 3226.00 | 3325.55 | 3281.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 3225.00 | 3325.55 | 3281.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 3214.40 | 3303.32 | 3275.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:45:00 | 3212.50 | 3303.32 | 3275.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 13:15:00 | 3213.40 | 3251.48 | 3255.48 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 14:15:00 | 3300.00 | 3252.80 | 3250.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 3318.50 | 3272.69 | 3260.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 3297.60 | 3329.85 | 3311.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 13:15:00 | 3297.60 | 3329.85 | 3311.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 3297.60 | 3329.85 | 3311.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 3297.60 | 3329.85 | 3311.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 3300.20 | 3323.92 | 3310.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:30:00 | 3315.00 | 3323.92 | 3310.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 3291.00 | 3317.33 | 3308.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 3325.00 | 3317.33 | 3308.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 3350.00 | 3395.81 | 3400.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 3350.00 | 3395.81 | 3400.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 3292.90 | 3352.44 | 3375.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 09:15:00 | 3297.10 | 3289.80 | 3316.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 3297.10 | 3289.80 | 3316.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 3297.10 | 3289.80 | 3316.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 3279.50 | 3294.20 | 3303.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 3284.70 | 3293.26 | 3301.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:30:00 | 3285.60 | 3291.29 | 3298.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 13:00:00 | 3284.00 | 3289.83 | 3297.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 3289.80 | 3287.52 | 3294.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 3299.20 | 3287.52 | 3294.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 3280.80 | 3285.45 | 3292.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 3295.20 | 3285.45 | 3292.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 3294.40 | 3284.21 | 3290.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 3294.40 | 3284.21 | 3290.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 3298.70 | 3287.11 | 3291.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:45:00 | 3298.40 | 3287.11 | 3291.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 3296.00 | 3288.89 | 3291.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:00:00 | 3296.00 | 3288.89 | 3291.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 3267.40 | 3284.59 | 3289.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 3245.60 | 3283.07 | 3288.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 3262.20 | 3266.94 | 3278.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 3300.00 | 3283.17 | 3282.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 3300.00 | 3283.17 | 3282.99 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 3250.00 | 3281.84 | 3283.80 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 3303.10 | 3283.54 | 3282.82 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 3275.30 | 3282.05 | 3282.27 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 3295.30 | 3284.70 | 3283.46 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 3260.80 | 3278.61 | 3280.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 3237.40 | 3270.36 | 3276.87 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-08 14:00:00 | 2708.40 | 2026-01-09 10:15:00 | 2663.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2026-01-09 09:45:00 | 2708.00 | 2026-01-09 10:15:00 | 2663.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-01-27 15:00:00 | 2702.50 | 2026-01-28 13:15:00 | 2833.00 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest2 | 2026-02-02 14:45:00 | 2916.00 | 2026-02-09 09:15:00 | 2995.10 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2026-02-12 12:30:00 | 3095.00 | 2026-02-13 09:15:00 | 2989.20 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2026-02-19 13:30:00 | 3035.70 | 2026-02-19 14:15:00 | 3027.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-03-05 10:30:00 | 3005.80 | 2026-03-09 09:15:00 | 2855.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 15:15:00 | 3005.00 | 2026-03-09 09:15:00 | 2859.03 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2026-03-06 11:15:00 | 3009.50 | 2026-03-09 09:15:00 | 2855.22 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2026-03-05 10:30:00 | 3005.80 | 2026-03-10 09:15:00 | 2956.50 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2026-03-05 15:15:00 | 3005.00 | 2026-03-10 09:15:00 | 2956.50 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2026-03-06 11:15:00 | 3009.50 | 2026-03-10 09:15:00 | 2956.50 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2026-03-06 12:00:00 | 3005.50 | 2026-03-10 13:15:00 | 3060.40 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-03-27 10:45:00 | 2892.10 | 2026-03-30 09:15:00 | 2772.00 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2026-04-21 09:15:00 | 3325.00 | 2026-04-23 14:15:00 | 3350.00 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2026-04-29 13:30:00 | 3279.50 | 2026-05-06 09:15:00 | 3300.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-04-30 09:30:00 | 3284.70 | 2026-05-06 09:15:00 | 3300.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-04-30 11:30:00 | 3285.60 | 2026-05-06 09:15:00 | 3300.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-04-30 13:00:00 | 3284.00 | 2026-05-06 09:15:00 | 3300.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-05-05 09:15:00 | 3245.60 | 2026-05-06 09:15:00 | 3300.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-05-05 11:45:00 | 3262.20 | 2026-05-06 09:15:00 | 3300.00 | STOP_HIT | 1.00 | -1.16% |
