# Godfrey Phillips India Ltd. (GODFRYPHLP)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 2424.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 53 |
| ALERT2 | 53 |
| ALERT2_SKIP | 27 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 51 |
| PARTIAL | 14 |
| TARGET_HIT | 4 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 30
- **Target hits / Stop hits / Partials:** 4 / 49 / 14
- **Avg / median % per leg:** 1.40% / 0.59%
- **Sum % (uncompounded):** 93.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 6 | 33.3% | 1 | 16 | 1 | -0.47% | -8.5% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.06% | 8.1% |
| BUY @ 3rd Alert (retest2) | 16 | 4 | 25.0% | 1 | 15 | 0 | -1.04% | -16.6% |
| SELL (all) | 49 | 31 | 63.3% | 3 | 33 | 13 | 2.09% | 102.3% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.55% | 1.5% |
| SELL @ 3rd Alert (retest2) | 48 | 30 | 62.5% | 3 | 32 | 13 | 2.10% | 100.8% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.22% | 9.7% |
| retest2 (combined) | 64 | 34 | 53.1% | 4 | 47 | 13 | 1.32% | 84.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2867.67 | 2798.60 | 2791.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 2985.00 | 2868.50 | 2831.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 2951.33 | 2958.78 | 2905.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:00:00 | 2951.33 | 2958.78 | 2905.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 2933.33 | 2953.69 | 2908.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:45:00 | 2932.50 | 2953.69 | 2908.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 2906.17 | 3005.11 | 2976.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:45:00 | 2906.17 | 3005.11 | 2976.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 2906.17 | 2985.32 | 2969.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 2906.17 | 2985.32 | 2969.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 2906.17 | 2956.83 | 2958.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 09:15:00 | 2789.00 | 2903.48 | 2931.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 2907.33 | 2842.87 | 2876.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 2907.33 | 2842.87 | 2876.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2907.33 | 2842.87 | 2876.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 2907.33 | 2842.87 | 2876.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2905.83 | 2855.46 | 2878.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:45:00 | 2898.00 | 2855.46 | 2878.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 2872.00 | 2873.70 | 2882.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 2869.67 | 2873.70 | 2882.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 2886.67 | 2876.29 | 2882.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 2888.67 | 2876.29 | 2882.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2903.17 | 2881.67 | 2884.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 2905.00 | 2881.67 | 2884.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2873.33 | 2880.00 | 2883.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:15:00 | 2858.00 | 2880.00 | 2883.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 09:15:00 | 2715.10 | 2738.05 | 2760.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 2736.17 | 2734.06 | 2752.83 | SL hit (close>ema200) qty=0.50 sl=2734.06 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 2771.33 | 2762.10 | 2761.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 2822.17 | 2776.23 | 2768.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 2828.33 | 2858.16 | 2824.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 2828.33 | 2858.16 | 2824.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 2828.33 | 2858.16 | 2824.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 2828.33 | 2858.16 | 2824.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 2848.67 | 2856.26 | 2826.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 2881.50 | 2824.15 | 2820.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 2903.67 | 2837.49 | 2827.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 2796.17 | 2829.22 | 2824.20 | SL hit (close<static) qty=1.00 sl=2820.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 2796.17 | 2829.22 | 2824.20 | SL hit (close<static) qty=1.00 sl=2820.50 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 2813.00 | 2820.00 | 2820.51 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 2858.67 | 2822.04 | 2820.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 2885.50 | 2834.73 | 2826.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 11:15:00 | 2812.83 | 2830.35 | 2825.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 11:15:00 | 2812.83 | 2830.35 | 2825.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 2812.83 | 2830.35 | 2825.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 2812.83 | 2830.35 | 2825.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 2820.67 | 2828.42 | 2825.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:15:00 | 2812.83 | 2828.42 | 2825.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 2810.33 | 2824.80 | 2823.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:30:00 | 2811.00 | 2824.80 | 2823.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 2810.33 | 2821.91 | 2822.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 11:15:00 | 2760.50 | 2806.35 | 2814.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 2740.50 | 2739.83 | 2761.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:15:00 | 2760.67 | 2739.83 | 2761.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 2766.50 | 2745.16 | 2761.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:45:00 | 2765.33 | 2745.16 | 2761.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 2767.50 | 2749.63 | 2762.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:30:00 | 2783.00 | 2749.63 | 2762.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 2775.33 | 2754.77 | 2763.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:30:00 | 2775.50 | 2754.77 | 2763.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 2760.17 | 2762.58 | 2764.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:30:00 | 2751.67 | 2759.86 | 2763.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:15:00 | 2752.17 | 2759.49 | 2761.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 2789.83 | 2757.61 | 2758.88 | SL hit (close>static) qty=1.00 sl=2776.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 2789.83 | 2757.61 | 2758.88 | SL hit (close>static) qty=1.00 sl=2776.67 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:30:00 | 2742.50 | 2752.78 | 2756.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:00:00 | 2750.00 | 2742.27 | 2747.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 2750.83 | 2743.98 | 2748.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 2778.17 | 2755.10 | 2752.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 2778.17 | 2755.10 | 2752.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 14:15:00 | 2778.17 | 2755.10 | 2752.48 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 2729.17 | 2752.18 | 2754.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 2720.17 | 2742.04 | 2748.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 2727.00 | 2725.44 | 2736.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 2727.00 | 2725.44 | 2736.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 2727.00 | 2725.44 | 2736.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:45:00 | 2702.83 | 2721.11 | 2732.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 14:30:00 | 2704.83 | 2709.29 | 2723.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 2666.50 | 2676.44 | 2695.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 12:15:00 | 2703.50 | 2680.34 | 2682.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 2701.17 | 2684.51 | 2684.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 2701.17 | 2684.51 | 2684.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 2701.17 | 2684.51 | 2684.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 2701.17 | 2684.51 | 2684.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 2701.17 | 2684.51 | 2684.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 2740.50 | 2704.34 | 2694.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 2988.00 | 3041.78 | 2967.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 2988.00 | 3041.78 | 2967.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 2955.00 | 3024.42 | 2966.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 3029.83 | 3011.36 | 2970.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:00:00 | 3004.67 | 3010.39 | 2976.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 2920.50 | 2963.08 | 2964.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 2920.50 | 2963.08 | 2964.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 2920.50 | 2963.08 | 2964.75 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 3041.17 | 2965.75 | 2960.53 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 2928.33 | 2959.64 | 2960.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 12:15:00 | 2888.33 | 2931.75 | 2946.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 2777.83 | 2767.62 | 2811.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 12:00:00 | 2777.83 | 2767.62 | 2811.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 2808.00 | 2775.70 | 2810.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 2817.00 | 2775.70 | 2810.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2839.83 | 2788.52 | 2813.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 2839.83 | 2788.52 | 2813.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2833.33 | 2797.48 | 2815.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:15:00 | 2836.67 | 2797.48 | 2815.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 2830.33 | 2823.67 | 2823.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 2845.33 | 2828.00 | 2825.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 2802.67 | 2826.44 | 2825.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 2802.67 | 2826.44 | 2825.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 2802.67 | 2826.44 | 2825.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 2802.67 | 2826.44 | 2825.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 2798.33 | 2820.82 | 2823.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 2786.67 | 2810.26 | 2817.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 15:15:00 | 2833.33 | 2812.30 | 2817.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 15:15:00 | 2833.33 | 2812.30 | 2817.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 2833.33 | 2812.30 | 2817.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 2840.50 | 2817.84 | 2819.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 2912.00 | 2836.67 | 2827.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 2978.33 | 2871.11 | 2845.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 3070.17 | 3092.67 | 3019.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:00:00 | 3070.17 | 3092.67 | 3019.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 3076.17 | 3094.41 | 3071.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:45:00 | 3074.50 | 3094.41 | 3071.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 3070.00 | 3089.53 | 3071.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:00:00 | 3070.00 | 3089.53 | 3071.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 3073.33 | 3086.29 | 3071.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 3102.00 | 3086.29 | 3071.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 3052.50 | 3087.18 | 3088.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 3052.50 | 3087.18 | 3088.12 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 3139.67 | 3091.01 | 3089.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 10:15:00 | 3168.33 | 3106.48 | 3096.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 3130.83 | 3136.86 | 3116.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 15:00:00 | 3130.83 | 3136.86 | 3116.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 3123.33 | 3134.12 | 3118.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 3085.33 | 3134.12 | 3118.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 3107.00 | 3128.70 | 3117.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 3107.00 | 3128.70 | 3117.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 3119.00 | 3126.76 | 3117.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:15:00 | 3126.83 | 3126.76 | 3117.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:45:00 | 3129.00 | 3126.91 | 3118.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 13:45:00 | 3130.33 | 3126.42 | 3119.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 14:30:00 | 3127.33 | 3126.51 | 3119.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 3059.33 | 3113.68 | 3115.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 3059.33 | 3113.68 | 3115.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 3059.33 | 3113.68 | 3115.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 3059.33 | 3113.68 | 3115.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 3059.33 | 3113.68 | 3115.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 3050.00 | 3100.95 | 3109.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 2913.33 | 2910.10 | 2942.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 10:30:00 | 2919.67 | 2910.10 | 2942.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2958.50 | 2921.87 | 2942.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 2958.50 | 2921.87 | 2942.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2955.67 | 2928.63 | 2943.76 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 3130.00 | 2984.29 | 2966.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 3155.67 | 3042.78 | 2997.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 3090.83 | 3097.03 | 3046.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 10:00:00 | 3090.83 | 3097.03 | 3046.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 3040.00 | 3077.79 | 3049.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 3040.00 | 3077.79 | 3049.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 3044.33 | 3071.09 | 3049.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 3045.67 | 3071.09 | 3049.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 3042.83 | 3065.44 | 3048.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 3042.83 | 3065.44 | 3048.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 3046.33 | 3061.62 | 3048.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 3014.33 | 3061.62 | 3048.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 3029.50 | 3055.20 | 3046.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 2998.50 | 3055.20 | 3046.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 2964.67 | 3037.09 | 3039.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 2931.17 | 3015.91 | 3029.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 2963.17 | 2955.62 | 2982.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 2967.17 | 2955.62 | 2982.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 2986.17 | 2963.23 | 2981.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 2986.17 | 2963.23 | 2981.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 3033.33 | 2977.25 | 2985.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 3273.33 | 2977.25 | 2985.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 3272.67 | 3036.34 | 3011.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 3466.67 | 3273.01 | 3163.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 3395.00 | 3496.76 | 3360.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:00:00 | 3395.00 | 3496.76 | 3360.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 3347.67 | 3466.94 | 3359.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 11:30:00 | 3408.83 | 3448.89 | 3360.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 3454.33 | 3385.08 | 3359.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 3315.00 | 3362.13 | 3356.15 | SL hit (close<static) qty=1.00 sl=3316.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 3315.00 | 3362.13 | 3356.15 | SL hit (close<static) qty=1.00 sl=3316.67 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 3325.00 | 3347.69 | 3350.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 10:15:00 | 3289.83 | 3336.12 | 3344.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 3247.83 | 3238.58 | 3270.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:15:00 | 3278.50 | 3238.58 | 3270.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 3279.67 | 3246.80 | 3271.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 3308.50 | 3246.80 | 3271.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 3258.67 | 3249.17 | 3270.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 12:15:00 | 3250.00 | 3249.17 | 3270.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 12:15:00 | 3287.33 | 3256.80 | 3271.72 | SL hit (close>static) qty=1.00 sl=3284.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 3337.50 | 3287.21 | 3283.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 3360.17 | 3326.36 | 3305.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 09:15:00 | 3228.50 | 3336.57 | 3323.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 3228.50 | 3336.57 | 3323.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 3228.50 | 3336.57 | 3323.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 3226.67 | 3336.57 | 3323.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 3235.83 | 3316.42 | 3315.29 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 3199.17 | 3292.97 | 3304.73 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 3533.17 | 3311.60 | 3290.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 3607.83 | 3370.85 | 3318.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 3639.00 | 3639.77 | 3538.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 3639.00 | 3639.77 | 3538.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 3566.33 | 3639.91 | 3593.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:30:00 | 3553.50 | 3639.91 | 3593.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 3523.33 | 3616.59 | 3586.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 3523.33 | 3616.59 | 3586.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 3487.67 | 3553.90 | 3562.62 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 3603.83 | 3568.11 | 3567.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 3614.83 | 3577.46 | 3572.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 3541.67 | 3582.53 | 3576.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 3541.67 | 3582.53 | 3576.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3541.67 | 3582.53 | 3576.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:00:00 | 3541.67 | 3582.53 | 3576.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 3556.67 | 3577.36 | 3575.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 3555.83 | 3577.36 | 3575.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 3548.17 | 3573.25 | 3573.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 3506.67 | 3555.81 | 3565.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 3613.33 | 3559.32 | 3564.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 3613.33 | 3559.32 | 3564.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 3613.33 | 3559.32 | 3564.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 3613.33 | 3559.32 | 3564.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 10:15:00 | 3680.67 | 3583.59 | 3575.41 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 3524.00 | 3574.84 | 3575.23 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 3665.83 | 3584.01 | 3578.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 3710.00 | 3609.21 | 3590.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 09:15:00 | 3641.67 | 3641.92 | 3617.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 3641.67 | 3641.92 | 3617.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 3641.67 | 3641.92 | 3617.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 3641.67 | 3641.92 | 3617.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 3558.00 | 3624.08 | 3613.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 3558.00 | 3624.08 | 3613.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 3557.00 | 3610.66 | 3608.08 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 13:15:00 | 3541.67 | 3596.87 | 3602.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 14:15:00 | 3524.67 | 3582.43 | 3595.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 3422.67 | 3412.24 | 3478.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 3523.67 | 3418.43 | 3444.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 3523.67 | 3418.43 | 3444.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:45:00 | 3530.67 | 3418.43 | 3444.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 3479.67 | 3430.68 | 3447.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:45:00 | 3468.33 | 3438.54 | 3449.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:45:00 | 3476.33 | 3443.03 | 3450.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 3496.67 | 3459.14 | 3456.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 3496.67 | 3459.14 | 3456.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 3496.67 | 3459.14 | 3456.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 3632.33 | 3493.78 | 3472.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 09:15:00 | 3629.00 | 3629.09 | 3566.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:45:00 | 3634.00 | 3629.09 | 3566.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 3561.67 | 3606.73 | 3571.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 3561.67 | 3606.73 | 3571.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 3550.00 | 3595.38 | 3569.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:30:00 | 3548.33 | 3595.38 | 3569.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 3547.00 | 3577.91 | 3565.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 3625.00 | 3577.91 | 3565.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3603.67 | 3583.06 | 3568.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:45:00 | 3661.00 | 3614.80 | 3596.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 3560.00 | 3590.11 | 3590.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 3560.00 | 3590.11 | 3590.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 3515.00 | 3575.09 | 3583.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 15:15:00 | 3413.33 | 3412.09 | 3467.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:15:00 | 3432.33 | 3412.09 | 3467.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 3408.33 | 3411.34 | 3462.38 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 3670.00 | 3503.03 | 3480.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 3900.00 | 3610.74 | 3535.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 3628.00 | 3632.90 | 3571.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 3628.00 | 3632.90 | 3571.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 3628.00 | 3632.90 | 3571.94 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 3518.00 | 3566.04 | 3566.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 3509.00 | 3554.63 | 3561.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 3544.00 | 3491.24 | 3511.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 3544.00 | 3491.24 | 3511.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 3544.00 | 3491.24 | 3511.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 3544.00 | 3491.24 | 3511.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 3498.00 | 3492.59 | 3510.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 3460.00 | 3490.60 | 3504.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3287.00 | 3349.23 | 3375.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 3366.00 | 3257.62 | 3299.62 | SL hit (close>ema200) qty=0.50 sl=3257.62 alert=retest2 |

### Cycle 37 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 3520.00 | 3342.00 | 3332.66 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 3385.00 | 3409.09 | 3412.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 3346.00 | 3386.15 | 3398.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 3344.00 | 3328.07 | 3346.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 3344.00 | 3328.07 | 3346.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 3344.00 | 3328.07 | 3346.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 3344.00 | 3328.07 | 3346.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 3338.00 | 3330.06 | 3345.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 3305.00 | 3330.06 | 3345.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:00:00 | 3325.10 | 3304.05 | 3319.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 3396.80 | 3316.74 | 3317.33 | SL hit (close>static) qty=1.00 sl=3346.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 3396.80 | 3316.74 | 3317.33 | SL hit (close>static) qty=1.00 sl=3346.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 3422.30 | 3337.86 | 3326.87 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 3316.00 | 3334.47 | 3335.73 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 3399.40 | 3339.61 | 3336.70 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 3231.80 | 3367.33 | 3373.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 10:15:00 | 3143.50 | 3322.57 | 3352.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 11:15:00 | 3142.60 | 3137.70 | 3187.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 12:15:00 | 3162.00 | 3137.70 | 3187.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3172.80 | 3154.49 | 3177.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 3164.00 | 3159.51 | 3175.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 3165.90 | 3146.51 | 3153.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:00:00 | 3157.60 | 3146.99 | 3149.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 3005.80 | 3091.46 | 3107.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 3007.61 | 3091.46 | 3107.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 2999.72 | 3091.46 | 3107.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 3044.30 | 3023.99 | 3051.66 | SL hit (close>ema200) qty=0.50 sl=3023.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 3044.30 | 3023.99 | 3051.66 | SL hit (close>ema200) qty=0.50 sl=3023.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 3044.30 | 3023.99 | 3051.66 | SL hit (close>ema200) qty=0.50 sl=3023.99 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 3028.40 | 3019.30 | 3019.12 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 3011.10 | 3017.77 | 3018.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 2998.90 | 3012.16 | 3015.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 2936.80 | 2909.16 | 2927.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 2936.80 | 2909.16 | 2927.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2936.80 | 2909.16 | 2927.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 2936.80 | 2909.16 | 2927.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 2938.60 | 2915.04 | 2928.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 2943.00 | 2915.04 | 2928.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 2934.40 | 2922.75 | 2929.79 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 2962.80 | 2937.60 | 2935.13 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 2903.70 | 2935.57 | 2936.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 2889.00 | 2926.26 | 2932.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 2901.00 | 2880.13 | 2896.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 2901.00 | 2880.13 | 2896.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 2901.00 | 2880.13 | 2896.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 2901.00 | 2880.13 | 2896.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 2900.00 | 2884.10 | 2897.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 2878.00 | 2884.10 | 2897.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 2909.00 | 2892.17 | 2890.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 2909.00 | 2892.17 | 2890.35 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 2875.50 | 2899.58 | 2899.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 2873.10 | 2889.30 | 2894.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 12:15:00 | 2863.80 | 2861.35 | 2876.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 12:15:00 | 2863.80 | 2861.35 | 2876.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 2863.80 | 2861.35 | 2876.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:45:00 | 2863.30 | 2861.35 | 2876.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 2872.80 | 2863.64 | 2876.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 2874.60 | 2863.64 | 2876.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 2792.50 | 2800.95 | 2828.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 2773.10 | 2793.02 | 2804.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 2634.44 | 2665.61 | 2710.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 2673.10 | 2667.10 | 2707.16 | SL hit (close>ema200) qty=0.50 sl=2667.10 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 2768.00 | 2717.83 | 2720.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 2768.00 | 2727.86 | 2725.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 2768.00 | 2727.86 | 2725.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 2878.50 | 2757.99 | 2739.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 2799.00 | 2814.31 | 2782.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:15:00 | 2750.40 | 2814.31 | 2782.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 2726.60 | 2796.77 | 2777.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 2720.00 | 2796.77 | 2777.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 2724.70 | 2782.35 | 2772.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:15:00 | 2756.60 | 2782.35 | 2772.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 13:15:00 | 2801.00 | 2835.65 | 2839.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 2801.00 | 2835.65 | 2839.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 2792.10 | 2822.00 | 2832.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 2787.40 | 2777.63 | 2796.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 11:00:00 | 2787.40 | 2777.63 | 2796.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2793.10 | 2780.73 | 2795.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 2793.10 | 2780.73 | 2795.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 2780.00 | 2780.58 | 2794.40 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 2863.30 | 2812.10 | 2806.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 2873.20 | 2847.99 | 2828.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 2852.00 | 2857.38 | 2842.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:45:00 | 2850.60 | 2857.38 | 2842.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 2853.50 | 2856.60 | 2843.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 2848.20 | 2856.60 | 2843.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 2849.40 | 2853.78 | 2844.56 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 2823.70 | 2837.29 | 2838.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 2809.90 | 2831.82 | 2836.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 2831.60 | 2826.56 | 2832.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 10:15:00 | 2831.60 | 2826.56 | 2832.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 2831.60 | 2826.56 | 2832.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 2841.10 | 2826.56 | 2832.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2833.20 | 2827.89 | 2832.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 2836.40 | 2827.89 | 2832.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 2834.80 | 2829.27 | 2832.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 2834.80 | 2829.27 | 2832.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 2837.80 | 2830.98 | 2832.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 2837.80 | 2830.98 | 2832.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 2821.30 | 2829.04 | 2831.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 2846.60 | 2829.04 | 2831.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 2824.90 | 2827.63 | 2830.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 2828.80 | 2827.63 | 2830.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2734.20 | 2789.77 | 2808.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 2713.60 | 2772.22 | 2798.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 2713.10 | 2710.55 | 2750.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 2572.00 | 2744.13 | 2752.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:15:00 | 2577.92 | 2694.36 | 2728.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:15:00 | 2577.44 | 2694.36 | 2728.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-01 10:15:00 | 2442.24 | 2645.05 | 2703.28 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-01 10:15:00 | 2441.79 | 2645.05 | 2703.28 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 10:15:00 | 2443.40 | 2645.05 | 2703.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-01 11:15:00 | 2314.80 | 2574.96 | 2666.13 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 2221.20 | 2152.54 | 2143.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 2252.80 | 2222.66 | 2199.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 2237.00 | 2240.07 | 2220.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 2215.30 | 2240.07 | 2220.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2224.60 | 2236.97 | 2220.75 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 2192.20 | 2210.90 | 2212.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 2184.90 | 2202.79 | 2208.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2106.20 | 2093.01 | 2126.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 13:15:00 | 2056.60 | 2090.68 | 2116.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 2016.50 | 2017.82 | 2036.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 2005.90 | 2016.12 | 2027.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:00:00 | 2006.30 | 2014.16 | 2025.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 2024.80 | 2015.43 | 2021.23 | SL hit (close>ema400) qty=1.00 sl=2021.23 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 2069.00 | 2026.14 | 2025.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 2069.00 | 2026.14 | 2025.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 2069.00 | 2026.14 | 2025.58 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 2006.10 | 2029.55 | 2031.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2000.70 | 2023.78 | 2028.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 1940.00 | 1933.21 | 1965.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 1956.40 | 1933.21 | 1965.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1934.60 | 1933.49 | 1962.84 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 2015.10 | 1973.70 | 1971.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 2029.40 | 2000.61 | 1986.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1994.70 | 2011.28 | 1995.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1994.70 | 2011.28 | 1995.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1994.70 | 2011.28 | 1995.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 1994.70 | 2011.28 | 1995.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1978.20 | 2004.67 | 1994.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 1978.20 | 2004.67 | 1994.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1978.40 | 1999.41 | 1992.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:45:00 | 1976.20 | 1999.41 | 1992.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 1983.40 | 1989.32 | 1989.45 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 2103.90 | 2012.23 | 1999.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 10:15:00 | 2185.00 | 2046.79 | 2016.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 2140.00 | 2178.10 | 2157.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 2140.00 | 2178.10 | 2157.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2140.00 | 2178.10 | 2157.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 2140.00 | 2178.10 | 2157.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 2133.00 | 2169.08 | 2155.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 2133.00 | 2169.08 | 2155.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 2134.00 | 2145.78 | 2147.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 2104.20 | 2137.47 | 2143.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 2126.00 | 2122.52 | 2132.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 13:30:00 | 2126.90 | 2122.52 | 2132.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 2105.00 | 2049.92 | 2071.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 2105.00 | 2049.92 | 2071.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 2085.00 | 2056.94 | 2072.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:00:00 | 2059.70 | 2064.87 | 2072.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 2302.90 | 2111.85 | 2092.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 2302.90 | 2111.85 | 2092.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 2329.90 | 2185.93 | 2131.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 2415.00 | 2472.95 | 2377.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:30:00 | 2413.90 | 2472.95 | 2377.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2314.80 | 2428.31 | 2399.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 2284.00 | 2428.31 | 2399.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 2295.10 | 2401.67 | 2389.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 2300.00 | 2401.67 | 2389.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 2275.70 | 2376.48 | 2379.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 2224.80 | 2329.89 | 2356.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 11:15:00 | 2135.70 | 2096.79 | 2123.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 11:15:00 | 2135.70 | 2096.79 | 2123.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 2135.70 | 2096.79 | 2123.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 2135.70 | 2096.79 | 2123.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 2119.90 | 2101.42 | 2122.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 2125.70 | 2101.42 | 2122.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 2118.20 | 2104.77 | 2122.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 14:30:00 | 2112.00 | 2106.24 | 2121.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 2112.10 | 2106.24 | 2121.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2056.00 | 2108.79 | 2121.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2006.40 | 2101.47 | 2116.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2006.49 | 2101.47 | 2116.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 2017.50 | 2016.55 | 2044.20 | SL hit (close>ema200) qty=0.50 sl=2016.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 2017.50 | 2016.55 | 2044.20 | SL hit (close>ema200) qty=0.50 sl=2016.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 2070.50 | 2046.89 | 2044.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 2070.50 | 2046.89 | 2044.28 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1987.00 | 2041.63 | 2044.32 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 2054.00 | 2033.17 | 2030.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 2063.90 | 2042.52 | 2035.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 2074.90 | 2095.03 | 2075.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 2074.90 | 2095.03 | 2075.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2074.90 | 2095.03 | 2075.07 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 2033.10 | 2074.08 | 2075.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 2025.50 | 2058.35 | 2067.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 2018.40 | 2008.77 | 2029.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 2018.40 | 2008.77 | 2029.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 2018.40 | 2008.77 | 2029.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 2028.60 | 2008.77 | 2029.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2031.30 | 2015.07 | 2028.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:00:00 | 2015.50 | 2015.16 | 2027.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 2010.50 | 2015.05 | 2023.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 2039.00 | 2028.37 | 2027.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 2039.00 | 2028.37 | 2027.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 2039.00 | 2028.37 | 2027.05 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 2012.30 | 2024.97 | 2026.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1992.60 | 2014.58 | 2020.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2017.40 | 2010.35 | 2016.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2017.40 | 2010.35 | 2016.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2017.40 | 2010.35 | 2016.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 2002.00 | 2011.09 | 2016.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 1998.60 | 2007.70 | 2013.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:15:00 | 1901.90 | 1951.53 | 1982.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:15:00 | 1898.67 | 1951.53 | 1982.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 1895.70 | 1882.13 | 1918.92 | SL hit (close>ema200) qty=0.50 sl=1882.13 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 1895.70 | 1882.13 | 1918.92 | SL hit (close>ema200) qty=0.50 sl=1882.13 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2063.50 | 1942.68 | 1934.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 2126.90 | 1979.53 | 1952.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 2035.00 | 2036.53 | 1993.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 2006.30 | 2036.53 | 1993.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1995.60 | 2028.34 | 1993.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1995.60 | 2028.34 | 1993.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1961.50 | 2014.98 | 1990.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1961.50 | 2014.98 | 1990.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1935.30 | 1999.04 | 1985.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 1935.30 | 1999.04 | 1985.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1904.00 | 1963.73 | 1971.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1890.10 | 1932.70 | 1953.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1969.20 | 1913.22 | 1931.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1969.20 | 1913.22 | 1931.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1969.20 | 1913.22 | 1931.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1965.50 | 1913.22 | 1931.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1948.80 | 1920.34 | 1932.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1941.20 | 1935.11 | 1937.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 15:15:00 | 1929.80 | 1920.04 | 1919.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 1929.80 | 1920.04 | 1919.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 11:15:00 | 1935.30 | 1924.29 | 1921.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 1927.00 | 1927.99 | 1924.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1986.70 | 1927.99 | 1924.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:15:00 | 2086.04 | 1998.88 | 1971.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2048.50 | 2072.20 | 2044.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 2048.50 | 2072.20 | 2044.70 | SL hit (close<ema200) qty=0.50 sl=2072.20 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2063.90 | 2069.56 | 2046.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 2102.30 | 2054.94 | 2046.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 2133.30 | 2149.00 | 2149.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 2133.30 | 2149.00 | 2149.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 2133.30 | 2149.00 | 2149.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 2110.60 | 2141.32 | 2146.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 2163.20 | 2135.26 | 2140.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 2163.20 | 2135.26 | 2140.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 2163.20 | 2135.26 | 2140.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 2163.20 | 2135.26 | 2140.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 2165.90 | 2141.39 | 2142.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 2177.20 | 2141.39 | 2142.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 2182.70 | 2149.65 | 2146.39 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 2112.90 | 2146.56 | 2150.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 2105.60 | 2133.00 | 2143.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 2135.70 | 2114.22 | 2125.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 2135.70 | 2114.22 | 2125.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 2135.70 | 2114.22 | 2125.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 2135.70 | 2114.22 | 2125.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 2127.00 | 2116.78 | 2126.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:30:00 | 2130.00 | 2116.78 | 2126.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 2112.80 | 2115.98 | 2124.83 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 2174.90 | 2133.45 | 2128.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 2219.10 | 2150.58 | 2136.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 2231.90 | 2245.64 | 2221.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 12:00:00 | 2231.90 | 2245.64 | 2221.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 2223.80 | 2241.27 | 2221.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:45:00 | 2224.00 | 2241.27 | 2221.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 2216.70 | 2236.35 | 2221.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 2218.40 | 2236.35 | 2221.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 2218.00 | 2232.68 | 2220.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:30:00 | 2212.70 | 2232.68 | 2220.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 2204.60 | 2224.72 | 2219.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 2227.00 | 2218.49 | 2217.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 11:15:00 | 2449.70 | 2378.25 | 2331.68 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 11:15:00 | 2858.00 | 2025-05-28 09:15:00 | 2715.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-21 11:15:00 | 2858.00 | 2025-05-28 12:15:00 | 2736.17 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest2 | 2025-06-03 09:45:00 | 2881.50 | 2025-06-03 11:15:00 | 2796.17 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-06-03 10:45:00 | 2903.67 | 2025-06-03 11:15:00 | 2796.17 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-06-10 11:30:00 | 2751.67 | 2025-06-12 09:15:00 | 2789.83 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-06-11 12:15:00 | 2752.17 | 2025-06-12 09:15:00 | 2789.83 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-06-12 11:30:00 | 2742.50 | 2025-06-13 14:15:00 | 2778.17 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-13 11:00:00 | 2750.00 | 2025-06-13 14:15:00 | 2778.17 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-06-19 11:45:00 | 2702.83 | 2025-06-24 12:15:00 | 2701.17 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-06-19 14:30:00 | 2704.83 | 2025-06-24 12:15:00 | 2701.17 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-06-20 15:00:00 | 2666.50 | 2025-06-24 12:15:00 | 2701.17 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-24 12:15:00 | 2703.50 | 2025-06-24 12:15:00 | 2701.17 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-07-01 09:15:00 | 3029.83 | 2025-07-02 09:15:00 | 2920.50 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2025-07-01 11:00:00 | 3004.67 | 2025-07-02 09:15:00 | 2920.50 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-07-18 09:15:00 | 3102.00 | 2025-07-21 13:15:00 | 3052.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-07-23 12:15:00 | 3126.83 | 2025-07-24 09:15:00 | 3059.33 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-07-23 12:45:00 | 3129.00 | 2025-07-24 09:15:00 | 3059.33 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-07-23 13:45:00 | 3130.33 | 2025-07-24 09:15:00 | 3059.33 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-07-23 14:30:00 | 3127.33 | 2025-07-24 09:15:00 | 3059.33 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-08-07 11:30:00 | 3408.83 | 2025-08-08 14:15:00 | 3315.00 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-08-08 10:45:00 | 3454.33 | 2025-08-08 14:15:00 | 3315.00 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-08-13 12:15:00 | 3250.00 | 2025-08-13 12:15:00 | 3287.33 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-04 11:45:00 | 3468.33 | 2025-09-04 15:15:00 | 3496.67 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-04 12:45:00 | 3476.33 | 2025-09-04 15:15:00 | 3496.67 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-10 10:45:00 | 3661.00 | 2025-09-11 09:15:00 | 3560.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-09-22 13:45:00 | 3460.00 | 2025-09-26 09:15:00 | 3287.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 3460.00 | 2025-09-29 12:15:00 | 3366.00 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest2 | 2025-10-13 09:15:00 | 3305.00 | 2025-10-15 09:15:00 | 3396.80 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-10-14 10:00:00 | 3325.10 | 2025-10-15 09:15:00 | 3396.80 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-10-28 11:45:00 | 3164.00 | 2025-11-04 09:15:00 | 3005.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3165.90 | 2025-11-04 09:15:00 | 3007.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:00:00 | 3157.60 | 2025-11-04 09:15:00 | 2999.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 11:45:00 | 3164.00 | 2025-11-06 11:15:00 | 3044.30 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3165.90 | 2025-11-06 11:15:00 | 3044.30 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-10-31 10:00:00 | 3157.60 | 2025-11-06 11:15:00 | 3044.30 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-11-25 09:15:00 | 2878.00 | 2025-11-27 09:15:00 | 2909.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-12-04 15:15:00 | 2773.10 | 2025-12-09 09:15:00 | 2634.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 15:15:00 | 2773.10 | 2025-12-09 10:15:00 | 2673.10 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-12-09 15:15:00 | 2768.00 | 2025-12-09 15:15:00 | 2768.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-12-11 11:15:00 | 2756.60 | 2025-12-17 13:15:00 | 2801.00 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2025-12-30 10:45:00 | 2713.60 | 2026-01-01 09:15:00 | 2577.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 09:45:00 | 2713.10 | 2026-01-01 09:15:00 | 2577.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 10:45:00 | 2713.60 | 2026-01-01 10:15:00 | 2442.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 09:45:00 | 2713.10 | 2026-01-01 10:15:00 | 2441.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 09:15:00 | 2572.00 | 2026-01-01 10:15:00 | 2443.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:15:00 | 2572.00 | 2026-01-01 11:15:00 | 2314.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-01-22 13:15:00 | 2056.60 | 2026-01-29 15:15:00 | 2024.80 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2026-01-29 09:30:00 | 2005.90 | 2026-01-30 09:15:00 | 2069.00 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2026-01-29 11:00:00 | 2006.30 | 2026-01-30 09:15:00 | 2069.00 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-02-17 15:00:00 | 2059.70 | 2026-02-18 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -11.81% |
| SELL | retest2 | 2026-02-27 14:30:00 | 2112.00 | 2026-03-02 09:15:00 | 2006.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 2112.10 | 2026-03-02 09:15:00 | 2006.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 14:30:00 | 2112.00 | 2026-03-05 09:15:00 | 2017.50 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2026-02-27 15:00:00 | 2112.10 | 2026-03-05 09:15:00 | 2017.50 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2056.00 | 2026-03-06 11:15:00 | 2070.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-03-17 11:00:00 | 2015.50 | 2026-03-18 13:15:00 | 2039.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-03-17 15:00:00 | 2010.50 | 2026-03-18 13:15:00 | 2039.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-03-20 12:15:00 | 2002.00 | 2026-03-23 11:15:00 | 1901.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1998.60 | 2026-03-23 11:15:00 | 1898.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 2002.00 | 2026-03-24 12:15:00 | 1895.70 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1998.60 | 2026-03-24 12:15:00 | 1895.70 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2026-04-01 14:15:00 | 1941.20 | 2026-04-06 15:15:00 | 1929.80 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest1 | 2026-04-08 09:15:00 | 1986.70 | 2026-04-09 09:15:00 | 2086.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-08 09:15:00 | 1986.70 | 2026-04-13 09:15:00 | 2048.50 | STOP_HIT | 0.50 | 3.11% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2063.90 | 2026-04-21 11:15:00 | 2133.30 | STOP_HIT | 1.00 | 3.36% |
| BUY | retest2 | 2026-04-15 09:15:00 | 2102.30 | 2026-04-21 11:15:00 | 2133.30 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2026-05-05 14:15:00 | 2227.00 | 2026-05-08 11:15:00 | 2449.70 | TARGET_HIT | 1.00 | 10.00% |
