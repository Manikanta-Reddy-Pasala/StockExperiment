# Waaree Energies Ltd. (WAAREEENER)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 3212.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 51 |
| ALERT2 | 49 |
| ALERT2_SKIP | 24 |
| ALERT3 | 119 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 46 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 30
- **Target hits / Stop hits / Partials:** 6 / 40 / 6
- **Avg / median % per leg:** 1.00% / -0.37%
- **Sum % (uncompounded):** 52.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 8 | 33.3% | 3 | 21 | 0 | -0.15% | -3.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 8 | 33.3% | 3 | 21 | 0 | -0.15% | -3.6% |
| SELL (all) | 28 | 14 | 50.0% | 3 | 19 | 6 | 1.99% | 55.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 14 | 50.0% | 3 | 19 | 6 | 1.99% | 55.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 22 | 42.3% | 6 | 40 | 6 | 1.00% | 52.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 2682.70 | 2632.59 | 2629.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 2708.40 | 2672.87 | 2653.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 2680.00 | 2683.11 | 2663.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 2680.00 | 2683.11 | 2663.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 2679.90 | 2682.47 | 2665.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:30:00 | 2678.00 | 2682.47 | 2665.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2960.80 | 2911.90 | 2859.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 2875.00 | 2911.90 | 2859.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2920.50 | 2954.22 | 2916.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 2933.70 | 2954.22 | 2916.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 2926.50 | 2948.68 | 2917.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 2919.70 | 2948.68 | 2917.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 2914.00 | 2941.74 | 2916.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 2914.00 | 2941.74 | 2916.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 2891.20 | 2931.63 | 2914.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 2891.20 | 2931.63 | 2914.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 2917.30 | 2928.77 | 2914.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 2937.60 | 2928.17 | 2916.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 2940.00 | 2935.05 | 2922.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 2696.60 | 2929.64 | 2941.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 2696.60 | 2929.64 | 2941.89 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 2961.90 | 2845.83 | 2830.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 3027.10 | 2932.87 | 2880.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 12:15:00 | 2967.60 | 2978.52 | 2927.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 13:00:00 | 2967.60 | 2978.52 | 2927.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 2946.90 | 2971.53 | 2940.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:15:00 | 2920.70 | 2971.53 | 2940.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 2935.00 | 2964.23 | 2940.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:00:00 | 2955.00 | 2962.38 | 2941.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 2898.90 | 2933.54 | 2934.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 2898.90 | 2933.54 | 2934.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 15:15:00 | 2881.00 | 2893.03 | 2904.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 2938.00 | 2902.03 | 2907.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 2938.00 | 2902.03 | 2907.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 2938.00 | 2902.03 | 2907.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 2938.00 | 2902.03 | 2907.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 2908.40 | 2903.30 | 2908.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:00:00 | 2893.10 | 2903.13 | 2907.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 2862.20 | 2840.30 | 2839.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 2862.20 | 2840.30 | 2839.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 10:15:00 | 2910.00 | 2854.24 | 2846.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 2868.00 | 2877.17 | 2860.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 2868.00 | 2877.17 | 2860.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 2868.00 | 2877.17 | 2860.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 2874.80 | 2877.17 | 2860.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 2891.60 | 2880.05 | 2863.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:30:00 | 2901.20 | 2878.48 | 2868.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 13:00:00 | 2897.00 | 2882.18 | 2870.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 2832.10 | 2872.16 | 2867.20 | SL hit (close<static) qty=1.00 sl=2859.70 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 2834.00 | 2859.85 | 2862.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 2812.90 | 2850.46 | 2857.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 2831.00 | 2822.12 | 2838.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 2831.00 | 2822.12 | 2838.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 2831.00 | 2822.12 | 2838.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 2831.00 | 2822.12 | 2838.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 2822.10 | 2822.12 | 2836.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 2799.90 | 2822.12 | 2836.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 2791.00 | 2815.89 | 2832.63 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 2889.70 | 2837.98 | 2836.82 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 2812.90 | 2837.01 | 2838.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 2793.00 | 2824.48 | 2832.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 2809.90 | 2710.57 | 2731.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 2809.90 | 2710.57 | 2731.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 2809.90 | 2710.57 | 2731.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 2809.90 | 2710.57 | 2731.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 2823.10 | 2733.08 | 2740.19 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 2875.00 | 2761.46 | 2752.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 13:15:00 | 2917.00 | 2812.77 | 2778.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 15:15:00 | 2921.90 | 2928.07 | 2878.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 09:30:00 | 2903.70 | 2931.58 | 2884.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 2958.10 | 2976.93 | 2945.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:45:00 | 2947.40 | 2976.93 | 2945.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 2939.50 | 2966.90 | 2946.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 2939.50 | 2966.90 | 2946.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 2942.00 | 2961.92 | 2946.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 2939.00 | 2961.92 | 2946.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 2927.70 | 2955.07 | 2944.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 2969.80 | 2944.76 | 2942.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 2970.80 | 2955.79 | 2948.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 3015.50 | 2949.50 | 2947.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 12:15:00 | 2971.00 | 3023.85 | 3030.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 2971.00 | 3023.85 | 3030.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 2962.00 | 3002.99 | 3019.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 2992.60 | 2990.12 | 3008.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 11:00:00 | 2992.60 | 2990.12 | 3008.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 3005.90 | 2993.27 | 3008.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 3005.90 | 2993.27 | 3008.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 3003.10 | 2995.24 | 3007.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:45:00 | 3008.00 | 2995.24 | 3007.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 3003.40 | 2996.87 | 3007.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 3008.00 | 2996.87 | 3007.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 2996.40 | 2996.78 | 3006.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 2982.30 | 2993.57 | 3001.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:00:00 | 2977.70 | 2982.73 | 2992.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 2983.10 | 2966.31 | 2975.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 3033.60 | 2979.77 | 2980.33 | SL hit (close>static) qty=1.00 sl=3008.60 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 3036.00 | 2991.02 | 2985.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 3072.00 | 3012.54 | 2996.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 3130.10 | 3142.67 | 3109.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 3130.10 | 3142.67 | 3109.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 3098.50 | 3133.83 | 3108.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 3098.50 | 3133.83 | 3108.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 3071.50 | 3121.37 | 3104.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 3071.50 | 3121.37 | 3104.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 3111.20 | 3113.07 | 3104.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:45:00 | 3107.90 | 3113.07 | 3104.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 3165.00 | 3122.77 | 3110.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 11:00:00 | 3171.60 | 3132.54 | 3115.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 3178.00 | 3139.03 | 3120.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 3183.00 | 3220.01 | 3223.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 3183.00 | 3220.01 | 3223.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 3166.00 | 3199.80 | 3212.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 3114.10 | 3106.67 | 3139.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 3114.10 | 3106.67 | 3139.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3114.10 | 3106.67 | 3139.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 3124.50 | 3106.67 | 3139.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 3119.70 | 3102.24 | 3117.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 3119.70 | 3102.24 | 3117.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 3156.20 | 3113.03 | 3121.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 3156.20 | 3113.03 | 3121.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 3178.80 | 3126.19 | 3126.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 3183.00 | 3126.19 | 3126.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 3188.20 | 3138.59 | 3132.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 3221.70 | 3163.12 | 3144.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 3196.60 | 3215.94 | 3190.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 11:00:00 | 3196.60 | 3215.94 | 3190.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 3180.00 | 3208.75 | 3189.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 3180.00 | 3208.75 | 3189.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 3184.90 | 3203.98 | 3188.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:15:00 | 3200.00 | 3203.98 | 3188.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 14:45:00 | 3199.90 | 3198.47 | 3188.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 3206.00 | 3196.80 | 3188.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 3170.20 | 3191.48 | 3187.20 | SL hit (close<static) qty=1.00 sl=3172.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 3138.50 | 3180.88 | 3182.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 3125.00 | 3169.71 | 3177.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 3230.70 | 3150.93 | 3161.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 3230.70 | 3150.93 | 3161.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 3230.70 | 3150.93 | 3161.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:15:00 | 3272.00 | 3150.93 | 3161.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 3247.00 | 3170.15 | 3169.15 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 3126.20 | 3167.00 | 3172.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 3011.20 | 3129.78 | 3153.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 3043.50 | 3035.49 | 3082.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 10:15:00 | 3060.60 | 3035.49 | 3082.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 3089.00 | 3049.06 | 3080.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 3089.00 | 3049.06 | 3080.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 3139.20 | 3067.09 | 3086.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:30:00 | 3155.80 | 3067.09 | 3086.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 3129.40 | 3079.55 | 3090.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:45:00 | 3140.00 | 3079.55 | 3090.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 3109.80 | 3087.21 | 3091.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 3115.90 | 3087.21 | 3091.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 3088.80 | 3087.53 | 3091.65 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 3109.40 | 3095.82 | 3094.94 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 3054.00 | 3086.94 | 3091.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 3048.40 | 3068.71 | 3080.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 3093.20 | 3068.46 | 3076.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 3093.20 | 3068.46 | 3076.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 3093.20 | 3068.46 | 3076.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:15:00 | 3162.00 | 3068.46 | 3076.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 10:15:00 | 3195.10 | 3093.79 | 3087.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 11:15:00 | 3232.00 | 3121.43 | 3100.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 3132.50 | 3164.79 | 3137.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 10:15:00 | 3132.50 | 3164.79 | 3137.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 3132.50 | 3164.79 | 3137.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:30:00 | 3125.50 | 3164.79 | 3137.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 3158.20 | 3163.47 | 3139.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 12:15:00 | 3182.00 | 3163.47 | 3139.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:15:00 | 3170.00 | 3159.37 | 3143.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 3121.00 | 3153.39 | 3143.56 | SL hit (close<static) qty=1.00 sl=3130.30 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 3084.30 | 3132.14 | 3135.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 3055.00 | 3104.87 | 3120.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 3100.00 | 3097.36 | 3114.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 3100.00 | 3097.36 | 3114.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 3100.00 | 3097.36 | 3114.31 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 3173.10 | 3121.68 | 3118.91 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 3093.00 | 3120.37 | 3121.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 2948.70 | 3086.04 | 3106.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2970.20 | 2923.50 | 2969.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 2970.20 | 2923.50 | 2969.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2970.20 | 2923.50 | 2969.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 2970.20 | 2923.50 | 2969.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 2992.40 | 2937.28 | 2971.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 2992.40 | 2937.28 | 2971.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 2972.00 | 2944.22 | 2971.99 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 3073.10 | 2988.83 | 2984.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 3095.00 | 3061.47 | 3034.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 11:15:00 | 3197.10 | 3207.35 | 3162.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 12:00:00 | 3197.10 | 3207.35 | 3162.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 3175.00 | 3196.06 | 3171.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 3104.30 | 3196.06 | 3171.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 3092.00 | 3175.25 | 3163.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 3097.40 | 3175.25 | 3163.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 3118.00 | 3163.80 | 3159.79 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 3108.30 | 3152.70 | 3155.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 3103.00 | 3142.76 | 3150.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 3156.50 | 3145.51 | 3150.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 13:15:00 | 3156.50 | 3145.51 | 3150.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 3156.50 | 3145.51 | 3150.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 3136.60 | 3145.51 | 3150.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 3156.10 | 3147.63 | 3151.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 3139.40 | 3148.30 | 3151.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 3174.70 | 3153.58 | 3153.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 09:15:00 | 3174.70 | 3153.58 | 3153.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 10:15:00 | 3205.30 | 3163.92 | 3158.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 12:15:00 | 3130.00 | 3159.26 | 3157.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 12:15:00 | 3130.00 | 3159.26 | 3157.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 3130.00 | 3159.26 | 3157.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 3130.00 | 3159.26 | 3157.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 3149.00 | 3157.21 | 3156.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 14:15:00 | 3163.40 | 3157.21 | 3156.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-29 09:15:00 | 3479.74 | 3371.11 | 3295.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 13:15:00 | 3216.90 | 3301.27 | 3311.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 14:15:00 | 3197.00 | 3280.41 | 3301.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 3219.70 | 3212.92 | 3243.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 3219.70 | 3212.92 | 3243.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 3219.70 | 3212.92 | 3243.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 3230.50 | 3212.92 | 3243.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 3224.40 | 3205.92 | 3227.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 3224.40 | 3205.92 | 3227.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 3226.10 | 3209.95 | 3227.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 3230.00 | 3209.95 | 3227.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 3236.90 | 3215.34 | 3228.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:00:00 | 3219.80 | 3216.23 | 3227.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:00:00 | 3223.00 | 3217.59 | 3227.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 3222.80 | 3212.32 | 3215.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 3221.60 | 3196.91 | 3204.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 3245.00 | 3206.53 | 3207.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 3245.00 | 3206.53 | 3207.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 3235.00 | 3212.22 | 3210.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 3235.00 | 3212.22 | 3210.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 3261.00 | 3229.21 | 3219.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 3634.80 | 3655.93 | 3568.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:45:00 | 3625.20 | 3655.93 | 3568.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 3613.10 | 3647.52 | 3592.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:45:00 | 3613.50 | 3647.52 | 3592.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 3579.70 | 3627.20 | 3592.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 3579.70 | 3627.20 | 3592.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 3575.90 | 3616.94 | 3591.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 3575.90 | 3616.94 | 3591.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 3593.70 | 3607.66 | 3591.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 3612.10 | 3607.66 | 3591.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 3578.70 | 3601.87 | 3589.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 3578.70 | 3601.87 | 3589.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 3552.10 | 3591.92 | 3586.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 3552.10 | 3591.92 | 3586.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 3551.70 | 3583.87 | 3583.30 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 3554.90 | 3578.08 | 3580.72 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 3589.00 | 3581.54 | 3581.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 14:15:00 | 3604.00 | 3588.15 | 3584.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 3569.00 | 3588.46 | 3585.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 3569.00 | 3588.46 | 3585.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 3569.00 | 3588.46 | 3585.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 3569.00 | 3588.46 | 3585.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 3558.00 | 3582.37 | 3583.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 3539.50 | 3573.79 | 3579.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 3529.10 | 3515.89 | 3535.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 15:00:00 | 3529.10 | 3515.89 | 3535.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 3542.00 | 3521.11 | 3535.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 3478.00 | 3521.11 | 3535.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:30:00 | 3516.00 | 3481.38 | 3484.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 3566.60 | 3498.43 | 3491.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 11:15:00 | 3566.60 | 3498.43 | 3491.74 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 3468.90 | 3489.79 | 3490.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 3455.60 | 3482.95 | 3487.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 3286.80 | 3230.38 | 3290.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 14:15:00 | 3286.80 | 3230.38 | 3290.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 3286.80 | 3230.38 | 3290.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 3286.80 | 3230.38 | 3290.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 3248.00 | 3233.90 | 3286.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 3258.00 | 3233.90 | 3286.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 3251.60 | 3237.44 | 3283.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:45:00 | 3224.40 | 3233.75 | 3277.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 3239.00 | 3233.35 | 3269.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 3332.50 | 3254.67 | 3273.29 | SL hit (close>static) qty=1.00 sl=3296.50 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 3309.50 | 3286.07 | 3284.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 3332.20 | 3295.29 | 3289.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 3421.00 | 3430.23 | 3400.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:00:00 | 3428.00 | 3429.79 | 3403.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 3427.20 | 3426.57 | 3406.24 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 3333.80 | 3398.63 | 3400.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 3292.40 | 3334.91 | 3361.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 3345.60 | 3315.31 | 3339.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 3345.60 | 3315.31 | 3339.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 3345.60 | 3315.31 | 3339.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 3345.60 | 3315.31 | 3339.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 3340.00 | 3320.24 | 3339.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 3355.80 | 3320.24 | 3339.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3327.00 | 3321.60 | 3338.15 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 3393.00 | 3349.39 | 3344.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 15:15:00 | 3422.10 | 3385.02 | 3365.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 14:15:00 | 3517.20 | 3521.36 | 3477.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 15:00:00 | 3517.20 | 3521.36 | 3477.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 3609.00 | 3592.63 | 3545.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 3557.80 | 3592.63 | 3545.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 3563.00 | 3583.05 | 3552.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 3563.00 | 3583.05 | 3552.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 3555.50 | 3577.54 | 3553.05 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 12:15:00 | 3521.00 | 3541.24 | 3542.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 13:15:00 | 3513.80 | 3535.76 | 3540.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 3576.70 | 3531.88 | 3534.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 3576.70 | 3531.88 | 3534.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 3576.70 | 3531.88 | 3534.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 3585.30 | 3531.88 | 3534.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 3597.00 | 3544.90 | 3540.33 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 3532.00 | 3554.63 | 3555.10 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 3604.90 | 3562.89 | 3558.32 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 3513.10 | 3563.94 | 3565.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 3500.20 | 3551.19 | 3559.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 3450.00 | 3449.94 | 3467.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 3450.00 | 3449.94 | 3467.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3458.20 | 3444.09 | 3459.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 15:15:00 | 3422.00 | 3438.63 | 3451.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 3384.80 | 3339.61 | 3334.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 3384.80 | 3339.61 | 3334.43 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 3328.20 | 3346.47 | 3347.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 3307.00 | 3336.32 | 3341.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 10:15:00 | 3313.00 | 3310.98 | 3322.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:45:00 | 3315.00 | 3310.98 | 3322.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 3316.50 | 3312.09 | 3321.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 3316.50 | 3312.09 | 3321.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 3310.80 | 3311.83 | 3320.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:15:00 | 3297.10 | 3311.83 | 3320.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:15:00 | 3132.24 | 3276.97 | 3300.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 3254.70 | 3207.21 | 3242.05 | SL hit (close>ema200) qty=0.50 sl=3207.21 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 09:15:00 | 3257.40 | 3201.48 | 3198.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 3275.00 | 3250.46 | 3229.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 3273.60 | 3274.32 | 3255.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 3273.60 | 3274.32 | 3255.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 3242.10 | 3267.87 | 3254.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:15:00 | 3230.00 | 3267.87 | 3254.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 3226.00 | 3259.50 | 3252.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:15:00 | 3217.80 | 3259.50 | 3252.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 3224.50 | 3246.98 | 3247.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 3214.50 | 3240.48 | 3244.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 3197.40 | 3184.47 | 3204.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 3197.40 | 3184.47 | 3204.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 3197.40 | 3184.47 | 3204.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:00:00 | 3185.00 | 3186.23 | 3202.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 3184.40 | 3185.62 | 3199.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 13:15:00 | 3025.75 | 3078.87 | 3115.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 13:15:00 | 3025.18 | 3078.87 | 3115.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-05 11:15:00 | 2866.50 | 2961.88 | 3038.73 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 2927.00 | 2895.21 | 2893.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 2970.00 | 2910.17 | 2900.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 2953.80 | 2954.85 | 2936.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 12:45:00 | 2955.30 | 2954.85 | 2936.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2943.80 | 2954.96 | 2942.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 2943.80 | 2954.96 | 2942.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2932.00 | 2950.37 | 2941.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 2932.00 | 2950.37 | 2941.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 2930.00 | 2946.29 | 2940.76 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 2912.70 | 2935.85 | 2936.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 2897.00 | 2922.77 | 2930.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 2879.50 | 2868.33 | 2885.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 2879.50 | 2868.33 | 2885.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 2879.50 | 2868.33 | 2885.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 2875.90 | 2868.33 | 2885.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2933.80 | 2883.45 | 2889.49 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 2934.90 | 2901.67 | 2897.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 2952.40 | 2911.81 | 2902.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 3039.40 | 3044.31 | 3000.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:30:00 | 3029.00 | 3044.31 | 3000.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 3060.50 | 3075.36 | 3056.45 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 3010.10 | 3045.76 | 3049.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 2997.00 | 3036.01 | 3044.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 2980.00 | 2968.65 | 2992.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 2980.00 | 2968.65 | 2992.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2980.00 | 2968.65 | 2992.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 2987.00 | 2968.65 | 2992.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2962.90 | 2968.67 | 2988.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 2940.00 | 2961.52 | 2971.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 2793.00 | 2858.16 | 2905.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-06 10:15:00 | 2646.00 | 2724.74 | 2798.55 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 2621.90 | 2535.06 | 2524.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 2637.00 | 2575.86 | 2547.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 2600.00 | 2601.89 | 2578.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 2600.00 | 2601.89 | 2578.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2634.50 | 2608.11 | 2585.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:30:00 | 2655.80 | 2620.82 | 2595.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 2653.80 | 2620.82 | 2595.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 2646.70 | 2734.02 | 2740.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 2646.70 | 2734.02 | 2740.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 2639.50 | 2715.12 | 2731.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 2749.30 | 2717.05 | 2727.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 11:15:00 | 2749.30 | 2717.05 | 2727.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 2749.30 | 2717.05 | 2727.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 2749.30 | 2717.05 | 2727.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 2760.60 | 2725.76 | 2730.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 2760.60 | 2725.76 | 2730.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 2794.50 | 2744.83 | 2738.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 2800.00 | 2755.86 | 2744.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 3080.00 | 3085.77 | 3010.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:30:00 | 3075.00 | 3087.94 | 3018.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 3057.00 | 3067.16 | 3047.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 3048.90 | 3067.16 | 3047.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3050.60 | 3063.85 | 3047.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:45:00 | 3050.00 | 3063.85 | 3047.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 3094.70 | 3070.02 | 3052.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 3115.00 | 3072.00 | 3054.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:00:00 | 3097.90 | 3143.37 | 3141.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 3110.70 | 3136.84 | 3138.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 3110.70 | 3136.84 | 3138.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 3085.00 | 3115.60 | 3127.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 3119.00 | 3107.33 | 3117.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 13:15:00 | 3119.00 | 3107.33 | 3117.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 3119.00 | 3107.33 | 3117.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 3119.00 | 3107.33 | 3117.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 3112.80 | 3108.43 | 3116.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:30:00 | 3129.90 | 3108.43 | 3116.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 3126.00 | 3111.94 | 3117.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 3105.70 | 3111.94 | 3117.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 3094.10 | 3110.45 | 3116.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:15:00 | 2950.41 | 3049.52 | 3081.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:15:00 | 2939.39 | 3049.52 | 3081.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 2910.70 | 2908.92 | 2955.68 | SL hit (close>ema200) qty=0.50 sl=2908.92 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 11:15:00 | 2986.90 | 2939.56 | 2933.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 12:15:00 | 3000.90 | 2951.83 | 2939.29 | Break + close above crossover candle high |

### Cycle 54 — SELL (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 09:15:00 | 2692.50 | 2926.31 | 2934.47 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 2653.90 | 2645.16 | 2644.91 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 2632.10 | 2642.54 | 2643.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 2565.40 | 2625.59 | 2635.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 2599.70 | 2593.54 | 2613.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 14:45:00 | 2598.30 | 2593.54 | 2613.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2625.70 | 2600.57 | 2612.98 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 2641.00 | 2620.24 | 2619.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 2655.60 | 2627.31 | 2622.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 2677.00 | 2679.40 | 2657.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 2677.00 | 2679.40 | 2657.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2636.80 | 2670.17 | 2657.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 2672.60 | 2670.66 | 2658.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-17 10:15:00 | 2939.86 | 2807.91 | 2757.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 13:15:00 | 3049.10 | 3078.72 | 3079.30 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 3086.00 | 3076.02 | 3075.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 3146.50 | 3090.11 | 3081.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3132.00 | 3151.13 | 3124.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:45:00 | 3131.00 | 3151.13 | 3124.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3137.00 | 3148.30 | 3125.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:45:00 | 3140.90 | 3146.64 | 3127.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:15:00 | 3146.00 | 3146.64 | 3127.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 13:15:00 | 3097.60 | 3134.75 | 3125.07 | SL hit (close<static) qty=1.00 sl=3122.20 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 3082.00 | 3117.60 | 3118.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 3065.40 | 3101.89 | 3110.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 13:15:00 | 3120.10 | 3104.57 | 3109.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 13:15:00 | 3120.10 | 3104.57 | 3109.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 3120.10 | 3104.57 | 3109.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:00:00 | 3120.10 | 3104.57 | 3109.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 3113.70 | 3106.39 | 3110.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:45:00 | 3114.00 | 3106.39 | 3110.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 3107.50 | 3106.62 | 3109.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 3079.40 | 3106.62 | 3109.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3074.00 | 3100.09 | 3106.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 10:00:00 | 3046.40 | 3082.82 | 3093.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 3077.90 | 3074.70 | 3074.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 3077.90 | 3074.70 | 3074.43 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 3070.60 | 3073.88 | 3074.08 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 11:15:00 | 3079.00 | 3074.90 | 3074.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 3088.00 | 3079.93 | 3077.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 3075.00 | 3078.95 | 3076.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 15:15:00 | 3075.00 | 3078.95 | 3076.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 3075.00 | 3078.95 | 3076.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 3102.10 | 3078.95 | 3076.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 3412.31 | 3354.43 | 3298.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 3424.40 | 3471.77 | 3475.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 15:15:00 | 3399.00 | 3440.18 | 3455.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 3354.60 | 3335.92 | 3378.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:00:00 | 3354.60 | 3335.92 | 3378.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 3366.00 | 3351.15 | 3368.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 3434.00 | 3351.15 | 3368.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 3450.00 | 3370.92 | 3375.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:45:00 | 3457.50 | 3370.92 | 3375.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 3477.00 | 3392.14 | 3384.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 3486.00 | 3449.25 | 3421.46 | Break + close above crossover candle high |

### Cycle 66 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 3143.00 | 3419.61 | 3425.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 3133.40 | 3362.37 | 3399.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 10:15:00 | 3158.00 | 3149.30 | 3205.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 11:00:00 | 3158.00 | 3149.30 | 3205.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3238.00 | 3184.39 | 3200.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 3238.00 | 3184.39 | 3200.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 3213.70 | 3190.25 | 3201.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:00:00 | 3206.40 | 3193.48 | 3201.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 3223.50 | 3206.67 | 3205.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 3223.50 | 3206.67 | 3205.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 3264.00 | 3228.87 | 3218.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 3232.00 | 3239.08 | 3228.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 3232.00 | 3239.08 | 3228.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 3229.00 | 3237.06 | 3228.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 09:15:00 | 3245.00 | 3237.06 | 3228.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 3258.00 | 3241.25 | 3231.62 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-05-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-11 15:15:00 | 3212.00 | 3226.31 | 3228.15 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 10:00:00 | 2937.60 | 2025-05-23 09:15:00 | 2696.60 | STOP_HIT | 1.00 | -8.20% |
| BUY | retest2 | 2025-05-21 12:15:00 | 2940.00 | 2025-05-23 09:15:00 | 2696.60 | STOP_HIT | 1.00 | -8.28% |
| BUY | retest2 | 2025-06-02 12:00:00 | 2955.00 | 2025-06-03 09:15:00 | 2898.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-06-05 13:00:00 | 2893.10 | 2025-06-11 09:15:00 | 2862.20 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-06-12 11:30:00 | 2901.20 | 2025-06-12 13:15:00 | 2832.10 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-06-12 13:00:00 | 2897.00 | 2025-06-12 13:15:00 | 2832.10 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-06-27 09:15:00 | 2969.80 | 2025-07-02 12:15:00 | 2971.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-06-27 11:15:00 | 2970.80 | 2025-07-02 12:15:00 | 2971.00 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-06-30 09:15:00 | 3015.50 | 2025-07-02 12:15:00 | 2971.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-07-04 11:30:00 | 2982.30 | 2025-07-08 11:15:00 | 3033.60 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-07-07 10:00:00 | 2977.70 | 2025-07-08 11:15:00 | 3033.60 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2983.10 | 2025-07-08 11:15:00 | 3033.60 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-07-14 11:00:00 | 3171.60 | 2025-07-18 09:15:00 | 3183.00 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-07-14 12:15:00 | 3178.00 | 2025-07-18 09:15:00 | 3183.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-07-25 13:15:00 | 3200.00 | 2025-07-28 09:15:00 | 3170.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-07-25 14:45:00 | 3199.90 | 2025-07-28 09:15:00 | 3170.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-07-28 09:15:00 | 3206.00 | 2025-07-28 09:15:00 | 3170.20 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-08-07 12:15:00 | 3182.00 | 2025-08-08 09:15:00 | 3121.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-08-07 15:15:00 | 3170.00 | 2025-08-08 09:15:00 | 3121.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-26 09:15:00 | 3139.40 | 2025-08-26 09:15:00 | 3174.70 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-08-26 14:15:00 | 3163.40 | 2025-08-29 09:15:00 | 3479.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-04 11:00:00 | 3219.80 | 2025-09-09 11:15:00 | 3235.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-09-04 12:00:00 | 3223.00 | 2025-09-09 11:15:00 | 3235.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-09-08 09:45:00 | 3222.80 | 2025-09-09 11:15:00 | 3235.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-09-09 09:45:00 | 3221.60 | 2025-09-09 11:15:00 | 3235.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-09-22 09:15:00 | 3478.00 | 2025-09-24 11:15:00 | 3566.60 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-09-24 10:30:00 | 3516.00 | 2025-09-24 11:15:00 | 3566.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-09-30 10:45:00 | 3224.40 | 2025-09-30 14:15:00 | 3332.50 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-09-30 13:15:00 | 3239.00 | 2025-09-30 14:15:00 | 3332.50 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-11-03 15:15:00 | 3422.00 | 2025-11-12 11:15:00 | 3384.80 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2025-11-18 13:15:00 | 3297.10 | 2025-11-19 09:15:00 | 3132.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 13:15:00 | 3297.10 | 2025-11-20 09:15:00 | 3254.70 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2025-12-01 12:00:00 | 3185.00 | 2025-12-04 13:15:00 | 3025.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 14:15:00 | 3184.40 | 2025-12-04 13:15:00 | 3025.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 12:00:00 | 3185.00 | 2025-12-05 11:15:00 | 2866.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-01 14:15:00 | 3184.40 | 2025-12-05 11:15:00 | 2865.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-02 09:15:00 | 2940.00 | 2026-01-05 09:15:00 | 2793.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 09:15:00 | 2940.00 | 2026-01-06 10:15:00 | 2646.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-27 11:30:00 | 2655.80 | 2026-02-01 14:15:00 | 2646.70 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-01-27 12:15:00 | 2653.80 | 2026-02-01 14:15:00 | 2646.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-02-09 09:15:00 | 3115.00 | 2026-02-13 10:15:00 | 3110.70 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-02-13 10:00:00 | 3097.90 | 2026-02-13 10:15:00 | 3110.70 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2026-02-17 09:15:00 | 3105.70 | 2026-02-18 09:15:00 | 2950.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 10:30:00 | 3094.10 | 2026-02-18 09:15:00 | 2939.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 09:15:00 | 3105.70 | 2026-02-20 09:15:00 | 2910.70 | STOP_HIT | 0.50 | 6.28% |
| SELL | retest2 | 2026-02-17 10:30:00 | 3094.10 | 2026-02-20 09:15:00 | 2910.70 | STOP_HIT | 0.50 | 5.93% |
| BUY | retest2 | 2026-03-12 11:00:00 | 2672.60 | 2026-03-17 10:15:00 | 2939.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-27 11:45:00 | 3140.90 | 2026-03-27 13:15:00 | 3097.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-03-27 12:15:00 | 3146.00 | 2026-03-27 13:15:00 | 3097.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-04-02 10:00:00 | 3046.40 | 2026-04-07 09:15:00 | 3077.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-04-08 09:15:00 | 3102.10 | 2026-04-15 10:15:00 | 3412.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-06 12:00:00 | 3206.40 | 2026-05-06 15:15:00 | 3223.50 | STOP_HIT | 1.00 | -0.53% |
