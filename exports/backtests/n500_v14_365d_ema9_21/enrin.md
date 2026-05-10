# Siemens Energy India Ltd. (ENRIN)

## Backtest Summary

- **Window:** 2025-06-19 09:15:00 → 2026-05-08 15:15:00 (1528 bars)
- **Last close:** 3186.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 43 |
| ALERT2 | 42 |
| ALERT2_SKIP | 19 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 50 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 43
- **Target hits / Stop hits / Partials:** 0 / 53 / 2
- **Avg / median % per leg:** -1.13% / -1.41%
- **Sum % (uncompounded):** -62.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 4 | 16.0% | 0 | 25 | 0 | -1.71% | -42.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 4 | 16.0% | 0 | 25 | 0 | -1.71% | -42.8% |
| SELL (all) | 30 | 8 | 26.7% | 0 | 28 | 2 | -0.64% | -19.2% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.23% | -3.7% |
| SELL @ 3rd Alert (retest2) | 27 | 7 | 25.9% | 0 | 25 | 2 | -0.58% | -15.6% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.23% | -3.7% |
| retest2 (combined) | 52 | 11 | 21.2% | 0 | 50 | 2 | -1.12% | -58.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 12:15:00 | 2782.00 | 2712.54 | 2709.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 13:15:00 | 2790.20 | 2728.08 | 2716.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 2945.00 | 2952.24 | 2906.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:30:00 | 2935.40 | 2952.24 | 2906.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 2949.90 | 2957.00 | 2927.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:30:00 | 2929.70 | 2957.00 | 2927.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2901.60 | 2944.80 | 2927.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 2901.60 | 2944.80 | 2927.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2913.80 | 2938.60 | 2925.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 2913.80 | 2938.60 | 2925.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 2923.70 | 2935.62 | 2925.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:30:00 | 2910.10 | 2935.62 | 2925.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 2912.60 | 2931.02 | 2924.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 2912.60 | 2931.02 | 2924.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2920.10 | 2928.83 | 2924.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 2900.00 | 2928.83 | 2924.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 2954.00 | 2933.87 | 2926.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 2937.10 | 2933.87 | 2926.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2941.00 | 2936.24 | 2929.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 2915.70 | 2936.24 | 2929.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2914.80 | 2931.95 | 2927.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 2913.30 | 2931.95 | 2927.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 2910.10 | 2927.58 | 2926.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 2910.10 | 2927.58 | 2926.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 2919.90 | 2927.22 | 2926.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:45:00 | 2922.30 | 2927.22 | 2926.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 2917.00 | 2925.18 | 2925.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 2903.90 | 2920.92 | 2923.59 | Break + close below crossover candle low |

### Cycle 3 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 2970.90 | 2930.92 | 2927.89 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 2892.40 | 2922.71 | 2924.82 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 2952.50 | 2925.71 | 2924.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 2994.70 | 2944.26 | 2933.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 3114.10 | 3119.29 | 3076.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:45:00 | 3124.30 | 3119.29 | 3076.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 3109.20 | 3138.97 | 3112.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 3109.20 | 3138.97 | 3112.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 3120.80 | 3135.34 | 3113.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:15:00 | 3133.60 | 3135.34 | 3113.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 3085.20 | 3140.26 | 3123.98 | SL hit (close<static) qty=1.00 sl=3109.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 3085.90 | 3113.29 | 3114.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 3037.20 | 3085.45 | 3099.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 14:15:00 | 3053.20 | 3035.82 | 3065.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 14:45:00 | 3051.80 | 3035.82 | 3065.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 3025.80 | 3036.71 | 3061.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 3036.00 | 3036.71 | 3061.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 3084.90 | 3039.51 | 3052.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 3084.90 | 3039.51 | 3052.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 3125.00 | 3056.61 | 3058.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 3162.10 | 3056.61 | 3058.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 3157.00 | 3076.68 | 3067.88 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 3032.10 | 3067.47 | 3069.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 3031.30 | 3060.23 | 3066.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 3071.90 | 3044.62 | 3052.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 3071.90 | 3044.62 | 3052.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 3071.90 | 3044.62 | 3052.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 3071.90 | 3044.62 | 3052.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 3061.60 | 3048.01 | 3053.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 3069.90 | 3048.01 | 3053.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 3057.90 | 3051.88 | 3054.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 3059.00 | 3051.88 | 3054.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 3057.90 | 3053.08 | 3054.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 3053.30 | 3053.08 | 3054.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 3053.10 | 3053.08 | 3054.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 3060.00 | 3053.08 | 3054.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 3050.30 | 3052.53 | 3054.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 3045.50 | 3052.53 | 3054.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3051.10 | 3052.24 | 3054.06 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 3099.50 | 3061.69 | 3058.19 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 3049.00 | 3057.89 | 3058.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 3004.50 | 3047.22 | 3053.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 3018.80 | 3013.57 | 3028.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 3018.80 | 3013.57 | 3028.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 3018.80 | 3013.57 | 3028.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 3018.80 | 3013.57 | 3028.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 3005.00 | 3003.96 | 3016.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:00:00 | 2994.00 | 3001.97 | 3014.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 10:00:00 | 2998.90 | 2985.97 | 2997.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 10:45:00 | 2995.80 | 2988.63 | 2997.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:15:00 | 2993.60 | 2988.63 | 2997.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 2979.80 | 2986.87 | 2996.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 2973.00 | 2986.87 | 2996.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 3022.30 | 2982.23 | 2990.67 | SL hit (close>static) qty=1.00 sl=3001.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:00:00 | 2967.80 | 2983.75 | 2990.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:45:00 | 2965.80 | 2982.56 | 2989.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 11:30:00 | 2974.30 | 2983.05 | 2988.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2991.70 | 2984.78 | 2988.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:45:00 | 2994.90 | 2984.78 | 2988.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 3010.60 | 2989.94 | 2990.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 3010.60 | 2989.94 | 2990.92 | SL hit (close>static) qty=1.00 sl=3001.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 3010.60 | 2989.94 | 2990.92 | SL hit (close>static) qty=1.00 sl=3001.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 3010.60 | 2989.94 | 2990.92 | SL hit (close>static) qty=1.00 sl=3001.90 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 3010.60 | 2989.94 | 2990.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 3033.60 | 2998.67 | 2994.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 3033.60 | 2998.67 | 2994.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 3033.60 | 2998.67 | 2994.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 3033.60 | 2998.67 | 2994.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 3033.60 | 2998.67 | 2994.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 3156.70 | 3037.21 | 3013.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 3184.60 | 3220.11 | 3173.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 3184.60 | 3220.11 | 3173.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 3184.60 | 3220.11 | 3173.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 3191.10 | 3220.11 | 3173.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 3184.40 | 3208.84 | 3179.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 3184.40 | 3208.84 | 3179.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 3174.90 | 3202.05 | 3178.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:30:00 | 3173.50 | 3202.05 | 3178.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 3167.50 | 3195.14 | 3177.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:30:00 | 3165.80 | 3195.14 | 3177.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 3177.00 | 3178.08 | 3173.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:15:00 | 3202.90 | 3178.08 | 3173.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 10:15:00 | 3184.80 | 3262.20 | 3244.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 3162.10 | 3226.63 | 3230.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 3162.10 | 3226.63 | 3230.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 3162.10 | 3226.63 | 3230.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 3140.00 | 3170.40 | 3194.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 3171.40 | 3160.20 | 3182.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 3171.40 | 3160.20 | 3182.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 3196.00 | 3167.36 | 3184.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 3194.30 | 3167.36 | 3184.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 3149.20 | 3163.73 | 3180.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 3125.90 | 3156.30 | 3175.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 3136.30 | 3114.43 | 3137.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:00:00 | 3139.20 | 3119.38 | 3137.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 3200.00 | 3147.09 | 3143.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 3200.00 | 3147.09 | 3143.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 3200.00 | 3147.09 | 3143.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 3200.00 | 3147.09 | 3143.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 3222.80 | 3181.18 | 3170.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 12:15:00 | 3180.00 | 3186.33 | 3176.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 12:15:00 | 3180.00 | 3186.33 | 3176.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 3180.00 | 3186.33 | 3176.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:00:00 | 3180.00 | 3186.33 | 3176.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 3170.90 | 3183.25 | 3175.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:45:00 | 3150.40 | 3183.25 | 3175.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 3160.60 | 3178.72 | 3174.47 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 3167.20 | 3172.98 | 3173.17 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 3196.70 | 3177.73 | 3175.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 3237.90 | 3193.66 | 3183.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 3300.50 | 3317.14 | 3273.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 3300.50 | 3317.14 | 3273.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3383.50 | 3384.69 | 3359.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 3346.10 | 3384.69 | 3359.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 3344.00 | 3377.60 | 3364.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 3344.00 | 3377.60 | 3364.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 3334.80 | 3369.04 | 3361.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 3334.80 | 3369.04 | 3361.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 3333.80 | 3361.99 | 3359.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 3352.80 | 3361.99 | 3359.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 3359.50 | 3377.40 | 3368.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 3359.50 | 3377.40 | 3368.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 3357.30 | 3373.38 | 3367.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 3357.30 | 3373.38 | 3367.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 3388.00 | 3376.51 | 3370.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 3391.60 | 3376.51 | 3370.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 3374.00 | 3376.01 | 3370.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 3469.60 | 3378.44 | 3374.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:15:00 | 3441.00 | 3489.97 | 3484.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 3441.00 | 3480.18 | 3480.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 3441.00 | 3480.18 | 3480.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 3441.00 | 3480.18 | 3480.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 09:15:00 | 3423.20 | 3468.78 | 3475.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 3467.00 | 3425.75 | 3444.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 3467.00 | 3425.75 | 3444.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 3467.00 | 3425.75 | 3444.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 3467.00 | 3425.75 | 3444.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 3428.60 | 3426.32 | 3442.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 14:15:00 | 3421.70 | 3428.89 | 3440.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 3405.10 | 3429.01 | 3439.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 12:15:00 | 3470.00 | 3446.92 | 3444.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 12:15:00 | 3470.00 | 3446.92 | 3444.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 3470.00 | 3446.92 | 3444.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 3520.00 | 3463.93 | 3452.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 3474.20 | 3476.95 | 3463.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 12:00:00 | 3474.20 | 3476.95 | 3463.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 3446.20 | 3470.80 | 3461.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 3446.20 | 3470.80 | 3461.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 3446.50 | 3465.94 | 3460.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 15:00:00 | 3460.20 | 3464.79 | 3460.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 3400.00 | 3450.43 | 3454.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 09:15:00 | 3400.00 | 3450.43 | 3454.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 10:15:00 | 3377.60 | 3435.86 | 3447.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 3397.20 | 3377.49 | 3395.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 3397.20 | 3377.49 | 3395.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 3397.20 | 3377.49 | 3395.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 3397.20 | 3377.49 | 3395.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3401.00 | 3382.19 | 3396.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 3400.00 | 3382.19 | 3396.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 3410.20 | 3387.79 | 3397.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 3410.20 | 3387.79 | 3397.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 3426.90 | 3395.62 | 3400.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 3426.90 | 3395.62 | 3400.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 3390.00 | 3394.49 | 3399.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 3389.00 | 3394.49 | 3399.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 3429.00 | 3399.61 | 3400.12 | SL hit (close>static) qty=1.00 sl=3428.80 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 3418.00 | 3402.98 | 3401.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 3458.80 | 3414.14 | 3406.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 3489.80 | 3493.12 | 3474.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:45:00 | 3488.50 | 3493.12 | 3474.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 3475.00 | 3489.49 | 3474.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 3475.00 | 3489.49 | 3474.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 3475.00 | 3486.59 | 3474.43 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 3432.60 | 3465.36 | 3467.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 3416.60 | 3455.61 | 3462.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 3461.40 | 3456.77 | 3462.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 13:15:00 | 3461.40 | 3456.77 | 3462.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 3461.40 | 3456.77 | 3462.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 3461.40 | 3456.77 | 3462.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 3436.60 | 3452.73 | 3460.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:15:00 | 3437.00 | 3452.73 | 3460.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 3437.00 | 3449.59 | 3458.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 3496.10 | 3449.59 | 3458.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 3481.70 | 3456.01 | 3460.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 3481.70 | 3456.01 | 3460.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 3507.60 | 3466.33 | 3464.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 14:15:00 | 3530.70 | 3487.27 | 3475.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 3492.10 | 3549.85 | 3524.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 3492.10 | 3549.85 | 3524.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 3492.10 | 3549.85 | 3524.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 3492.10 | 3549.85 | 3524.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 3487.00 | 3537.28 | 3521.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:30:00 | 3477.00 | 3537.28 | 3521.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 3511.00 | 3513.47 | 3513.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 3486.70 | 3513.47 | 3513.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 3530.00 | 3516.77 | 3514.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 10:15:00 | 3537.70 | 3516.77 | 3514.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 3551.80 | 3516.55 | 3515.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:45:00 | 3574.40 | 3527.53 | 3521.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 3478.50 | 3519.97 | 3523.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 3478.50 | 3519.97 | 3523.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 3478.50 | 3519.97 | 3523.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 09:15:00 | 3478.50 | 3519.97 | 3523.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 09:15:00 | 3412.70 | 3480.55 | 3499.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 12:15:00 | 3472.90 | 3467.84 | 3488.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 12:45:00 | 3473.00 | 3467.84 | 3488.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 3420.20 | 3408.98 | 3431.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 3420.20 | 3408.98 | 3431.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3296.20 | 3286.69 | 3335.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 3318.80 | 3286.69 | 3335.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3314.70 | 3286.21 | 3309.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 3328.70 | 3286.21 | 3309.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 3295.00 | 3287.97 | 3308.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 3263.00 | 3282.98 | 3304.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 12:15:00 | 3344.90 | 3295.36 | 3307.86 | SL hit (close>static) qty=1.00 sl=3328.50 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 3339.20 | 3314.99 | 3314.64 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 3256.70 | 3303.33 | 3309.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 3242.50 | 3291.16 | 3303.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 15:15:00 | 3245.00 | 3242.65 | 3260.34 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:15:00 | 3229.00 | 3242.65 | 3260.34 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 3199.20 | 3178.96 | 3201.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 3199.20 | 3178.96 | 3201.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 3185.50 | 3180.27 | 3200.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 3176.00 | 3180.27 | 3200.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 3137.40 | 3115.55 | 3124.27 | SL hit (close>ema400) qty=1.00 sl=3124.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 3162.20 | 3131.51 | 3130.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 3162.20 | 3131.51 | 3130.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 3184.10 | 3142.03 | 3135.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 3152.10 | 3155.26 | 3144.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 14:00:00 | 3152.10 | 3155.26 | 3144.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 3136.40 | 3151.49 | 3143.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 3136.40 | 3151.49 | 3143.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 3136.60 | 3148.51 | 3142.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 3174.40 | 3148.51 | 3142.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 3146.90 | 3148.19 | 3143.14 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 3131.00 | 3140.31 | 3140.43 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 13:15:00 | 3148.90 | 3142.03 | 3141.20 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 3134.00 | 3139.63 | 3140.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 3109.50 | 3133.61 | 3137.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 11:15:00 | 3135.00 | 3129.73 | 3134.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 3135.00 | 3129.73 | 3134.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 3135.00 | 3129.73 | 3134.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 3135.00 | 3129.73 | 3134.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 3122.60 | 3128.30 | 3133.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:15:00 | 3113.50 | 3128.30 | 3133.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 3152.80 | 3133.81 | 3135.25 | SL hit (close>static) qty=1.00 sl=3135.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 3154.30 | 3137.91 | 3136.98 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 3126.00 | 3134.42 | 3135.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 3108.50 | 3129.24 | 3133.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 13:15:00 | 3112.80 | 3098.06 | 3109.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 3112.80 | 3098.06 | 3109.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 3112.80 | 3098.06 | 3109.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 3112.80 | 3098.06 | 3109.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 3128.80 | 3104.21 | 3111.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 3137.30 | 3104.21 | 3111.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 3144.00 | 3112.17 | 3114.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 3141.20 | 3112.17 | 3114.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 3160.80 | 3121.89 | 3118.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 3213.00 | 3158.47 | 3139.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 3187.00 | 3191.56 | 3170.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 3187.00 | 3191.56 | 3170.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 3216.90 | 3205.52 | 3191.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:45:00 | 3197.80 | 3205.52 | 3191.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 3170.00 | 3220.94 | 3209.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 3170.00 | 3220.94 | 3209.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 3160.00 | 3208.75 | 3204.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 3152.70 | 3208.75 | 3204.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 3179.20 | 3198.90 | 3200.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 3150.10 | 3183.16 | 3192.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 3187.90 | 3184.11 | 3192.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 3187.90 | 3184.11 | 3192.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 3198.80 | 3187.05 | 3192.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 3198.00 | 3187.05 | 3192.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3208.50 | 3191.34 | 3194.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 3221.60 | 3191.34 | 3194.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 3226.00 | 3198.27 | 3197.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 3268.80 | 3220.84 | 3208.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 11:15:00 | 3257.10 | 3272.84 | 3250.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 12:00:00 | 3257.10 | 3272.84 | 3250.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 3248.20 | 3267.92 | 3250.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 14:30:00 | 3262.20 | 3257.19 | 3248.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 3275.30 | 3254.71 | 3248.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:30:00 | 3285.00 | 3262.40 | 3252.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:15:00 | 3265.00 | 3280.26 | 3277.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 3248.70 | 3270.42 | 3273.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 3248.70 | 3270.42 | 3273.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 3248.70 | 3270.42 | 3273.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 3248.70 | 3270.42 | 3273.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 3248.70 | 3270.42 | 3273.04 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 3286.90 | 3275.06 | 3273.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 09:15:00 | 3319.70 | 3292.71 | 3283.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 12:15:00 | 3329.60 | 3337.25 | 3320.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 13:00:00 | 3329.60 | 3337.25 | 3320.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 3308.10 | 3329.91 | 3320.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 3308.10 | 3329.91 | 3320.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 3311.20 | 3326.17 | 3319.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 3359.20 | 3326.17 | 3319.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:15:00 | 3331.60 | 3329.89 | 3329.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 3335.40 | 3336.43 | 3334.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 13:00:00 | 3331.10 | 3336.43 | 3334.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 3329.00 | 3334.94 | 3333.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:30:00 | 3329.30 | 3334.94 | 3333.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 3146.00 | 3297.15 | 3316.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 3146.00 | 3297.15 | 3316.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 3146.00 | 3297.15 | 3316.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 3146.00 | 3297.15 | 3316.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 3146.00 | 3297.15 | 3316.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 3092.80 | 3210.06 | 3268.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 3125.70 | 3123.03 | 3188.71 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-26 14:00:00 | 3106.50 | 3116.09 | 3164.37 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-26 14:30:00 | 3105.20 | 3116.66 | 3160.24 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 3094.50 | 3112.58 | 3150.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 3207.20 | 3133.02 | 3142.31 | SL hit (close>ema400) qty=1.00 sl=3142.31 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 3207.20 | 3133.02 | 3142.31 | SL hit (close>ema400) qty=1.00 sl=3142.31 alert=retest1 |

### Cycle 37 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 3162.00 | 3148.37 | 3148.26 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 3147.20 | 3148.13 | 3148.17 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 13:15:00 | 3164.40 | 3151.39 | 3149.64 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 3127.20 | 3148.08 | 3148.78 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 3180.40 | 3151.17 | 3147.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 11:15:00 | 3185.80 | 3158.10 | 3151.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 3162.00 | 3175.17 | 3163.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 3162.00 | 3175.17 | 3163.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 3162.00 | 3175.17 | 3163.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 3162.00 | 3175.17 | 3163.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 3166.70 | 3173.47 | 3164.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:15:00 | 3155.90 | 3173.47 | 3164.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 3159.20 | 3170.62 | 3163.71 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 3127.10 | 3157.02 | 3158.42 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 3168.70 | 3159.77 | 3159.14 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 3154.40 | 3158.70 | 3158.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 3130.00 | 3152.96 | 3156.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 3099.90 | 3098.45 | 3116.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 09:15:00 | 3103.60 | 3098.45 | 3116.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 3075.00 | 3093.76 | 3113.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 3070.50 | 3082.04 | 3105.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 14:15:00 | 2916.97 | 2944.61 | 2981.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 2966.20 | 2927.47 | 2947.65 | SL hit (close>ema200) qty=0.50 sl=2927.47 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 10:15:00 | 2996.00 | 2959.50 | 2955.40 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 2856.10 | 2950.69 | 2960.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 10:15:00 | 2840.50 | 2928.65 | 2949.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 2612.50 | 2601.10 | 2631.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 2612.50 | 2601.10 | 2631.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2612.50 | 2601.10 | 2631.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:30:00 | 2596.00 | 2614.02 | 2624.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 2572.10 | 2555.81 | 2554.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 2572.10 | 2555.81 | 2554.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 2606.60 | 2568.99 | 2561.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 2583.60 | 2600.48 | 2585.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 2583.60 | 2600.48 | 2585.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 2583.60 | 2600.48 | 2585.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 2586.00 | 2600.48 | 2585.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 2595.80 | 2599.54 | 2586.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 15:00:00 | 2606.30 | 2597.82 | 2589.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 2627.60 | 2598.45 | 2590.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 2568.90 | 2595.47 | 2591.27 | SL hit (close<static) qty=1.00 sl=2583.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 2568.90 | 2595.47 | 2591.27 | SL hit (close<static) qty=1.00 sl=2583.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 2570.40 | 2586.18 | 2587.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 2534.60 | 2575.86 | 2582.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 2390.00 | 2377.08 | 2429.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 2383.70 | 2377.08 | 2429.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2394.00 | 2390.80 | 2410.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:45:00 | 2388.50 | 2390.06 | 2408.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 2269.07 | 2301.05 | 2325.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 2268.00 | 2256.55 | 2277.66 | SL hit (close>ema200) qty=0.50 sl=2256.55 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 2318.30 | 2213.73 | 2203.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 2335.40 | 2238.06 | 2215.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 2455.00 | 2459.62 | 2419.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 2455.00 | 2459.62 | 2419.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2449.30 | 2457.56 | 2422.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2446.10 | 2457.56 | 2422.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2448.20 | 2462.44 | 2436.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 2444.40 | 2462.44 | 2436.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 2375.90 | 2445.13 | 2431.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 2375.90 | 2445.13 | 2431.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 2400.00 | 2436.10 | 2428.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:15:00 | 2418.90 | 2430.18 | 2426.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 2388.00 | 2421.75 | 2422.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 13:15:00 | 2388.00 | 2421.75 | 2422.97 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 2437.10 | 2424.82 | 2424.26 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 15:15:00 | 2417.70 | 2423.39 | 2423.66 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2525.80 | 2443.88 | 2432.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 2656.50 | 2535.94 | 2490.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2572.80 | 2587.41 | 2545.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 2565.00 | 2587.41 | 2545.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2735.90 | 2707.23 | 2652.06 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 2646.20 | 2675.35 | 2677.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 11:15:00 | 2636.60 | 2667.60 | 2673.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 15:15:00 | 2675.00 | 2664.07 | 2669.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 15:15:00 | 2675.00 | 2664.07 | 2669.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 2675.00 | 2664.07 | 2669.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 2734.50 | 2664.07 | 2669.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2704.10 | 2672.07 | 2672.43 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 10:15:00 | 2709.10 | 2679.48 | 2675.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 11:15:00 | 2722.00 | 2687.98 | 2679.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 14:15:00 | 2740.60 | 2745.01 | 2720.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 14:15:00 | 2740.60 | 2745.01 | 2720.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 2740.60 | 2745.01 | 2720.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 2740.60 | 2745.01 | 2720.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 2782.00 | 2793.14 | 2767.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 12:30:00 | 2839.30 | 2800.51 | 2777.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 2731.90 | 2783.77 | 2784.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 2731.90 | 2783.77 | 2784.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 2720.00 | 2763.48 | 2774.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 15:15:00 | 2760.00 | 2758.34 | 2768.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:15:00 | 2749.60 | 2758.34 | 2768.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 2732.40 | 2753.15 | 2764.89 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 2795.00 | 2772.99 | 2772.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 13:15:00 | 2810.10 | 2780.41 | 2775.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 14:15:00 | 2770.30 | 2778.39 | 2775.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 14:15:00 | 2770.30 | 2778.39 | 2775.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 2770.30 | 2778.39 | 2775.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 2770.30 | 2778.39 | 2775.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 2780.00 | 2778.71 | 2775.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 2808.20 | 2778.71 | 2775.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 2814.50 | 2793.86 | 2785.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 2860.50 | 2933.82 | 2940.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 2860.50 | 2933.82 | 2940.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 09:15:00 | 2860.50 | 2933.82 | 2940.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 2844.20 | 2915.90 | 2931.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 2878.90 | 2876.94 | 2900.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:30:00 | 2886.00 | 2876.94 | 2900.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 2913.70 | 2878.93 | 2893.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 2913.70 | 2878.93 | 2893.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 2920.10 | 2887.16 | 2895.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 2920.10 | 2887.16 | 2895.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 2913.20 | 2886.41 | 2892.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 13:00:00 | 2913.20 | 2886.41 | 2892.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 13:15:00 | 2881.70 | 2885.47 | 2891.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 14:15:00 | 2874.90 | 2885.47 | 2891.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 14:45:00 | 2877.90 | 2884.12 | 2890.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 2953.20 | 2898.49 | 2895.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 2953.20 | 2898.49 | 2895.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 2953.20 | 2898.49 | 2895.58 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 2895.60 | 2920.03 | 2922.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 2867.50 | 2896.71 | 2909.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 2824.00 | 2821.11 | 2853.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:30:00 | 2837.00 | 2821.11 | 2853.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 2882.10 | 2786.59 | 2794.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 2885.00 | 2786.59 | 2794.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 2877.00 | 2804.67 | 2802.42 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 2787.10 | 2808.01 | 2809.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 2773.70 | 2801.15 | 2806.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2798.90 | 2785.32 | 2795.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2798.90 | 2785.32 | 2795.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2798.90 | 2785.32 | 2795.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 2826.40 | 2785.32 | 2795.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 2793.60 | 2786.97 | 2795.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 2798.00 | 2786.97 | 2795.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 2787.80 | 2787.14 | 2794.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 2780.00 | 2790.06 | 2794.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 2775.00 | 2735.84 | 2732.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 2775.00 | 2735.84 | 2732.54 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 2683.30 | 2730.51 | 2731.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 2645.60 | 2713.53 | 2723.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 2599.00 | 2596.10 | 2635.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 09:15:00 | 2641.50 | 2596.10 | 2635.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2622.80 | 2601.44 | 2634.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 2549.80 | 2613.41 | 2626.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 2586.00 | 2587.61 | 2608.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 2627.50 | 2615.79 | 2615.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 2627.50 | 2615.79 | 2615.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 2627.50 | 2615.79 | 2615.34 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 2603.30 | 2614.26 | 2614.89 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 2626.30 | 2616.67 | 2615.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2674.70 | 2635.56 | 2626.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 2677.00 | 2683.13 | 2661.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 2677.00 | 2683.13 | 2661.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 2677.00 | 2683.13 | 2661.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 2677.00 | 2683.13 | 2661.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 3165.00 | 3219.20 | 3199.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 3165.00 | 3219.20 | 3199.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 3169.00 | 3209.16 | 3196.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 12:15:00 | 3170.00 | 3209.16 | 3196.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 13:15:00 | 3182.00 | 3199.65 | 3193.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 3190.40 | 3273.53 | 3280.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 3190.40 | 3273.53 | 3280.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 09:15:00 | 3190.40 | 3273.53 | 3280.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 3167.30 | 3252.29 | 3270.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 09:15:00 | 3180.00 | 3165.71 | 3194.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 3180.00 | 3165.71 | 3194.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 3180.00 | 3165.71 | 3194.14 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-11 13:15:00 | 3133.60 | 2025-07-14 09:15:00 | 3085.20 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-07-25 10:00:00 | 2994.00 | 2025-07-28 14:15:00 | 3022.30 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-07-28 10:00:00 | 2998.90 | 2025-07-29 13:15:00 | 3010.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-28 10:45:00 | 2995.80 | 2025-07-29 13:15:00 | 3010.60 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-07-28 11:15:00 | 2993.60 | 2025-07-29 13:15:00 | 3010.60 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-28 12:15:00 | 2973.00 | 2025-07-29 14:15:00 | 3033.60 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-29 10:00:00 | 2967.80 | 2025-07-29 14:15:00 | 3033.60 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-07-29 10:45:00 | 2965.80 | 2025-07-29 14:15:00 | 3033.60 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-07-29 11:30:00 | 2974.30 | 2025-07-29 14:15:00 | 3033.60 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-08-04 11:15:00 | 3202.90 | 2025-08-06 11:15:00 | 3162.10 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-06 10:15:00 | 3184.80 | 2025-08-06 11:15:00 | 3162.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-08-08 10:30:00 | 3125.90 | 2025-08-12 11:15:00 | 3200.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-08-11 11:30:00 | 3136.30 | 2025-08-12 11:15:00 | 3200.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-11 13:00:00 | 3139.20 | 2025-08-12 11:15:00 | 3200.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-09-01 09:15:00 | 3469.60 | 2025-09-03 15:15:00 | 3441.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-09-03 15:15:00 | 3441.00 | 2025-09-03 15:15:00 | 3441.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-09-05 14:15:00 | 3421.70 | 2025-09-08 12:15:00 | 3470.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-05 15:15:00 | 3405.10 | 2025-09-08 12:15:00 | 3470.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-09-09 15:00:00 | 3460.20 | 2025-09-10 09:15:00 | 3400.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-12 14:15:00 | 3389.00 | 2025-09-15 09:15:00 | 3429.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-09-25 10:15:00 | 3537.70 | 2025-09-29 09:15:00 | 3478.50 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-09-25 13:15:00 | 3551.80 | 2025-09-29 09:15:00 | 3478.50 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-09-26 09:45:00 | 3574.40 | 2025-09-29 09:15:00 | 3478.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-10-08 12:00:00 | 3263.00 | 2025-10-08 12:15:00 | 3344.90 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest1 | 2025-10-13 09:15:00 | 3229.00 | 2025-10-21 13:15:00 | 3137.40 | STOP_HIT | 1.00 | 2.84% |
| SELL | retest2 | 2025-10-15 09:15:00 | 3176.00 | 2025-10-23 09:15:00 | 3162.20 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-10-27 13:15:00 | 3113.50 | 2025-10-27 14:15:00 | 3152.80 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-11 14:30:00 | 3262.20 | 2025-11-14 12:15:00 | 3248.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-11-12 09:15:00 | 3275.30 | 2025-11-14 12:15:00 | 3248.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-12 10:30:00 | 3285.00 | 2025-11-14 12:15:00 | 3248.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-14 11:15:00 | 3265.00 | 2025-11-14 12:15:00 | 3248.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-11-20 09:15:00 | 3359.20 | 2025-11-24 14:15:00 | 3146.00 | STOP_HIT | 1.00 | -6.35% |
| BUY | retest2 | 2025-11-21 12:15:00 | 3331.60 | 2025-11-24 14:15:00 | 3146.00 | STOP_HIT | 1.00 | -5.57% |
| BUY | retest2 | 2025-11-24 12:30:00 | 3335.40 | 2025-11-24 14:15:00 | 3146.00 | STOP_HIT | 1.00 | -5.68% |
| BUY | retest2 | 2025-11-24 13:00:00 | 3331.10 | 2025-11-24 14:15:00 | 3146.00 | STOP_HIT | 1.00 | -5.56% |
| SELL | retest1 | 2025-11-26 14:00:00 | 3106.50 | 2025-11-28 09:15:00 | 3207.20 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest1 | 2025-11-26 14:30:00 | 3105.20 | 2025-11-28 09:15:00 | 3207.20 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-12-08 10:30:00 | 3070.50 | 2025-12-11 14:15:00 | 2916.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 10:30:00 | 3070.50 | 2025-12-15 09:15:00 | 2966.20 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2025-12-29 12:30:00 | 2596.00 | 2026-01-05 12:15:00 | 2572.10 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2026-01-07 15:00:00 | 2606.30 | 2026-01-08 11:15:00 | 2568.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-01-08 09:15:00 | 2627.60 | 2026-01-08 11:15:00 | 2568.90 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-01-14 10:45:00 | 2388.50 | 2026-01-20 12:15:00 | 2269.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:45:00 | 2388.50 | 2026-01-22 09:15:00 | 2268.00 | STOP_HIT | 0.50 | 5.05% |
| BUY | retest2 | 2026-02-02 13:15:00 | 2418.90 | 2026-02-02 13:15:00 | 2388.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-02-17 12:30:00 | 2839.30 | 2026-02-19 09:15:00 | 2731.90 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2026-02-23 09:15:00 | 2808.20 | 2026-03-05 09:15:00 | 2860.50 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2026-02-23 15:00:00 | 2814.50 | 2026-03-05 09:15:00 | 2860.50 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2026-03-09 14:15:00 | 2874.90 | 2026-03-10 09:15:00 | 2953.20 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-03-09 14:45:00 | 2877.90 | 2026-03-10 09:15:00 | 2953.20 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2026-03-20 15:15:00 | 2780.00 | 2026-03-25 14:15:00 | 2775.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-04-02 09:15:00 | 2549.80 | 2026-04-06 13:15:00 | 2627.50 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-04-02 12:30:00 | 2586.00 | 2026-04-06 13:15:00 | 2627.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-04-24 12:15:00 | 3170.00 | 2026-05-06 09:15:00 | 3190.40 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2026-04-24 13:15:00 | 3182.00 | 2026-05-06 09:15:00 | 3190.40 | STOP_HIT | 1.00 | 0.26% |
