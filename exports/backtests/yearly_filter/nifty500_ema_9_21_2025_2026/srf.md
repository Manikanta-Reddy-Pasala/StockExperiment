# SRF Ltd. (SRF)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2778.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 85 |
| ALERT1 | 54 |
| ALERT2 | 53 |
| ALERT2_SKIP | 28 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 38 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 31
- **Target hits / Stop hits / Partials:** 0 / 39 / 1
- **Avg / median % per leg:** -0.52% / -1.22%
- **Sum % (uncompounded):** -20.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 0 | 13 | 0 | 0.17% | 2.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.85% | -0.8% |
| BUY @ 3rd Alert (retest2) | 12 | 6 | 50.0% | 0 | 12 | 0 | 0.25% | 3.0% |
| SELL (all) | 27 | 3 | 11.1% | 0 | 26 | 1 | -0.86% | -23.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 3 | 11.1% | 0 | 26 | 1 | -0.86% | -23.1% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.85% | -0.8% |
| retest2 (combined) | 39 | 9 | 23.1% | 0 | 38 | 1 | -0.51% | -20.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3022.20 | 3005.94 | 3004.60 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 09:15:00 | 2955.00 | 2999.95 | 3003.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 2912.20 | 2964.56 | 2985.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 09:15:00 | 2922.50 | 2889.23 | 2904.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 2922.50 | 2889.23 | 2904.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 2922.50 | 2889.23 | 2904.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 2922.50 | 2889.23 | 2904.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 2908.00 | 2892.98 | 2904.65 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 2975.10 | 2921.56 | 2914.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 3019.20 | 2941.09 | 2924.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 2982.80 | 2983.80 | 2961.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 12:45:00 | 2976.00 | 2983.80 | 2961.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 2952.10 | 2975.27 | 2961.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 2945.90 | 2975.27 | 2961.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 2950.30 | 2970.28 | 2960.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 2957.00 | 2970.28 | 2960.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2959.80 | 2972.63 | 2964.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 2959.80 | 2972.63 | 2964.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 2967.20 | 2971.54 | 2964.63 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 2912.30 | 2953.82 | 2958.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 15:15:00 | 2901.00 | 2918.19 | 2935.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 2919.40 | 2918.43 | 2933.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 2919.40 | 2918.43 | 2933.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2919.40 | 2918.43 | 2933.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 2904.90 | 2918.43 | 2933.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 2904.90 | 2915.12 | 2929.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:15:00 | 2898.50 | 2911.07 | 2925.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:30:00 | 2897.00 | 2904.67 | 2918.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 2914.40 | 2908.43 | 2917.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:30:00 | 2906.30 | 2907.84 | 2916.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 11:15:00 | 2909.00 | 2894.85 | 2899.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:00:00 | 2904.50 | 2896.78 | 2900.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:30:00 | 2908.80 | 2898.43 | 2900.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 2897.50 | 2900.41 | 2901.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 15:15:00 | 2890.00 | 2900.41 | 2901.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:00:00 | 2893.50 | 2890.94 | 2893.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 2944.30 | 2891.64 | 2885.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 2944.30 | 2891.64 | 2885.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 2987.90 | 2949.87 | 2923.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 3122.00 | 3126.16 | 3103.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:15:00 | 3136.30 | 3126.16 | 3103.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 3109.70 | 3130.02 | 3111.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-10 11:15:00 | 3109.70 | 3130.02 | 3111.64 | SL hit (close<ema400) qty=1.00 sl=3111.64 alert=retest1 |

### Cycle 6 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 3068.60 | 3100.74 | 3103.01 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 12:15:00 | 3120.00 | 3100.55 | 3098.31 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 11:15:00 | 3089.00 | 3098.67 | 3099.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 12:15:00 | 3081.00 | 3095.14 | 3097.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 3101.40 | 3096.39 | 3098.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 13:15:00 | 3101.40 | 3096.39 | 3098.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 3101.40 | 3096.39 | 3098.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 3101.40 | 3096.39 | 3098.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 3096.90 | 3096.49 | 3098.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:15:00 | 3104.70 | 3096.49 | 3098.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 3104.70 | 3098.13 | 3098.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 3127.00 | 3098.13 | 3098.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 3097.70 | 3098.05 | 3098.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 3114.60 | 3098.05 | 3098.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 3102.70 | 3098.98 | 3098.96 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 3088.90 | 3098.93 | 3099.46 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 3114.60 | 3102.06 | 3100.84 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 3088.70 | 3099.39 | 3099.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 3075.10 | 3089.57 | 3094.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 3031.20 | 3024.68 | 3040.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 3031.20 | 3024.68 | 3040.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 3031.20 | 3024.68 | 3040.38 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 3077.70 | 3045.72 | 3043.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 3091.40 | 3060.12 | 3052.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 3069.60 | 3073.77 | 3064.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 10:15:00 | 3069.60 | 3073.77 | 3064.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 3069.60 | 3073.77 | 3064.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 3069.60 | 3073.77 | 3064.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 3190.50 | 3197.55 | 3183.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:45:00 | 3255.60 | 3208.84 | 3189.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 3222.50 | 3232.67 | 3231.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 15:15:00 | 3252.80 | 3262.15 | 3262.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 3252.80 | 3262.15 | 3262.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 3244.00 | 3258.52 | 3260.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 3182.30 | 3178.56 | 3199.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 3182.30 | 3178.56 | 3199.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3182.30 | 3178.56 | 3199.87 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 3227.30 | 3206.14 | 3205.07 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 3196.00 | 3209.94 | 3210.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 3185.80 | 3201.87 | 3205.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 12:15:00 | 3135.30 | 3130.19 | 3155.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 13:00:00 | 3135.30 | 3130.19 | 3155.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 3117.20 | 3130.51 | 3147.92 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 3206.40 | 3158.47 | 3153.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 3229.70 | 3172.71 | 3160.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 3178.20 | 3179.38 | 3166.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 11:15:00 | 3178.20 | 3179.38 | 3166.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 3178.20 | 3179.38 | 3166.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 3178.20 | 3179.38 | 3166.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 3161.70 | 3176.58 | 3167.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 3161.70 | 3176.58 | 3167.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 3144.30 | 3170.12 | 3165.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 3144.30 | 3170.12 | 3165.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 3141.00 | 3164.30 | 3162.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 3062.40 | 3164.30 | 3162.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 3082.20 | 3147.88 | 3155.56 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 3109.00 | 3095.05 | 3094.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 3137.50 | 3103.54 | 3098.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 12:15:00 | 3100.60 | 3107.40 | 3101.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 12:15:00 | 3100.60 | 3107.40 | 3101.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 3100.60 | 3107.40 | 3101.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 3100.60 | 3107.40 | 3101.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 3094.20 | 3104.76 | 3101.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 3086.20 | 3104.76 | 3101.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 3084.80 | 3100.77 | 3099.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 3084.80 | 3100.77 | 3099.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 3086.20 | 3097.86 | 3098.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 3020.10 | 3082.31 | 3091.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 12:15:00 | 3069.60 | 3066.96 | 3080.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:45:00 | 3073.40 | 3066.96 | 3080.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 3049.50 | 3018.46 | 3038.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 3049.50 | 3018.46 | 3038.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 3058.40 | 3026.44 | 3040.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 3058.40 | 3026.44 | 3040.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 3077.30 | 3052.63 | 3049.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 3087.50 | 3059.60 | 3052.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 3049.90 | 3058.49 | 3053.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 13:15:00 | 3049.90 | 3058.49 | 3053.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 3049.90 | 3058.49 | 3053.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 3049.90 | 3058.49 | 3053.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 3050.30 | 3056.86 | 3053.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 3050.30 | 3056.86 | 3053.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 2974.50 | 3036.49 | 3044.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 2919.90 | 2974.18 | 3003.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 2872.00 | 2868.98 | 2903.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:45:00 | 2873.70 | 2868.98 | 2903.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 2871.80 | 2859.14 | 2873.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 2842.60 | 2854.60 | 2864.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:00:00 | 2843.90 | 2850.41 | 2860.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 2845.00 | 2847.80 | 2857.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 2911.50 | 2864.02 | 2862.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 2911.50 | 2864.02 | 2862.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 2931.00 | 2892.41 | 2876.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 12:15:00 | 2924.10 | 2926.93 | 2904.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:00:00 | 2924.10 | 2926.93 | 2904.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2916.50 | 2925.49 | 2910.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 2916.50 | 2925.49 | 2910.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2936.70 | 2927.73 | 2913.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:15:00 | 2941.40 | 2927.73 | 2913.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:00:00 | 2940.00 | 2928.95 | 2916.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 2953.70 | 2924.63 | 2921.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 2944.00 | 2924.63 | 2921.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 2915.00 | 2930.05 | 2926.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 2915.00 | 2930.05 | 2926.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 2910.00 | 2926.04 | 2925.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 2867.00 | 2926.04 | 2925.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 2901.30 | 2921.09 | 2922.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 2901.30 | 2921.09 | 2922.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 2853.30 | 2896.84 | 2909.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 2857.50 | 2852.25 | 2870.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 2857.50 | 2852.25 | 2870.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 2863.20 | 2854.44 | 2869.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 2864.70 | 2854.44 | 2869.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2864.50 | 2852.04 | 2862.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 2863.20 | 2852.04 | 2862.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2877.80 | 2857.19 | 2863.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 2872.30 | 2857.19 | 2863.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 2872.30 | 2860.21 | 2864.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:15:00 | 2878.70 | 2860.21 | 2864.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 2888.00 | 2869.45 | 2868.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 2899.60 | 2875.48 | 2870.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 2925.00 | 2931.98 | 2917.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:15:00 | 2918.30 | 2931.98 | 2917.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 2914.60 | 2928.50 | 2917.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 2914.60 | 2928.50 | 2917.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 2899.60 | 2922.72 | 2915.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 2899.60 | 2922.72 | 2915.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 2885.70 | 2915.32 | 2913.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 2885.70 | 2915.32 | 2913.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 2887.80 | 2909.81 | 2910.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 2867.30 | 2901.31 | 2906.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 2876.30 | 2867.13 | 2882.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 11:15:00 | 2887.50 | 2872.64 | 2882.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 2887.50 | 2872.64 | 2882.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 2887.50 | 2872.64 | 2882.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 2891.40 | 2876.39 | 2883.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 2891.40 | 2876.39 | 2883.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 2896.60 | 2880.43 | 2884.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 2896.60 | 2880.43 | 2884.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 2885.00 | 2882.06 | 2884.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 2890.50 | 2882.06 | 2884.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2897.60 | 2885.17 | 2885.66 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 2893.00 | 2887.11 | 2886.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 2916.30 | 2892.95 | 2889.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 2964.00 | 2965.38 | 2944.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 2964.00 | 2965.38 | 2944.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 2970.50 | 2976.50 | 2964.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 2965.00 | 2976.50 | 2964.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 2973.50 | 2976.44 | 2966.81 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 09:15:00 | 2938.70 | 2960.12 | 2963.00 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 2980.00 | 2960.56 | 2959.15 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 2949.70 | 2959.33 | 2959.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 2932.90 | 2951.11 | 2955.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 2939.50 | 2935.57 | 2944.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 14:45:00 | 2936.10 | 2935.57 | 2944.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2953.00 | 2938.97 | 2944.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:15:00 | 2955.00 | 2938.97 | 2944.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 2950.50 | 2941.27 | 2945.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 2954.00 | 2941.27 | 2945.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 2943.10 | 2940.68 | 2943.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 2943.10 | 2940.68 | 2943.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 2939.50 | 2940.45 | 2943.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:15:00 | 2947.30 | 2940.45 | 2943.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 2947.30 | 2941.82 | 2943.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 2946.60 | 2941.82 | 2943.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 2938.90 | 2941.23 | 2943.41 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 2951.20 | 2944.92 | 2944.81 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 2928.90 | 2943.36 | 2944.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 2916.00 | 2937.89 | 2941.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 2907.80 | 2907.28 | 2921.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 2907.80 | 2907.28 | 2921.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 2911.10 | 2906.36 | 2915.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 2904.50 | 2906.49 | 2914.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:30:00 | 2903.30 | 2908.03 | 2914.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 2896.30 | 2908.82 | 2913.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:45:00 | 2903.80 | 2907.94 | 2913.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 2917.00 | 2909.75 | 2913.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-25 10:15:00 | 2917.00 | 2909.75 | 2913.43 | SL hit (close>static) qty=1.00 sl=2916.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 2864.40 | 2832.32 | 2830.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 2879.80 | 2841.82 | 2835.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 2970.00 | 2972.61 | 2953.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 15:00:00 | 2970.00 | 2972.61 | 2953.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3022.00 | 3011.18 | 2995.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 3003.90 | 3011.18 | 2995.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 3170.00 | 3182.74 | 3168.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 3173.30 | 3182.74 | 3168.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 3144.00 | 3174.99 | 3166.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 3144.00 | 3174.99 | 3166.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 3146.10 | 3169.21 | 3164.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 3146.10 | 3169.21 | 3164.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 3123.10 | 3159.99 | 3160.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 3116.10 | 3151.21 | 3156.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 3078.20 | 3074.45 | 3101.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 15:00:00 | 3078.20 | 3074.45 | 3101.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3024.20 | 3065.61 | 3092.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 3002.50 | 3019.93 | 3033.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 2934.70 | 2906.42 | 2904.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 2934.70 | 2906.42 | 2904.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 2941.80 | 2921.66 | 2913.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 2925.60 | 2930.38 | 2921.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 2925.60 | 2930.38 | 2921.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 2905.70 | 2924.57 | 2920.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 2905.70 | 2924.57 | 2920.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 2903.70 | 2920.39 | 2918.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 2903.70 | 2920.39 | 2918.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 2889.00 | 2914.11 | 2916.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 2857.20 | 2902.73 | 2910.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 2831.90 | 2831.79 | 2845.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 14:00:00 | 2831.90 | 2831.79 | 2845.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2812.00 | 2800.02 | 2816.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 2806.40 | 2800.02 | 2816.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 2836.70 | 2807.35 | 2817.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 2838.50 | 2807.35 | 2817.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 2841.00 | 2814.08 | 2820.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 2846.00 | 2814.08 | 2820.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 2870.70 | 2832.86 | 2828.01 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 2823.60 | 2832.68 | 2833.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 2805.20 | 2827.18 | 2830.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 2812.80 | 2803.49 | 2811.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 2812.80 | 2803.49 | 2811.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2812.80 | 2803.49 | 2811.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 2812.80 | 2803.49 | 2811.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 2811.00 | 2804.99 | 2811.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 2806.70 | 2806.64 | 2811.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 2854.30 | 2817.18 | 2814.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 2854.30 | 2817.18 | 2814.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 2919.80 | 2850.13 | 2833.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 2897.70 | 2904.41 | 2876.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:45:00 | 2898.20 | 2904.41 | 2876.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 2887.00 | 2905.99 | 2891.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 2890.20 | 2905.99 | 2891.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 2887.50 | 2902.29 | 2890.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 2884.30 | 2902.29 | 2890.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 2877.80 | 2897.39 | 2889.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 2877.80 | 2897.39 | 2889.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 2855.80 | 2880.36 | 2883.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 2840.90 | 2867.20 | 2875.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 2846.80 | 2838.65 | 2853.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 2846.80 | 2838.65 | 2853.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 2840.20 | 2839.61 | 2848.12 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 2878.70 | 2856.71 | 2854.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 2887.90 | 2867.62 | 2859.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 2842.90 | 2862.68 | 2858.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 2842.90 | 2862.68 | 2858.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 2842.90 | 2862.68 | 2858.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 2842.90 | 2862.68 | 2858.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 2846.00 | 2859.34 | 2857.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 2846.00 | 2859.34 | 2857.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 2839.90 | 2855.45 | 2855.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 2822.80 | 2848.92 | 2852.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 2838.80 | 2838.04 | 2845.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:15:00 | 2863.00 | 2838.04 | 2845.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 2888.30 | 2848.09 | 2849.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 2888.30 | 2848.09 | 2849.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 2886.40 | 2855.75 | 2852.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 13:15:00 | 2895.70 | 2868.90 | 2859.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 13:15:00 | 2992.60 | 2997.05 | 2975.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:00:00 | 2992.60 | 2997.05 | 2975.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3011.30 | 3011.01 | 2998.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:45:00 | 3016.90 | 3012.36 | 3001.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 13:00:00 | 3020.50 | 3013.99 | 3002.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 3016.50 | 3014.49 | 3004.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:30:00 | 3020.00 | 3018.57 | 3007.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 3104.90 | 3092.74 | 3076.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 3087.00 | 3103.36 | 3103.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 3087.00 | 3103.36 | 3103.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 3080.30 | 3094.58 | 3099.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 3067.00 | 3047.67 | 3066.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 3067.00 | 3047.67 | 3066.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 3067.00 | 3047.67 | 3066.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 3083.10 | 3047.67 | 3066.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 3071.20 | 3052.38 | 3066.94 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 3077.10 | 3064.46 | 3064.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 3085.50 | 3068.67 | 3066.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 3058.00 | 3074.41 | 3071.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 3058.00 | 3074.41 | 3071.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 3058.00 | 3074.41 | 3071.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 3052.60 | 3074.41 | 3071.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 2976.30 | 3054.78 | 3062.44 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 3080.50 | 3060.30 | 3059.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 3090.40 | 3071.43 | 3065.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 3058.80 | 3075.66 | 3069.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 3058.80 | 3075.66 | 3069.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 3058.80 | 3075.66 | 3069.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 3058.80 | 3075.66 | 3069.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 3046.80 | 3069.89 | 3067.41 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 3048.50 | 3065.61 | 3065.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 3041.30 | 3056.25 | 3060.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 3039.90 | 3031.34 | 3041.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 3039.90 | 3031.34 | 3041.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 3039.90 | 3031.34 | 3041.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 3039.90 | 3031.34 | 3041.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3011.30 | 3027.33 | 3039.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 3010.80 | 3027.33 | 3039.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 3051.30 | 3037.79 | 3039.01 | SL hit (close>static) qty=1.00 sl=3045.10 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 3055.90 | 3039.78 | 3039.40 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 14:15:00 | 3033.90 | 3038.95 | 3039.49 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 3059.50 | 3041.08 | 3040.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 3065.00 | 3048.68 | 3044.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 3023.20 | 3043.75 | 3042.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 3023.20 | 3043.75 | 3042.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 3023.20 | 3043.75 | 3042.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 3023.20 | 3043.75 | 3042.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 3027.50 | 3040.50 | 3041.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 2994.00 | 3029.10 | 3035.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 13:15:00 | 2733.20 | 2730.75 | 2792.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 14:00:00 | 2733.20 | 2730.75 | 2792.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 2737.30 | 2697.27 | 2717.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 2738.00 | 2697.27 | 2717.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 2772.90 | 2712.39 | 2722.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 2772.90 | 2712.39 | 2722.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 2773.50 | 2733.70 | 2731.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 2802.10 | 2753.57 | 2741.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 2801.00 | 2801.34 | 2779.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 2801.00 | 2801.34 | 2779.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 2800.20 | 2805.20 | 2789.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:30:00 | 2799.60 | 2805.20 | 2789.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2815.00 | 2808.03 | 2794.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:45:00 | 2847.40 | 2821.63 | 2801.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 2783.00 | 2818.39 | 2804.15 | SL hit (close<static) qty=1.00 sl=2792.30 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 2717.00 | 2790.02 | 2793.22 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2929.60 | 2808.92 | 2794.61 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 2928.80 | 2940.16 | 2941.66 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 2948.70 | 2942.80 | 2942.66 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 2933.70 | 2940.98 | 2941.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 2905.90 | 2933.96 | 2938.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 2838.10 | 2831.95 | 2853.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 2838.10 | 2831.95 | 2853.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 2847.50 | 2838.14 | 2851.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 2847.50 | 2838.14 | 2851.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 2845.00 | 2839.51 | 2850.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 2837.40 | 2839.51 | 2850.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 2867.50 | 2845.11 | 2852.35 | SL hit (close>static) qty=1.00 sl=2851.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 2876.00 | 2856.86 | 2856.34 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 2768.50 | 2839.19 | 2848.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 14:15:00 | 2751.50 | 2821.65 | 2839.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 2701.60 | 2696.79 | 2731.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 2671.50 | 2689.08 | 2710.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2671.50 | 2689.08 | 2710.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 2667.30 | 2684.26 | 2706.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2533.93 | 2570.99 | 2590.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 12:15:00 | 2533.10 | 2530.62 | 2551.66 | SL hit (close>ema200) qty=0.50 sl=2530.62 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 2564.90 | 2552.83 | 2552.78 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 14:15:00 | 2550.80 | 2552.42 | 2552.60 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 2571.00 | 2556.14 | 2554.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 2618.20 | 2568.55 | 2560.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2554.80 | 2600.59 | 2586.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2554.80 | 2600.59 | 2586.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2554.80 | 2600.59 | 2586.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 2562.50 | 2600.59 | 2586.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 2550.20 | 2590.51 | 2583.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:30:00 | 2544.10 | 2590.51 | 2583.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 2546.80 | 2574.02 | 2576.35 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 2582.10 | 2574.41 | 2574.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 2598.10 | 2581.07 | 2577.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 2567.10 | 2583.23 | 2579.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 2567.10 | 2583.23 | 2579.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 2567.10 | 2583.23 | 2579.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 2567.10 | 2583.23 | 2579.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 2543.60 | 2575.30 | 2576.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 2526.50 | 2560.77 | 2569.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 2573.90 | 2534.85 | 2549.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 2573.90 | 2534.85 | 2549.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 2573.90 | 2534.85 | 2549.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 2573.90 | 2534.85 | 2549.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 2584.40 | 2544.76 | 2552.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 2584.40 | 2544.76 | 2552.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 2610.30 | 2557.87 | 2557.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 2625.60 | 2571.42 | 2563.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 2543.00 | 2580.50 | 2571.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 2543.00 | 2580.50 | 2571.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 2543.00 | 2580.50 | 2571.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 2543.00 | 2580.50 | 2571.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 2524.30 | 2569.26 | 2567.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 2523.10 | 2569.26 | 2567.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 2524.20 | 2560.25 | 2563.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 2512.30 | 2550.66 | 2558.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 2479.70 | 2472.82 | 2499.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 11:00:00 | 2479.70 | 2472.82 | 2499.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 2489.20 | 2478.84 | 2497.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 2497.70 | 2478.84 | 2497.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 2504.30 | 2483.94 | 2498.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 2504.30 | 2483.94 | 2498.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 2499.10 | 2486.97 | 2498.42 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 2564.90 | 2504.64 | 2504.59 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 2474.50 | 2518.85 | 2520.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 2466.40 | 2491.47 | 2505.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 2409.20 | 2407.16 | 2438.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 2409.20 | 2407.16 | 2438.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 2409.20 | 2407.16 | 2438.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:45:00 | 2393.00 | 2404.73 | 2434.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 2479.80 | 2442.69 | 2442.90 | SL hit (close>static) qty=1.00 sl=2470.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 2538.40 | 2461.83 | 2451.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 2568.50 | 2510.56 | 2478.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2524.80 | 2538.24 | 2504.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 2537.00 | 2538.24 | 2504.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 2492.30 | 2523.59 | 2505.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 2492.30 | 2523.59 | 2505.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 2512.00 | 2521.27 | 2506.41 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 2443.30 | 2498.21 | 2498.94 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 2530.70 | 2485.75 | 2485.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 2572.00 | 2503.00 | 2493.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 2412.00 | 2511.80 | 2505.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 2412.00 | 2511.80 | 2505.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2412.00 | 2511.80 | 2505.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 2412.00 | 2511.80 | 2505.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 2404.00 | 2490.24 | 2496.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 09:15:00 | 2374.90 | 2429.81 | 2459.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 2439.10 | 2408.89 | 2434.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 14:15:00 | 2439.10 | 2408.89 | 2434.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 2439.10 | 2408.89 | 2434.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 2439.10 | 2408.89 | 2434.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 2435.40 | 2414.20 | 2434.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 2408.20 | 2410.76 | 2431.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 2477.90 | 2409.55 | 2416.60 | SL hit (close>static) qty=1.00 sl=2442.90 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 2474.70 | 2422.58 | 2421.89 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 2399.40 | 2421.53 | 2424.51 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 11:15:00 | 2447.40 | 2427.34 | 2426.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 12:15:00 | 2461.70 | 2434.21 | 2429.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 2425.20 | 2446.69 | 2438.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 2425.20 | 2446.69 | 2438.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2425.20 | 2446.69 | 2438.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2444.70 | 2446.69 | 2438.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 12:15:00 | 2475.20 | 2485.94 | 2486.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 2475.20 | 2485.94 | 2486.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 13:15:00 | 2460.10 | 2480.77 | 2484.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 2492.40 | 2478.39 | 2481.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 2492.40 | 2478.39 | 2481.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 2492.40 | 2478.39 | 2481.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 2498.10 | 2478.39 | 2481.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 2488.00 | 2480.31 | 2482.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 11:15:00 | 2482.70 | 2480.31 | 2482.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 2496.90 | 2479.67 | 2480.45 | SL hit (close>static) qty=1.00 sl=2495.70 alert=retest2 |

### Cycle 79 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 2500.80 | 2483.90 | 2482.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 2511.40 | 2494.46 | 2488.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 2514.00 | 2523.07 | 2509.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 2514.00 | 2523.07 | 2509.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 2514.00 | 2523.07 | 2509.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 2500.80 | 2523.07 | 2509.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 2483.80 | 2515.22 | 2506.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 2483.80 | 2515.22 | 2506.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 2484.20 | 2509.01 | 2504.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 2476.00 | 2509.01 | 2504.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 2489.60 | 2499.93 | 2501.09 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 2522.50 | 2501.88 | 2501.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 2595.50 | 2533.64 | 2519.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 2543.30 | 2551.59 | 2536.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:45:00 | 2543.10 | 2551.59 | 2536.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 2496.60 | 2540.01 | 2533.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 2496.60 | 2540.01 | 2533.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 2497.00 | 2531.41 | 2530.12 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 2505.20 | 2526.17 | 2527.86 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 2555.50 | 2530.19 | 2527.99 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 2518.30 | 2531.48 | 2532.69 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 2638.00 | 2553.45 | 2542.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 2694.00 | 2581.56 | 2556.19 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-23 10:15:00 | 2904.90 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-05-23 12:15:00 | 2904.90 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-05-23 14:15:00 | 2898.50 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-05-26 09:30:00 | 2897.00 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-05-26 12:30:00 | 2906.30 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-05-28 11:15:00 | 2909.00 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-05-28 12:00:00 | 2904.50 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-28 12:30:00 | 2908.80 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-28 15:15:00 | 2890.00 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-05-30 10:00:00 | 2893.50 | 2025-06-03 09:15:00 | 2944.30 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest1 | 2025-06-10 09:15:00 | 3136.30 | 2025-06-10 11:15:00 | 3109.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-02 10:45:00 | 3255.60 | 2025-07-10 15:15:00 | 3252.80 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-07-07 09:45:00 | 3222.50 | 2025-07-10 15:15:00 | 3252.80 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-08-14 09:45:00 | 2842.60 | 2025-08-18 10:15:00 | 2911.50 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-08-14 12:00:00 | 2843.90 | 2025-08-18 10:15:00 | 2911.50 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-08-14 13:30:00 | 2845.00 | 2025-08-18 10:15:00 | 2911.50 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-08-20 11:15:00 | 2941.40 | 2025-08-26 09:15:00 | 2901.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-08-20 13:00:00 | 2940.00 | 2025-08-26 09:15:00 | 2901.30 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-25 09:30:00 | 2953.70 | 2025-08-26 09:15:00 | 2901.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-25 10:15:00 | 2944.00 | 2025-08-26 09:15:00 | 2901.30 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-24 13:15:00 | 2904.50 | 2025-09-25 10:15:00 | 2917.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-09-24 14:30:00 | 2903.30 | 2025-09-25 10:15:00 | 2917.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-09-25 09:15:00 | 2896.30 | 2025-09-25 10:15:00 | 2917.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-09-25 09:45:00 | 2903.80 | 2025-09-25 10:15:00 | 2917.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2855.90 | 2025-10-01 12:15:00 | 2864.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-10-30 09:30:00 | 3002.50 | 2025-11-11 15:15:00 | 2934.70 | STOP_HIT | 1.00 | 2.26% |
| SELL | retest2 | 2025-11-26 13:15:00 | 2806.70 | 2025-11-27 09:15:00 | 2854.30 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-12-17 11:45:00 | 3016.90 | 2025-12-29 11:15:00 | 3087.00 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2025-12-17 13:00:00 | 3020.50 | 2025-12-29 11:15:00 | 3087.00 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-12-17 14:00:00 | 3016.50 | 2025-12-29 11:15:00 | 3087.00 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-12-17 14:30:00 | 3020.00 | 2025-12-29 11:15:00 | 3087.00 | STOP_HIT | 1.00 | 2.22% |
| SELL | retest2 | 2026-01-12 11:15:00 | 3010.80 | 2026-01-13 11:15:00 | 3051.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-01 10:45:00 | 2847.40 | 2026-02-01 12:15:00 | 2783.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-02-17 09:15:00 | 2837.40 | 2026-02-17 09:15:00 | 2867.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-23 10:45:00 | 2667.30 | 2026-03-02 09:15:00 | 2533.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:45:00 | 2667.30 | 2026-03-04 12:15:00 | 2533.10 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2026-03-24 10:45:00 | 2393.00 | 2026-03-24 15:15:00 | 2479.80 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2026-04-07 09:30:00 | 2408.20 | 2026-04-08 09:15:00 | 2477.90 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2444.70 | 2026-04-20 12:15:00 | 2475.20 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2026-04-21 11:15:00 | 2482.70 | 2026-04-22 10:15:00 | 2496.90 | STOP_HIT | 1.00 | -0.57% |
