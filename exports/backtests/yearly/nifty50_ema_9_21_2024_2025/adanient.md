# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2024-03-12 09:15:00 → 2026-05-08 15:15:00 (3717 bars)
- **Last close:** 2502.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 95 |
| ALERT2 | 93 |
| ALERT2_SKIP | 39 |
| ALERT3 | 259 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 129 |
| PARTIAL | 17 |
| TARGET_HIT | 15 |
| STOP_HIT | 119 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 149 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 89
- **Target hits / Stop hits / Partials:** 15 / 117 / 17
- **Avg / median % per leg:** 1.18% / -0.44%
- **Sum % (uncompounded):** 175.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 16 | 30.2% | 9 | 44 | 0 | 1.06% | 56.2% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.12% | 1.1% |
| BUY @ 3rd Alert (retest2) | 52 | 15 | 28.8% | 9 | 43 | 0 | 1.06% | 55.0% |
| SELL (all) | 96 | 44 | 45.8% | 6 | 73 | 17 | 1.24% | 119.2% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.99% | 12.0% |
| SELL @ 3rd Alert (retest2) | 92 | 40 | 43.5% | 6 | 71 | 15 | 1.17% | 107.2% |
| retest1 (combined) | 5 | 5 | 100.0% | 0 | 3 | 2 | 2.62% | 13.1% |
| retest2 (combined) | 144 | 55 | 38.2% | 15 | 114 | 15 | 1.13% | 162.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 2792.07 | 2749.40 | 2744.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 15:15:00 | 2797.40 | 2759.00 | 2749.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 2922.08 | 2943.97 | 2899.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:00:00 | 2922.08 | 2943.97 | 2899.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 2890.03 | 2927.76 | 2905.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 2899.39 | 2927.76 | 2905.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 2949.71 | 2932.15 | 2909.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:00:00 | 2965.56 | 2940.32 | 2917.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:45:00 | 2958.29 | 2964.77 | 2947.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-23 12:15:00 | 3262.12 | 3104.97 | 3050.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 3166.29 | 3191.94 | 3194.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 3152.28 | 3184.01 | 3190.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 11:15:00 | 3180.88 | 3172.45 | 3181.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 11:15:00 | 3180.88 | 3172.45 | 3181.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 3180.88 | 3172.45 | 3181.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:45:00 | 3186.50 | 3172.45 | 3181.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 3197.41 | 3177.44 | 3183.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:00:00 | 3197.41 | 3177.44 | 3183.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 3190.58 | 3180.07 | 3183.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:00:00 | 3155.63 | 3175.18 | 3181.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 3210.93 | 3165.55 | 3163.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 3210.93 | 3165.55 | 3163.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 3255.09 | 3183.46 | 3172.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 3325.24 | 3447.86 | 3364.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 3325.24 | 3447.86 | 3364.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 3325.24 | 3447.86 | 3364.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 3180.64 | 3447.86 | 3364.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3019.99 | 3362.29 | 3333.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 3019.99 | 3362.29 | 3333.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 2827.21 | 3255.27 | 3287.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 2769.09 | 2988.84 | 3128.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 3008.17 | 2962.95 | 3058.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 3008.17 | 2962.95 | 3058.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 3110.11 | 2998.09 | 3058.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 3110.11 | 2998.09 | 3058.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 3109.62 | 3020.40 | 3062.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 11:30:00 | 3086.55 | 3033.66 | 3065.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:30:00 | 3085.48 | 3059.77 | 3070.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 11:15:00 | 3089.94 | 3074.37 | 3075.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 11:15:00 | 3099.10 | 3079.31 | 3077.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 11:15:00 | 3099.10 | 3079.31 | 3077.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 3120.77 | 3087.60 | 3081.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 3122.76 | 3130.60 | 3113.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 3122.76 | 3130.60 | 3113.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 3114.28 | 3127.34 | 3113.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 3136.72 | 3127.34 | 3113.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 3125.28 | 3126.93 | 3114.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:15:00 | 3148.16 | 3126.78 | 3116.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:00:00 | 3147.19 | 3130.86 | 3119.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 14:15:00 | 3145.98 | 3133.62 | 3121.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 3152.28 | 3123.26 | 3122.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 3146.95 | 3128.00 | 3124.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:00:00 | 3165.37 | 3135.48 | 3128.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:00:00 | 3169.68 | 3148.44 | 3137.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:45:00 | 3162.46 | 3177.67 | 3168.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:30:00 | 3160.96 | 3168.81 | 3166.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 3164.64 | 3167.97 | 3166.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 3164.69 | 3167.97 | 3166.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 3203.18 | 3175.01 | 3169.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 3204.29 | 3175.01 | 3169.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 3121.50 | 3163.74 | 3168.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 3121.50 | 3163.74 | 3168.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 3104.53 | 3140.75 | 3155.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 3084.61 | 3081.88 | 3096.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 10:30:00 | 3079.28 | 3081.88 | 3096.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 3088.15 | 3079.22 | 3088.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 3091.40 | 3079.22 | 3088.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 3072.44 | 3077.86 | 3086.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:30:00 | 3090.91 | 3077.86 | 3086.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 3080.88 | 3074.44 | 3081.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 3080.88 | 3074.44 | 3081.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 3078.11 | 3075.18 | 3081.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 3100.66 | 3075.18 | 3081.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 3096.54 | 3079.45 | 3082.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 3100.32 | 3079.45 | 3082.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 3075.64 | 3080.16 | 3082.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 3067.50 | 3080.16 | 3082.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 3069.53 | 3079.00 | 3081.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:00:00 | 3072.39 | 3077.78 | 3080.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:30:00 | 3071.81 | 3075.98 | 3079.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 3083.93 | 3075.77 | 3078.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:00:00 | 3083.93 | 3075.77 | 3078.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 3087.81 | 3078.18 | 3079.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:00:00 | 3087.81 | 3078.18 | 3079.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 3093.38 | 3081.04 | 3080.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 3093.38 | 3081.04 | 3080.36 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 3056.83 | 3076.20 | 3078.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 13:15:00 | 3054.36 | 3068.83 | 3074.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 12:15:00 | 3062.26 | 3061.15 | 3067.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-03 13:00:00 | 3062.26 | 3061.15 | 3067.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 3064.54 | 3061.82 | 3067.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 3063.23 | 3061.82 | 3067.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 3094.60 | 3068.38 | 3069.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:30:00 | 3089.75 | 3068.38 | 3069.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 15:15:00 | 3087.81 | 3072.27 | 3071.26 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 10:15:00 | 3052.62 | 3068.03 | 3069.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 12:15:00 | 3044.18 | 3053.12 | 3059.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 3052.91 | 3050.65 | 3057.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-05 15:00:00 | 3052.91 | 3050.65 | 3057.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 3050.15 | 3050.55 | 3056.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 3040.21 | 3050.55 | 3056.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 3028.77 | 3046.20 | 3053.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:30:00 | 3019.41 | 3036.31 | 3048.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 15:15:00 | 3017.04 | 3016.37 | 3033.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 11:45:00 | 3013.16 | 3020.73 | 3030.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 3017.04 | 3015.61 | 3024.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 3015.63 | 3006.27 | 3013.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 3024.41 | 3006.27 | 3013.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 3020.67 | 3009.15 | 3014.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:00:00 | 3020.67 | 3009.15 | 3014.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 3007.73 | 3008.87 | 3013.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:45:00 | 2985.34 | 3002.47 | 3009.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:30:00 | 2990.76 | 2997.82 | 3006.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 3025.33 | 2996.49 | 2994.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 3025.33 | 2996.49 | 2994.78 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 2968.56 | 2998.24 | 2999.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 2944.67 | 2981.51 | 2990.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 2943.16 | 2920.84 | 2937.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 10:15:00 | 2943.16 | 2920.84 | 2937.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 2943.16 | 2920.84 | 2937.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 2943.16 | 2920.84 | 2937.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 2917.96 | 2920.26 | 2935.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 2836.86 | 2920.26 | 2935.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 2907.58 | 2912.82 | 2929.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:00:00 | 2899.68 | 2910.20 | 2926.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 10:45:00 | 2906.66 | 2895.98 | 2906.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 2889.40 | 2894.67 | 2904.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:45:00 | 2898.08 | 2894.67 | 2904.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 2903.61 | 2892.41 | 2900.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 2908.41 | 2892.41 | 2900.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 2922.08 | 2898.34 | 2902.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 2922.08 | 2898.34 | 2902.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-26 10:15:00 | 2950.68 | 2908.81 | 2906.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 2950.68 | 2908.81 | 2906.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 2952.08 | 2917.47 | 2910.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 2984.17 | 2987.84 | 2967.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:45:00 | 3013.79 | 2995.73 | 2974.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 3047.62 | 3089.85 | 3067.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 3047.62 | 3089.85 | 3067.50 | SL hit (close<ema400) qty=1.00 sl=3067.50 alert=retest1 |

### Cycle 14 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 3014.03 | 3068.79 | 3068.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 2954.51 | 3045.93 | 3058.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3032.16 | 2988.43 | 3016.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 3032.16 | 2988.43 | 3016.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3032.16 | 2988.43 | 3016.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 3032.16 | 2988.43 | 3016.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 3028.96 | 2996.54 | 3018.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 2995.71 | 3007.42 | 3018.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 3040.74 | 3005.67 | 3014.22 | SL hit (close>static) qty=1.00 sl=3039.09 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 3066.48 | 3024.92 | 3021.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 3085.68 | 3050.07 | 3035.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 3066.87 | 3080.74 | 3062.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 3066.87 | 3080.74 | 3062.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 3075.74 | 3079.74 | 3063.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 3107.15 | 3079.74 | 3063.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 2977.43 | 3066.69 | 3067.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 2977.43 | 3066.69 | 3067.85 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 3026.93 | 2997.41 | 2996.31 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 12:15:00 | 2979.66 | 2999.20 | 3001.09 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 3027.56 | 3002.93 | 3001.00 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 2995.61 | 3010.57 | 3012.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 2980.49 | 3004.55 | 3009.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 2991.15 | 2980.61 | 2988.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 2991.15 | 2980.61 | 2988.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 2991.15 | 2980.61 | 2988.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 2991.15 | 2980.61 | 2988.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 2987.95 | 2982.08 | 2988.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 12:45:00 | 2981.85 | 2983.65 | 2988.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 13:15:00 | 2980.39 | 2983.65 | 2988.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 2937.88 | 2933.53 | 2933.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 2937.88 | 2933.53 | 2933.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 12:15:00 | 2947.91 | 2937.29 | 2934.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 2924.45 | 2937.53 | 2936.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 2924.45 | 2937.53 | 2936.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 2924.45 | 2937.53 | 2936.20 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 2922.61 | 2934.55 | 2934.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 2892.75 | 2918.65 | 2924.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 2917.18 | 2913.30 | 2920.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 12:00:00 | 2917.18 | 2913.30 | 2920.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 2907.10 | 2912.06 | 2919.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 14:00:00 | 2898.76 | 2909.40 | 2917.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:45:00 | 2899.34 | 2880.20 | 2886.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:30:00 | 2900.21 | 2887.14 | 2888.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 2889.94 | 2868.59 | 2872.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 2887.27 | 2875.74 | 2875.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 2887.27 | 2875.74 | 2875.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 2921.06 | 2886.79 | 2881.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 2895.27 | 2896.07 | 2888.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 2895.27 | 2896.07 | 2888.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 2891.00 | 2895.06 | 2888.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 2893.91 | 2895.06 | 2888.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 2877.48 | 2891.54 | 2887.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 2877.48 | 2891.54 | 2887.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 2869.87 | 2887.21 | 2886.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 2869.87 | 2887.21 | 2886.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 2867.54 | 2883.27 | 2884.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 2854.16 | 2875.57 | 2880.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 2853.53 | 2844.56 | 2859.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 2853.53 | 2844.56 | 2859.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 2884.99 | 2853.72 | 2860.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 2884.99 | 2853.72 | 2860.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 2890.37 | 2861.05 | 2863.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 2892.41 | 2861.05 | 2863.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 2908.36 | 2870.51 | 2867.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 2926.05 | 2881.62 | 2872.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 12:15:00 | 3007.34 | 3010.05 | 2988.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 13:00:00 | 3007.34 | 3010.05 | 2988.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 3033.76 | 3048.20 | 3032.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:30:00 | 3040.26 | 3048.20 | 3032.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 3042.20 | 3047.00 | 3033.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:30:00 | 3026.78 | 3047.00 | 3033.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 3035.46 | 3044.69 | 3033.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 3023.87 | 3044.69 | 3033.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 3036.14 | 3042.98 | 3033.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 3036.14 | 3042.98 | 3033.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 3021.93 | 3038.77 | 3032.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:00:00 | 3021.93 | 3038.77 | 3032.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 3036.43 | 3038.30 | 3033.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 12:15:00 | 3048.64 | 3038.30 | 3033.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3008.31 | 3039.98 | 3042.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 3008.31 | 3039.98 | 3042.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 2969.24 | 3010.47 | 3023.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2955.96 | 2946.79 | 2977.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 2955.96 | 2946.79 | 2977.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 2968.51 | 2951.14 | 2976.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 2972.93 | 2951.14 | 2976.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 3011.03 | 2963.11 | 2979.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 3011.03 | 2963.11 | 2979.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 3048.06 | 2980.10 | 2986.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 3048.06 | 2980.10 | 2986.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 3049.47 | 2993.98 | 2991.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 3073.27 | 3021.07 | 3005.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 2989.89 | 3039.05 | 3028.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 09:15:00 | 2989.89 | 3039.05 | 3028.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 2989.89 | 3039.05 | 3028.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:30:00 | 2976.90 | 3039.05 | 3028.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 3003.17 | 3031.87 | 3025.80 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 2995.71 | 3017.63 | 3019.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 2982.43 | 3010.59 | 3016.53 | Break + close below crossover candle low |

### Cycle 29 — BUY (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 14:15:00 | 3076.85 | 3023.84 | 3022.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 15:15:00 | 3097.31 | 3038.54 | 3028.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 3039.34 | 3045.35 | 3034.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 10:15:00 | 3039.34 | 3045.35 | 3034.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 3039.34 | 3045.35 | 3034.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 3039.34 | 3045.35 | 3034.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 3039.34 | 3044.15 | 3034.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:15:00 | 3043.31 | 3044.15 | 3034.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:30:00 | 3047.96 | 3040.53 | 3036.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 3014.81 | 3032.30 | 3033.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 3014.81 | 3032.30 | 3033.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 14:15:00 | 3007.34 | 3024.65 | 3029.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 11:15:00 | 3019.90 | 3016.70 | 3023.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 11:15:00 | 3019.90 | 3016.70 | 3023.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 3019.90 | 3016.70 | 3023.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:45:00 | 3024.11 | 3016.70 | 3023.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 3009.43 | 3015.25 | 3022.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:45:00 | 3021.45 | 3015.25 | 3022.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 3016.36 | 3015.47 | 3021.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:45:00 | 3017.67 | 3015.47 | 3021.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 3004.87 | 3013.35 | 3020.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:30:00 | 3010.54 | 3013.35 | 3020.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 3008.55 | 3012.39 | 3018.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:15:00 | 3002.20 | 3012.39 | 3018.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 3002.06 | 3010.32 | 3017.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:30:00 | 2988.10 | 3002.80 | 3012.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 2992.61 | 2996.32 | 3006.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:15:00 | 2842.98 | 2891.58 | 2921.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 12:15:00 | 2838.69 | 2880.38 | 2913.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 2785.86 | 2783.00 | 2825.29 | SL hit (close>ema200) qty=0.50 sl=2783.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 2774.47 | 2710.82 | 2706.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 2844.47 | 2744.66 | 2723.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 2851.89 | 2855.12 | 2814.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 13:00:00 | 2851.89 | 2855.12 | 2814.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 2830.90 | 2853.41 | 2831.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 2830.90 | 2853.41 | 2831.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 2784.26 | 2839.58 | 2827.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 2784.26 | 2839.58 | 2827.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 2794.98 | 2830.66 | 2824.55 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 2796.53 | 2817.10 | 2819.03 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 2828.52 | 2817.02 | 2816.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 2885.19 | 2830.65 | 2822.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 2874.81 | 2905.85 | 2873.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 2874.81 | 2905.85 | 2873.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 2874.81 | 2905.85 | 2873.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 2874.81 | 2905.85 | 2873.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 2874.28 | 2899.54 | 2873.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 2868.71 | 2899.54 | 2873.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 2876.37 | 2894.91 | 2873.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 2865.75 | 2894.91 | 2873.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 2891.00 | 2894.12 | 2875.48 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 2853.97 | 2869.68 | 2871.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 2842.58 | 2864.26 | 2868.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 2753.00 | 2742.37 | 2757.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 12:00:00 | 2753.00 | 2742.37 | 2757.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 2761.14 | 2746.12 | 2758.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 2762.06 | 2746.12 | 2758.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 2749.80 | 2746.86 | 2757.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 2749.80 | 2746.86 | 2757.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 2790.27 | 2751.86 | 2756.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 2790.27 | 2751.86 | 2756.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 2794.35 | 2760.36 | 2760.05 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 2728.13 | 2760.21 | 2761.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 2188.27 | 2641.40 | 2706.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 2213.67 | 2186.13 | 2303.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 15:00:00 | 2186.14 | 2214.13 | 2275.59 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 09:30:00 | 2180.95 | 2194.47 | 2255.81 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 13:15:00 | 2076.83 | 2143.44 | 2209.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 14:15:00 | 2071.90 | 2132.74 | 2198.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-27 09:15:00 | 2161.95 | 2133.46 | 2187.37 | SL hit (close>ema200) qty=0.50 sl=2133.46 alert=retest1 |

### Cycle 37 — BUY (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 13:15:00 | 2309.31 | 2207.86 | 2207.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 2424.29 | 2285.11 | 2246.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 10:15:00 | 2330.74 | 2353.02 | 2313.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:00:00 | 2330.74 | 2353.02 | 2313.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2409.17 | 2418.41 | 2398.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 2401.41 | 2418.41 | 2398.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2398.51 | 2414.11 | 2404.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 2398.51 | 2414.11 | 2404.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2412.08 | 2413.70 | 2405.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 2437.87 | 2418.54 | 2408.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 2431.47 | 2429.12 | 2426.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:45:00 | 2423.71 | 2427.00 | 2426.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 15:15:00 | 2418.67 | 2425.33 | 2425.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 15:15:00 | 2418.67 | 2425.33 | 2425.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 09:15:00 | 2404.27 | 2421.12 | 2423.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 10:15:00 | 2435.35 | 2393.60 | 2398.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 10:15:00 | 2435.35 | 2393.60 | 2398.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 2435.35 | 2393.60 | 2398.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 2435.35 | 2393.60 | 2398.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 2416.93 | 2398.27 | 2399.83 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 12:15:00 | 2446.50 | 2407.91 | 2404.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 13:15:00 | 2459.92 | 2437.68 | 2423.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 09:15:00 | 2441.99 | 2442.00 | 2429.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-16 10:00:00 | 2441.99 | 2442.00 | 2429.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 2438.64 | 2439.55 | 2431.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 14:45:00 | 2443.30 | 2438.40 | 2432.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 15:15:00 | 2430.50 | 2436.82 | 2432.07 | SL hit (close<static) qty=1.00 sl=2430.89 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 2414.07 | 2429.29 | 2430.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 10:15:00 | 2403.79 | 2418.59 | 2424.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 11:15:00 | 2365.16 | 2356.23 | 2373.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 11:15:00 | 2365.16 | 2356.23 | 2373.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 2365.16 | 2356.23 | 2373.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 2339.03 | 2352.79 | 2370.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 2324.29 | 2314.84 | 2314.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 2324.29 | 2314.84 | 2314.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 15:15:00 | 2327.73 | 2317.42 | 2315.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 10:15:00 | 2442.62 | 2455.90 | 2413.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 11:00:00 | 2442.62 | 2455.90 | 2413.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 2484.84 | 2496.46 | 2483.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 2484.84 | 2496.46 | 2483.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 2483.82 | 2493.93 | 2483.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 2462.49 | 2493.93 | 2483.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2453.23 | 2485.79 | 2480.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2445.91 | 2485.79 | 2480.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 2414.94 | 2471.62 | 2474.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 2401.27 | 2441.79 | 2459.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 2431.95 | 2426.99 | 2446.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:45:00 | 2425.36 | 2426.99 | 2446.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 2435.10 | 2428.93 | 2442.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 2436.41 | 2428.93 | 2442.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 2444.70 | 2432.08 | 2443.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 2443.34 | 2432.08 | 2443.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 2444.80 | 2434.63 | 2443.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 2444.80 | 2434.63 | 2443.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 2447.76 | 2437.25 | 2443.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 2430.50 | 2437.25 | 2443.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2443.93 | 2438.59 | 2443.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 2440.44 | 2438.59 | 2443.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 2431.47 | 2437.17 | 2442.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:45:00 | 2420.46 | 2433.76 | 2440.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:30:00 | 2419.83 | 2428.95 | 2435.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 2299.44 | 2348.83 | 2383.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 2298.84 | 2348.83 | 2383.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 2178.41 | 2256.51 | 2317.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 2320.07 | 2305.95 | 2305.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 2369.71 | 2321.18 | 2312.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 2334.33 | 2346.21 | 2333.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 2334.33 | 2346.21 | 2333.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 2334.33 | 2346.21 | 2333.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 2327.83 | 2346.21 | 2333.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 2322.40 | 2341.45 | 2332.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 2322.40 | 2341.45 | 2332.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 2318.67 | 2336.89 | 2330.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 2317.07 | 2336.89 | 2330.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 2323.27 | 2332.11 | 2329.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:30:00 | 2321.58 | 2332.11 | 2329.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 2326.28 | 2330.04 | 2329.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 2326.76 | 2330.04 | 2329.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 2329.96 | 2330.03 | 2329.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:45:00 | 2343.78 | 2332.48 | 2330.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:30:00 | 2345.38 | 2335.55 | 2331.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:45:00 | 2344.46 | 2339.89 | 2338.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 13:15:00 | 2328.17 | 2336.08 | 2336.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 2328.17 | 2336.08 | 2336.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 2317.55 | 2332.37 | 2334.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 2309.31 | 2293.88 | 2309.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 2309.31 | 2293.88 | 2309.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 2309.31 | 2293.88 | 2309.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 2309.31 | 2293.88 | 2309.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 2304.47 | 2295.99 | 2309.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 2286.34 | 2295.99 | 2309.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2311.64 | 2299.12 | 2309.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 2311.64 | 2299.12 | 2309.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2304.81 | 2300.26 | 2308.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 2290.89 | 2303.10 | 2307.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 2289.39 | 2300.56 | 2305.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 2176.35 | 2215.06 | 2246.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 2174.92 | 2215.06 | 2246.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 2219.15 | 2214.53 | 2241.11 | SL hit (close>ema200) qty=0.50 sl=2214.53 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 2250.95 | 2241.29 | 2241.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 2274.17 | 2247.86 | 2244.03 | Break + close above crossover candle high |

### Cycle 46 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 2153.76 | 2233.84 | 2239.61 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 2223.90 | 2210.44 | 2209.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 15:15:00 | 2232.97 | 2219.92 | 2214.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 2240.48 | 2245.22 | 2234.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:00:00 | 2240.48 | 2245.22 | 2234.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 2233.01 | 2242.10 | 2234.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 2233.01 | 2242.10 | 2234.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 2237.52 | 2241.19 | 2234.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 2232.82 | 2241.19 | 2234.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 2238.25 | 2240.60 | 2235.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 2234.86 | 2240.60 | 2235.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 2238.54 | 2240.10 | 2235.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 2251.14 | 2240.10 | 2235.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:45:00 | 2244.79 | 2256.98 | 2250.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 2227.34 | 2251.05 | 2248.18 | SL hit (close<static) qty=1.00 sl=2235.10 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 2216.97 | 2244.24 | 2245.34 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 09:15:00 | 2305.53 | 2244.83 | 2243.36 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 2219.10 | 2247.75 | 2248.43 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 10:15:00 | 2259.38 | 2250.08 | 2249.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 11:15:00 | 2288.47 | 2257.75 | 2252.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 10:15:00 | 2257.59 | 2271.06 | 2263.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 10:15:00 | 2257.59 | 2271.06 | 2263.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 2257.59 | 2271.06 | 2263.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:45:00 | 2258.56 | 2271.06 | 2263.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 2238.10 | 2264.47 | 2261.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 2238.10 | 2264.47 | 2261.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 12:15:00 | 2231.12 | 2257.80 | 2258.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 13:15:00 | 2188.32 | 2243.90 | 2252.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 2116.00 | 2110.10 | 2147.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 2116.00 | 2110.10 | 2147.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 2156.81 | 2122.89 | 2147.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 2156.81 | 2122.89 | 2147.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 2169.03 | 2132.12 | 2149.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 2147.07 | 2132.12 | 2149.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:15:00 | 2145.52 | 2132.41 | 2140.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 2039.72 | 2079.75 | 2099.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 2038.24 | 2079.75 | 2099.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-25 09:15:00 | 2073.44 | 2062.66 | 2078.91 | SL hit (close>ema200) qty=0.50 sl=2062.66 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 2065.10 | 2041.20 | 2040.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 2078.82 | 2062.54 | 2052.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 2174.75 | 2179.86 | 2154.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 2174.75 | 2179.86 | 2154.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 2161.81 | 2181.68 | 2171.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 2158.07 | 2181.68 | 2171.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 2156.52 | 2176.65 | 2170.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 2172.18 | 2176.65 | 2170.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 2193.94 | 2181.53 | 2173.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:30:00 | 2184.64 | 2181.53 | 2173.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 2181.78 | 2183.51 | 2177.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:45:00 | 2181.34 | 2183.51 | 2177.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 2183.91 | 2183.59 | 2177.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:30:00 | 2191.04 | 2183.33 | 2178.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 2170.77 | 2180.82 | 2177.67 | SL hit (close<static) qty=1.00 sl=2177.41 alert=retest2 |

### Cycle 54 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 2128.84 | 2170.42 | 2173.23 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 2183.76 | 2173.08 | 2172.07 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 11:15:00 | 2163.70 | 2171.21 | 2171.31 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 2178.97 | 2171.31 | 2170.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 2212.99 | 2186.57 | 2179.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 2231.46 | 2242.73 | 2227.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 2231.46 | 2242.73 | 2227.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 2254.25 | 2245.04 | 2229.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 2280.13 | 2256.99 | 2241.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 2263.60 | 2278.26 | 2278.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 2263.60 | 2278.26 | 2278.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 2249.98 | 2270.84 | 2275.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 2282.60 | 2269.65 | 2273.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 2282.60 | 2269.65 | 2273.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 2282.60 | 2269.65 | 2273.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 2282.60 | 2269.65 | 2273.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 2273.59 | 2270.44 | 2273.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:45:00 | 2284.79 | 2270.44 | 2273.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 2268.59 | 2270.07 | 2273.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:30:00 | 2257.25 | 2266.97 | 2271.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:15:00 | 2255.12 | 2266.97 | 2271.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 2291.72 | 2254.56 | 2259.08 | SL hit (close>static) qty=1.00 sl=2282.17 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 2285.12 | 2266.48 | 2264.08 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 2242.27 | 2262.09 | 2263.17 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 15:15:00 | 2268.50 | 2262.03 | 2261.95 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 2255.12 | 2260.64 | 2261.33 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 2283.38 | 2265.19 | 2263.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 11:15:00 | 2294.77 | 2271.11 | 2266.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 2273.93 | 2316.90 | 2302.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 2273.93 | 2316.90 | 2302.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 2273.93 | 2316.90 | 2302.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 2273.93 | 2316.90 | 2302.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 2292.25 | 2311.97 | 2301.87 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 2255.65 | 2295.03 | 2295.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 2254.00 | 2286.83 | 2291.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2176.49 | 2166.94 | 2207.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 2176.49 | 2166.94 | 2207.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2176.49 | 2166.94 | 2207.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 2161.71 | 2169.25 | 2205.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 2170.82 | 2169.25 | 2205.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 2162.68 | 2190.59 | 2203.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:45:00 | 2173.10 | 2181.92 | 2194.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 2257.74 | 2194.20 | 2196.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2257.74 | 2194.20 | 2196.62 | SL hit (close>static) qty=1.00 sl=2224.97 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 2245.23 | 2204.41 | 2201.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 2313.19 | 2249.31 | 2227.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 2322.79 | 2323.54 | 2287.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 2322.79 | 2323.54 | 2287.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 2355.95 | 2345.71 | 2327.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:30:00 | 2350.03 | 2345.71 | 2327.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2361.57 | 2372.77 | 2363.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 2361.57 | 2372.77 | 2363.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 2363.12 | 2370.84 | 2363.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 2351.87 | 2370.84 | 2363.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 2371.07 | 2370.89 | 2363.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:15:00 | 2383.87 | 2372.44 | 2367.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 12:15:00 | 2356.14 | 2368.00 | 2366.74 | SL hit (close<static) qty=1.00 sl=2361.67 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 2305.73 | 2354.32 | 2360.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 2272.67 | 2337.99 | 2352.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 2301.56 | 2300.38 | 2323.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 2301.56 | 2300.38 | 2323.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 2315.42 | 2303.39 | 2322.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 2320.75 | 2303.39 | 2322.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 2319.98 | 2306.70 | 2322.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 2324.34 | 2306.70 | 2322.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 2304.95 | 2306.35 | 2321.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 2314.16 | 2306.35 | 2321.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2297.58 | 2300.94 | 2313.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 2273.73 | 2295.13 | 2308.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 2354.59 | 2274.27 | 2268.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 2354.59 | 2274.27 | 2268.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 2403.06 | 2300.03 | 2280.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 2342.86 | 2346.38 | 2314.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 2342.86 | 2346.38 | 2314.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 2308.83 | 2335.57 | 2315.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:00:00 | 2308.83 | 2335.57 | 2315.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 2297.29 | 2327.92 | 2313.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 2297.29 | 2327.92 | 2313.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 2262.10 | 2297.65 | 2302.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 2225.65 | 2268.29 | 2281.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 2314.35 | 2218.95 | 2234.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 2314.35 | 2218.95 | 2234.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2314.35 | 2218.95 | 2234.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 2314.35 | 2218.95 | 2234.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2341.69 | 2263.49 | 2253.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 2347.90 | 2290.88 | 2268.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 2462.78 | 2470.92 | 2448.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 2462.78 | 2470.92 | 2448.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 2456.29 | 2466.88 | 2454.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 2460.17 | 2466.88 | 2454.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 2447.08 | 2462.92 | 2453.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 2444.85 | 2462.92 | 2453.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 2436.41 | 2457.62 | 2452.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 2436.41 | 2457.62 | 2452.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 2429.43 | 2446.99 | 2448.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 2417.22 | 2439.53 | 2444.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 2433.70 | 2430.31 | 2437.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 2433.70 | 2430.31 | 2437.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2433.70 | 2430.31 | 2437.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 2442.23 | 2430.31 | 2437.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2424.10 | 2429.07 | 2435.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 2422.36 | 2429.07 | 2435.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 2446.98 | 2423.85 | 2427.61 | SL hit (close>static) qty=1.00 sl=2437.29 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 2461.52 | 2436.02 | 2432.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 2482.17 | 2453.79 | 2442.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 2465.98 | 2466.56 | 2455.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 2464.24 | 2466.56 | 2455.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2456.58 | 2464.56 | 2455.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 2456.58 | 2464.56 | 2455.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2471.22 | 2465.89 | 2457.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 2483.82 | 2465.89 | 2457.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:30:00 | 2473.35 | 2470.13 | 2462.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 2445.53 | 2460.02 | 2460.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 12:15:00 | 2445.53 | 2460.02 | 2460.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 2443.10 | 2456.64 | 2458.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 2472.38 | 2452.93 | 2454.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 2472.38 | 2452.93 | 2454.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 2472.38 | 2452.93 | 2454.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 2472.38 | 2452.93 | 2454.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 2448.43 | 2452.03 | 2454.06 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 2465.40 | 2456.94 | 2456.08 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 2444.36 | 2454.24 | 2454.99 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 2473.16 | 2456.65 | 2454.68 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 2430.50 | 2450.60 | 2453.03 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 2447.76 | 2425.88 | 2424.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 2463.95 | 2437.78 | 2431.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 2525.02 | 2528.51 | 2505.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:00:00 | 2525.02 | 2528.51 | 2505.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 2495.75 | 2518.25 | 2507.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 2500.59 | 2518.25 | 2507.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 2504.86 | 2515.57 | 2507.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 2514.46 | 2513.51 | 2507.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 2484.31 | 2505.21 | 2504.46 | SL hit (close<static) qty=1.00 sl=2494.10 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 2481.11 | 2500.39 | 2502.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 2465.01 | 2493.75 | 2499.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 2458.81 | 2438.67 | 2456.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 2458.81 | 2438.67 | 2456.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 2458.81 | 2438.67 | 2456.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 2458.81 | 2438.67 | 2456.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2460.75 | 2443.09 | 2456.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:00:00 | 2445.14 | 2455.84 | 2459.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 2442.81 | 2453.31 | 2457.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 2436.12 | 2448.94 | 2455.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 2402.29 | 2386.28 | 2384.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 2402.29 | 2386.28 | 2384.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 2446.40 | 2402.55 | 2393.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 2427.69 | 2429.06 | 2412.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 2427.69 | 2429.06 | 2412.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 2531.62 | 2539.92 | 2528.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 2531.62 | 2539.92 | 2528.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 2535.01 | 2538.94 | 2529.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 2529.97 | 2538.94 | 2529.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 2538.89 | 2538.93 | 2530.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:30:00 | 2533.46 | 2538.93 | 2530.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2543.15 | 2547.84 | 2541.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 2535.30 | 2547.84 | 2541.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2539.57 | 2546.19 | 2541.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 2539.57 | 2546.19 | 2541.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 2536.37 | 2544.22 | 2540.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 2535.59 | 2544.22 | 2540.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 2537.05 | 2542.79 | 2540.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 13:30:00 | 2546.35 | 2541.95 | 2540.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 2528.42 | 2537.84 | 2538.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 2528.42 | 2537.84 | 2538.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 2512.91 | 2530.79 | 2534.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 2508.64 | 2506.31 | 2515.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 2514.17 | 2506.31 | 2515.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2505.05 | 2506.06 | 2514.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 2500.30 | 2504.91 | 2513.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 2495.55 | 2503.91 | 2512.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 2521.82 | 2511.00 | 2512.27 | SL hit (close>static) qty=1.00 sl=2520.66 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 2511.84 | 2500.70 | 2500.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 2522.99 | 2505.16 | 2502.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 2535.11 | 2537.37 | 2529.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:15:00 | 2537.72 | 2537.37 | 2529.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 2526.96 | 2535.28 | 2528.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 2526.96 | 2535.28 | 2528.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2514.65 | 2531.16 | 2527.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 2510.00 | 2531.16 | 2527.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 2513.10 | 2524.35 | 2524.88 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 2537.53 | 2524.98 | 2524.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 2540.15 | 2531.50 | 2528.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 2530.74 | 2532.27 | 2529.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 2530.74 | 2532.27 | 2529.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 2530.74 | 2532.27 | 2529.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 2530.74 | 2532.27 | 2529.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 2524.83 | 2530.78 | 2528.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 2526.48 | 2530.78 | 2528.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 2521.53 | 2528.93 | 2528.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 2521.53 | 2528.93 | 2528.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 2519.40 | 2527.02 | 2527.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 2510.38 | 2523.70 | 2525.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 2532.59 | 2524.04 | 2525.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2532.59 | 2524.04 | 2525.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2532.59 | 2524.04 | 2525.61 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 2531.33 | 2527.03 | 2526.71 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 2525.51 | 2527.26 | 2527.49 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 2530.36 | 2527.88 | 2527.75 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 2502.05 | 2523.29 | 2525.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 2480.82 | 2514.80 | 2521.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2500.11 | 2490.71 | 2504.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2500.11 | 2490.71 | 2504.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2500.11 | 2490.71 | 2504.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 2496.42 | 2490.71 | 2504.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 2484.69 | 2489.50 | 2502.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 2476.84 | 2486.97 | 2499.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 2474.80 | 2482.75 | 2496.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 2353.00 | 2415.51 | 2439.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 2351.06 | 2415.51 | 2439.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-06 11:15:00 | 2229.16 | 2255.62 | 2284.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2216.73 | 2184.54 | 2182.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 2241.26 | 2203.36 | 2192.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 2212.07 | 2216.59 | 2205.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:45:00 | 2214.89 | 2216.59 | 2205.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2187.93 | 2210.86 | 2203.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 2187.93 | 2210.86 | 2203.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 2208.39 | 2210.36 | 2204.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 2213.14 | 2210.36 | 2204.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:30:00 | 2212.27 | 2210.27 | 2205.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 2217.21 | 2210.27 | 2205.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:00:00 | 2210.81 | 2211.49 | 2206.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 2218.18 | 2213.26 | 2208.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 2224.87 | 2214.85 | 2209.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 2252.79 | 2213.79 | 2210.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 2259.77 | 2291.08 | 2293.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 2259.77 | 2291.08 | 2293.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 2256.86 | 2279.18 | 2287.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 2226.13 | 2216.27 | 2233.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:45:00 | 2226.91 | 2216.27 | 2233.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 2254.83 | 2223.98 | 2235.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:45:00 | 2254.54 | 2223.98 | 2235.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2233.69 | 2225.93 | 2235.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 2222.45 | 2225.93 | 2235.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 2224.87 | 2213.10 | 2211.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 2224.87 | 2213.10 | 2211.91 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 2208.49 | 2211.53 | 2211.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 10:15:00 | 2200.92 | 2208.95 | 2210.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 11:15:00 | 2209.36 | 2209.03 | 2210.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:00:00 | 2209.36 | 2209.03 | 2210.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 2206.35 | 2208.50 | 2210.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:30:00 | 2208.49 | 2208.50 | 2210.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 2209.46 | 2208.69 | 2210.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 2209.46 | 2208.69 | 2210.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 2220.70 | 2211.09 | 2211.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 10:15:00 | 2222.64 | 2216.13 | 2213.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 2215.27 | 2216.77 | 2214.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 2215.27 | 2216.77 | 2214.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 2215.27 | 2216.77 | 2214.39 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 2209.26 | 2212.99 | 2213.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 2203.54 | 2211.10 | 2212.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 2215.76 | 2204.97 | 2208.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 2215.76 | 2204.97 | 2208.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 2215.76 | 2204.97 | 2208.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 2215.76 | 2204.97 | 2208.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 2209.07 | 2205.79 | 2208.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 2215.37 | 2205.79 | 2208.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 2215.37 | 2207.70 | 2208.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 2241.74 | 2207.70 | 2208.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 2252.50 | 2216.66 | 2212.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2261.81 | 2245.09 | 2237.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 2318.04 | 2323.30 | 2304.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:30:00 | 2320.85 | 2323.30 | 2304.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 2312.03 | 2318.03 | 2309.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 2309.02 | 2318.03 | 2309.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 2311.83 | 2316.79 | 2310.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:15:00 | 2309.80 | 2316.79 | 2310.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 2309.80 | 2315.39 | 2310.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 2322.21 | 2315.39 | 2310.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 13:15:00 | 2554.43 | 2501.78 | 2442.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 2494.78 | 2529.17 | 2533.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 2484.98 | 2514.05 | 2525.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 2463.66 | 2434.41 | 2446.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 13:15:00 | 2463.66 | 2434.41 | 2446.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 2463.66 | 2434.41 | 2446.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 2463.66 | 2434.41 | 2446.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 2510.29 | 2449.59 | 2452.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 2510.29 | 2449.59 | 2452.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 2514.84 | 2462.64 | 2458.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 2523.67 | 2501.87 | 2492.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 2495.36 | 2502.21 | 2494.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 2495.36 | 2502.21 | 2494.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 2495.36 | 2502.21 | 2494.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 2498.36 | 2502.21 | 2494.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 2501.08 | 2501.98 | 2495.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:15:00 | 2494.19 | 2501.98 | 2495.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 2493.61 | 2500.31 | 2494.95 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 2477.81 | 2489.15 | 2490.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 2458.32 | 2482.99 | 2487.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 2467.82 | 2455.06 | 2466.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 2467.82 | 2455.06 | 2466.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2467.82 | 2455.06 | 2466.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 2467.92 | 2455.06 | 2466.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2474.13 | 2458.88 | 2467.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 2474.13 | 2458.88 | 2467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2469.67 | 2461.03 | 2467.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 2463.46 | 2461.03 | 2467.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:00:00 | 2463.75 | 2461.58 | 2467.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 2461.33 | 2461.53 | 2466.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 2473.83 | 2469.30 | 2468.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 2473.83 | 2469.30 | 2468.83 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 2447.76 | 2465.57 | 2467.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2427.11 | 2445.98 | 2454.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 2440.58 | 2434.91 | 2445.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:30:00 | 2435.73 | 2434.91 | 2445.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2458.90 | 2440.87 | 2446.60 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 2471.12 | 2453.94 | 2451.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 2484.01 | 2463.19 | 2457.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 2473.25 | 2474.98 | 2467.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2473.25 | 2474.98 | 2467.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2473.25 | 2474.98 | 2467.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 2475.19 | 2474.98 | 2467.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 2492.74 | 2478.53 | 2469.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 2469.86 | 2478.53 | 2469.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 2476.74 | 2478.56 | 2472.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 2473.93 | 2478.56 | 2472.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 2471.12 | 2477.07 | 2472.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 2472.09 | 2477.07 | 2472.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 2467.34 | 2475.12 | 2471.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 2490.32 | 2475.12 | 2471.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 2480.23 | 2474.34 | 2471.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 2478.97 | 2473.29 | 2472.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 14:15:00 | 2477.71 | 2473.29 | 2472.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 2470.73 | 2472.78 | 2472.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 2470.73 | 2472.78 | 2472.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 2475.77 | 2480.11 | 2477.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 2475.77 | 2480.11 | 2477.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2464.24 | 2476.93 | 2475.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 2464.24 | 2476.93 | 2475.86 | SL hit (close<static) qty=1.00 sl=2467.34 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 2467.34 | 2475.01 | 2475.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 2456.00 | 2471.21 | 2473.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 11:15:00 | 2422.74 | 2420.61 | 2432.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:45:00 | 2424.00 | 2420.61 | 2432.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 2427.49 | 2421.98 | 2432.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 2427.49 | 2421.98 | 2432.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 2413.44 | 2420.27 | 2430.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 2406.36 | 2420.27 | 2430.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 2437.38 | 2424.10 | 2429.74 | SL hit (close>static) qty=1.00 sl=2433.41 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 2477.91 | 2434.86 | 2434.12 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 2426.43 | 2446.94 | 2448.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 2404.81 | 2434.15 | 2441.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 2416.25 | 2413.56 | 2426.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 2416.25 | 2413.56 | 2426.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 2300.30 | 2290.19 | 2314.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 2322.60 | 2290.19 | 2314.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 2295.55 | 2293.55 | 2310.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 2282.17 | 2293.38 | 2303.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2436.22 | 2320.07 | 2309.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2436.22 | 2320.07 | 2309.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 2462.78 | 2422.61 | 2393.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 2438.25 | 2439.51 | 2412.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 2438.25 | 2439.51 | 2412.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 2443.50 | 2448.49 | 2437.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 2438.50 | 2448.49 | 2437.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 2435.50 | 2445.56 | 2438.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 2435.50 | 2445.56 | 2438.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2439.70 | 2444.38 | 2438.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 2434.20 | 2444.38 | 2438.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2437.60 | 2443.03 | 2438.24 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 15:15:00 | 2429.60 | 2435.86 | 2435.88 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 2462.10 | 2441.11 | 2438.26 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 2431.70 | 2439.89 | 2440.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 2425.90 | 2434.40 | 2437.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 2297.90 | 2287.37 | 2316.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:00:00 | 2297.90 | 2287.37 | 2316.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 2320.80 | 2297.47 | 2316.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:30:00 | 2323.50 | 2297.47 | 2316.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 2308.70 | 2299.71 | 2315.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:30:00 | 2303.10 | 2298.49 | 2313.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 13:15:00 | 2187.94 | 2219.22 | 2244.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 2216.40 | 2211.06 | 2234.01 | SL hit (close>ema200) qty=0.50 sl=2211.06 alert=retest2 |

### Cycle 109 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 2256.60 | 2230.43 | 2229.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 2266.50 | 2237.65 | 2232.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 2242.60 | 2242.69 | 2236.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 2242.60 | 2242.69 | 2236.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 2236.10 | 2243.50 | 2237.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 2236.10 | 2243.50 | 2237.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 2227.00 | 2240.20 | 2236.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 2227.00 | 2240.20 | 2236.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 2208.20 | 2233.80 | 2234.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 2199.90 | 2221.61 | 2228.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2244.20 | 2225.74 | 2228.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 2244.20 | 2225.74 | 2228.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 2244.20 | 2225.74 | 2228.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 2244.20 | 2225.74 | 2228.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2248.80 | 2230.36 | 2230.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 2248.80 | 2230.36 | 2230.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 2249.00 | 2234.08 | 2232.30 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 2213.00 | 2233.45 | 2234.70 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 2244.10 | 2235.07 | 2234.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 2278.80 | 2243.82 | 2238.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 2279.00 | 2282.12 | 2272.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 2279.00 | 2282.12 | 2272.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2253.40 | 2275.43 | 2271.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 2255.00 | 2275.43 | 2271.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2249.00 | 2270.14 | 2269.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 2249.00 | 2270.14 | 2269.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 2246.20 | 2265.35 | 2266.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 2235.30 | 2259.34 | 2264.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 2241.80 | 2235.03 | 2242.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 2241.80 | 2235.03 | 2242.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 2241.80 | 2235.03 | 2242.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 2242.50 | 2235.03 | 2242.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 2242.60 | 2236.55 | 2242.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 2241.50 | 2236.55 | 2242.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2229.70 | 2235.18 | 2241.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 2222.30 | 2235.18 | 2241.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 2222.70 | 2233.96 | 2240.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 2224.70 | 2230.75 | 2237.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 2243.90 | 2231.84 | 2234.72 | SL hit (close>static) qty=1.00 sl=2243.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 2257.90 | 2237.05 | 2236.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 2268.50 | 2254.26 | 2246.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 2259.60 | 2260.48 | 2252.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:30:00 | 2261.10 | 2260.48 | 2252.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2251.60 | 2258.71 | 2252.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 2251.60 | 2258.71 | 2252.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 2255.50 | 2258.06 | 2252.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 2252.50 | 2258.06 | 2252.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 2251.40 | 2256.73 | 2252.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 2251.90 | 2256.73 | 2252.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 2246.00 | 2254.59 | 2252.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 2257.20 | 2254.59 | 2252.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 2248.10 | 2252.25 | 2251.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 2247.20 | 2252.25 | 2251.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 2240.70 | 2249.94 | 2250.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 2238.50 | 2247.65 | 2249.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 2239.20 | 2236.55 | 2242.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 2239.20 | 2236.55 | 2242.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2239.20 | 2236.55 | 2242.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 2238.10 | 2236.55 | 2242.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 2235.10 | 2236.26 | 2242.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 2246.60 | 2236.26 | 2242.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 2230.00 | 2233.17 | 2237.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 2233.70 | 2233.17 | 2237.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2217.30 | 2211.47 | 2218.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 2217.30 | 2211.47 | 2218.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2226.20 | 2215.11 | 2219.00 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 2235.60 | 2222.13 | 2221.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 2238.70 | 2225.45 | 2223.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 2280.00 | 2280.67 | 2268.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 2280.00 | 2280.67 | 2268.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2265.00 | 2277.07 | 2270.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2265.00 | 2277.07 | 2270.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 2271.10 | 2275.87 | 2270.84 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 2262.00 | 2267.21 | 2267.87 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 2276.10 | 2268.58 | 2268.36 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2253.10 | 2268.31 | 2268.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 2234.90 | 2261.63 | 2265.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 2176.00 | 2162.44 | 2189.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 2175.00 | 2162.44 | 2189.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2163.70 | 2166.49 | 2182.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 2148.10 | 2161.32 | 2175.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 2151.20 | 2158.81 | 2170.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:15:00 | 2151.70 | 2159.16 | 2167.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 2152.00 | 2157.20 | 2165.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2179.30 | 2160.79 | 2165.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 2179.30 | 2160.79 | 2165.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2185.80 | 2165.79 | 2167.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 2184.40 | 2165.79 | 2167.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 2189.20 | 2170.47 | 2169.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 2189.20 | 2170.47 | 2169.36 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 2156.30 | 2167.79 | 2168.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 2146.60 | 2162.15 | 2165.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 2092.10 | 2055.79 | 2079.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2070.20 | 2058.67 | 2078.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 2063.50 | 2073.61 | 2078.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:15:00 | 2064.00 | 2073.61 | 2078.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 1960.32 | 2046.48 | 2065.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 1960.80 | 2046.48 | 2065.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 14:15:00 | 1857.15 | 1985.50 | 2032.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 2018.50 | 1991.22 | 1990.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 2030.00 | 2006.04 | 1998.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:15:00 | 2027.00 | 2015.07 | 2008.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 1965.80 | 2000.32 | 2003.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1965.80 | 2000.32 | 2003.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1943.60 | 1988.98 | 1997.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 1982.80 | 1971.89 | 1983.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1978.70 | 1973.25 | 1983.36 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2181.90 | 2021.22 | 2003.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 2239.10 | 2189.37 | 2124.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 2214.50 | 2215.40 | 2172.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:00:00 | 2214.50 | 2215.40 | 2172.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 2215.70 | 2219.19 | 2200.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 2208.90 | 2219.19 | 2200.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 2217.50 | 2240.51 | 2229.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 2209.00 | 2240.51 | 2229.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 2245.50 | 2241.51 | 2230.80 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 2222.50 | 2227.25 | 2227.74 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 2235.40 | 2229.41 | 2228.68 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 2215.00 | 2226.62 | 2227.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 2208.30 | 2222.96 | 2225.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 2171.70 | 2154.15 | 2174.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 2171.70 | 2154.15 | 2174.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 2180.30 | 2159.38 | 2175.20 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 2199.80 | 2183.36 | 2181.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 2234.20 | 2193.53 | 2186.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 2210.40 | 2214.87 | 2200.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 2210.40 | 2214.87 | 2200.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2197.60 | 2211.41 | 2200.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 2198.60 | 2211.41 | 2200.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 2207.70 | 2210.67 | 2200.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:00:00 | 2211.80 | 2209.74 | 2202.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 2193.30 | 2206.98 | 2202.81 | SL hit (close<static) qty=1.00 sl=2193.80 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 2166.90 | 2198.96 | 2199.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 2156.20 | 2178.44 | 2188.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 2165.00 | 2164.43 | 2175.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 2165.00 | 2164.43 | 2175.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2192.20 | 2169.31 | 2174.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 2198.60 | 2169.31 | 2174.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 2188.90 | 2173.23 | 2176.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 2180.00 | 2174.76 | 2176.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 2180.60 | 2176.93 | 2177.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 2192.90 | 2178.68 | 2178.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 2192.90 | 2178.68 | 2178.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 2200.90 | 2185.55 | 2182.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 2208.90 | 2211.94 | 2201.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:45:00 | 2207.40 | 2211.94 | 2201.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 2203.40 | 2209.51 | 2201.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 2203.40 | 2209.51 | 2201.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 2216.60 | 2210.93 | 2203.15 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 2190.90 | 2200.52 | 2200.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 2160.20 | 2191.39 | 2196.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 2127.50 | 2126.24 | 2153.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 2127.50 | 2126.24 | 2153.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2080.50 | 2064.03 | 2088.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 2080.50 | 2064.03 | 2088.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 2080.00 | 2067.23 | 2087.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2084.20 | 2069.38 | 2086.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2062.90 | 2068.09 | 2084.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 2059.00 | 2068.27 | 2081.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 2061.50 | 2066.45 | 2079.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1956.05 | 2040.69 | 2063.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1958.42 | 2040.69 | 2063.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 2000.10 | 1997.12 | 2016.67 | SL hit (close>ema200) qty=0.50 sl=1997.12 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 2010.00 | 1978.91 | 1976.35 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1948.40 | 1984.01 | 1984.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 1941.40 | 1970.47 | 1977.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 1964.50 | 1955.14 | 1966.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1852.20 | 1856.22 | 1892.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 1848.30 | 1856.22 | 1892.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 1907.00 | 1857.89 | 1870.56 | SL hit (close>static) qty=1.00 sl=1893.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 1891.50 | 1877.76 | 1877.72 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1849.00 | 1874.30 | 1876.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 1832.80 | 1861.87 | 1870.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1850.40 | 1803.28 | 1822.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1844.60 | 1811.54 | 1824.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 1856.00 | 1811.54 | 1824.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1854.90 | 1835.31 | 1833.76 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1786.20 | 1827.32 | 1830.69 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1839.20 | 1827.57 | 1826.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1863.40 | 1834.74 | 1829.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1868.70 | 1870.30 | 1852.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 10:45:00 | 1866.90 | 1870.30 | 1852.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2124.80 | 2082.59 | 2043.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2144.90 | 2082.59 | 2043.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2132.10 | 2093.87 | 2052.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 2130.00 | 2117.77 | 2078.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 2130.20 | 2117.77 | 2078.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2249.80 | 2247.03 | 2235.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 2261.90 | 2247.03 | 2235.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 2261.70 | 2263.29 | 2259.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-28 09:15:00 | 2359.39 | 2328.13 | 2301.77 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 13:15:00 | 2755.86 | 2024-05-13 14:15:00 | 2792.07 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-05-17 10:00:00 | 2965.56 | 2024-05-23 12:15:00 | 3262.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-21 09:45:00 | 2958.29 | 2024-05-23 12:15:00 | 3254.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-29 15:00:00 | 3155.63 | 2024-05-31 11:15:00 | 3210.93 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-06-06 11:30:00 | 3086.55 | 2024-06-07 11:15:00 | 3099.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-06-06 14:30:00 | 3085.48 | 2024-06-07 11:15:00 | 3099.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-06-07 11:15:00 | 3089.94 | 2024-06-07 11:15:00 | 3099.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-06-11 12:15:00 | 3148.16 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-06-11 13:00:00 | 3147.19 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-06-11 14:15:00 | 3145.98 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-06-14 09:15:00 | 3152.28 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-06-14 11:00:00 | 3165.37 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-06-14 15:00:00 | 3169.68 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-06-19 11:45:00 | 3162.46 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-06-19 14:30:00 | 3160.96 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-06-20 10:15:00 | 3204.29 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-06-28 12:15:00 | 3067.50 | 2024-07-02 10:15:00 | 3093.38 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-06-28 14:30:00 | 3069.53 | 2024-07-02 10:15:00 | 3093.38 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-07-01 11:00:00 | 3072.39 | 2024-07-02 10:15:00 | 3093.38 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-07-01 11:30:00 | 3071.81 | 2024-07-02 10:15:00 | 3093.38 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-07-08 10:30:00 | 3019.41 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-07-08 15:15:00 | 3017.04 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-07-09 11:45:00 | 3013.16 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-07-10 09:30:00 | 3017.04 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-07-11 14:45:00 | 2985.34 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-12 09:30:00 | 2990.76 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-23 12:15:00 | 2836.86 | 2024-07-26 10:15:00 | 2950.68 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2024-07-23 14:15:00 | 2907.58 | 2024-07-26 10:15:00 | 2950.68 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-07-23 15:00:00 | 2899.68 | 2024-07-26 10:15:00 | 2950.68 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-07-25 10:45:00 | 2906.66 | 2024-07-26 10:15:00 | 2950.68 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest1 | 2024-07-30 11:45:00 | 3013.79 | 2024-08-02 09:15:00 | 3047.62 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-08-02 11:15:00 | 3112.58 | 2024-08-05 09:15:00 | 3014.03 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-08-02 13:15:00 | 3108.75 | 2024-08-05 09:15:00 | 3014.03 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2024-08-06 14:00:00 | 2995.71 | 2024-08-07 09:15:00 | 3040.74 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-08-09 09:15:00 | 3107.15 | 2024-08-12 09:15:00 | 2977.43 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2024-08-27 12:45:00 | 2981.85 | 2024-09-03 10:15:00 | 2937.88 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2024-08-27 13:15:00 | 2980.39 | 2024-09-03 10:15:00 | 2937.88 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2024-09-06 14:00:00 | 2898.76 | 2024-09-13 09:15:00 | 2887.27 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2024-09-10 12:45:00 | 2899.34 | 2024-09-13 09:15:00 | 2887.27 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2024-09-10 14:30:00 | 2900.21 | 2024-09-13 09:15:00 | 2887.27 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2024-09-12 15:15:00 | 2889.94 | 2024-09-13 09:15:00 | 2887.27 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-10-01 12:15:00 | 3048.64 | 2024-10-03 13:15:00 | 3008.31 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-10-11 12:15:00 | 3043.31 | 2024-10-14 12:15:00 | 3014.81 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-10-14 09:30:00 | 3047.96 | 2024-10-14 12:15:00 | 3014.81 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-16 11:30:00 | 2988.10 | 2024-10-21 11:15:00 | 2842.98 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2024-10-16 15:00:00 | 2992.61 | 2024-10-21 12:15:00 | 2838.69 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2024-10-16 11:30:00 | 2988.10 | 2024-10-23 10:15:00 | 2785.86 | STOP_HIT | 0.50 | 6.77% |
| SELL | retest2 | 2024-10-16 15:00:00 | 2992.61 | 2024-10-23 10:15:00 | 2785.86 | STOP_HIT | 0.50 | 6.91% |
| SELL | retest1 | 2024-11-25 15:00:00 | 2186.14 | 2024-11-26 13:15:00 | 2076.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-26 09:30:00 | 2180.95 | 2024-11-26 14:15:00 | 2071.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-25 15:00:00 | 2186.14 | 2024-11-27 09:15:00 | 2161.95 | STOP_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2024-11-26 09:30:00 | 2180.95 | 2024-11-27 09:15:00 | 2161.95 | STOP_HIT | 0.50 | 0.87% |
| BUY | retest2 | 2024-12-05 12:00:00 | 2437.87 | 2024-12-09 15:15:00 | 2418.67 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-12-09 11:15:00 | 2431.47 | 2024-12-09 15:15:00 | 2418.67 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-12-09 14:45:00 | 2423.71 | 2024-12-09 15:15:00 | 2418.67 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2024-12-16 14:45:00 | 2443.30 | 2024-12-16 15:15:00 | 2430.50 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-12-20 13:00:00 | 2339.03 | 2024-12-26 14:15:00 | 2324.29 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-01-08 11:45:00 | 2420.46 | 2025-01-10 14:15:00 | 2299.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:30:00 | 2419.83 | 2025-01-10 14:15:00 | 2298.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:45:00 | 2420.46 | 2025-01-13 13:15:00 | 2178.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 10:30:00 | 2419.83 | 2025-01-13 13:15:00 | 2177.85 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-20 10:45:00 | 2343.78 | 2025-01-21 13:15:00 | 2328.17 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-01-20 11:30:00 | 2345.38 | 2025-01-21 13:15:00 | 2328.17 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-01-21 11:45:00 | 2344.46 | 2025-01-21 13:15:00 | 2328.17 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-01-24 09:45:00 | 2290.89 | 2025-01-28 09:15:00 | 2176.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 2289.39 | 2025-01-28 09:15:00 | 2174.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 2290.89 | 2025-01-28 11:15:00 | 2219.15 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-01-24 12:30:00 | 2289.39 | 2025-01-28 11:15:00 | 2219.15 | STOP_HIT | 0.50 | 3.07% |
| BUY | retest2 | 2025-02-07 09:15:00 | 2251.14 | 2025-02-10 10:15:00 | 2227.34 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-02-10 09:45:00 | 2244.79 | 2025-02-10 10:15:00 | 2227.34 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-02-18 09:15:00 | 2147.07 | 2025-02-24 09:15:00 | 2039.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 15:15:00 | 2145.52 | 2025-02-24 09:15:00 | 2038.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 09:15:00 | 2147.07 | 2025-02-25 09:15:00 | 2073.44 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2025-02-18 15:15:00 | 2145.52 | 2025-02-25 09:15:00 | 2073.44 | STOP_HIT | 0.50 | 3.36% |
| BUY | retest2 | 2025-03-12 09:30:00 | 2191.04 | 2025-03-12 10:15:00 | 2170.77 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-03-21 09:15:00 | 2280.13 | 2025-03-25 12:15:00 | 2263.60 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-03-26 13:30:00 | 2257.25 | 2025-03-27 14:15:00 | 2291.72 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-03-26 14:15:00 | 2255.12 | 2025-03-27 14:15:00 | 2291.72 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-04-08 10:30:00 | 2161.71 | 2025-04-11 09:15:00 | 2257.74 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-04-08 11:15:00 | 2170.82 | 2025-04-11 09:15:00 | 2257.74 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-04-09 09:45:00 | 2162.68 | 2025-04-11 09:15:00 | 2257.74 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2025-04-09 13:45:00 | 2173.10 | 2025-04-11 09:15:00 | 2257.74 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-04-24 10:15:00 | 2383.87 | 2025-04-24 12:15:00 | 2356.14 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-04-29 11:45:00 | 2273.73 | 2025-05-05 10:15:00 | 2354.59 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-05-22 11:15:00 | 2422.36 | 2025-05-23 10:15:00 | 2446.98 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-05-27 11:15:00 | 2483.82 | 2025-05-28 12:15:00 | 2445.53 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-05-27 14:30:00 | 2473.35 | 2025-05-28 12:15:00 | 2445.53 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-12 09:15:00 | 2514.46 | 2025-06-12 10:15:00 | 2484.31 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-06-17 10:00:00 | 2445.14 | 2025-06-23 12:15:00 | 2402.29 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2025-06-17 10:45:00 | 2442.81 | 2025-06-23 12:15:00 | 2402.29 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-06-17 11:45:00 | 2436.12 | 2025-06-23 12:15:00 | 2402.29 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-07-03 13:30:00 | 2546.35 | 2025-07-03 15:15:00 | 2528.42 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-08 11:00:00 | 2500.30 | 2025-07-09 10:15:00 | 2521.82 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-08 11:30:00 | 2495.55 | 2025-07-09 10:15:00 | 2521.82 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-09 13:00:00 | 2498.56 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-09 13:30:00 | 2500.30 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-10 10:45:00 | 2501.17 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-07-10 12:45:00 | 2504.37 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-10 14:45:00 | 2501.85 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-14 11:15:00 | 2502.63 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-07-28 12:00:00 | 2476.84 | 2025-07-31 14:15:00 | 2353.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 12:30:00 | 2474.80 | 2025-07-31 14:15:00 | 2351.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 12:00:00 | 2476.84 | 2025-08-06 11:15:00 | 2229.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-28 12:30:00 | 2474.80 | 2025-08-06 14:15:00 | 2227.32 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-13 13:15:00 | 2213.14 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-08-13 14:30:00 | 2212.27 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2025-08-13 15:15:00 | 2217.21 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2025-08-14 10:00:00 | 2210.81 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2025-08-14 12:45:00 | 2224.87 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-08-18 09:15:00 | 2252.79 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-08-28 12:15:00 | 2222.45 | 2025-09-02 10:15:00 | 2224.87 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-16 09:15:00 | 2322.21 | 2025-09-22 13:15:00 | 2554.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-09 12:15:00 | 2463.46 | 2025-10-10 12:15:00 | 2473.83 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-10-09 13:00:00 | 2463.75 | 2025-10-10 12:15:00 | 2473.83 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-09 14:00:00 | 2461.33 | 2025-10-10 12:15:00 | 2473.83 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-20 09:15:00 | 2490.32 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-20 10:15:00 | 2480.23 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-21 13:45:00 | 2478.97 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-21 14:15:00 | 2477.71 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-10-28 14:15:00 | 2406.36 | 2025-10-29 09:15:00 | 2437.38 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-11 09:30:00 | 2282.17 | 2025-11-12 09:15:00 | 2436.22 | STOP_HIT | 1.00 | -6.75% |
| SELL | retest2 | 2025-11-28 13:30:00 | 2303.10 | 2025-12-03 13:15:00 | 2187.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 13:30:00 | 2303.10 | 2025-12-04 09:15:00 | 2216.40 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-12-18 14:15:00 | 2222.30 | 2025-12-19 15:15:00 | 2243.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-18 15:15:00 | 2222.70 | 2025-12-19 15:15:00 | 2243.90 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-19 11:15:00 | 2224.70 | 2025-12-19 15:15:00 | 2243.90 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-13 14:00:00 | 2148.10 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2151.20 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2151.70 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-14 15:15:00 | 2152.00 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-01-23 10:45:00 | 2063.50 | 2026-01-23 12:15:00 | 1960.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2064.00 | 2026-01-23 12:15:00 | 1960.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 2063.50 | 2026-01-23 14:15:00 | 1857.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2064.00 | 2026-01-23 14:15:00 | 1857.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-01 11:15:00 | 2027.00 | 2026-02-01 13:15:00 | 1965.80 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-02-18 15:00:00 | 2211.80 | 2026-02-19 09:15:00 | 2193.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-23 11:45:00 | 2180.00 | 2026-02-23 14:15:00 | 2192.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-02-23 13:15:00 | 2180.60 | 2026-02-23 14:15:00 | 2192.90 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-03-06 13:15:00 | 2059.00 | 2026-03-09 09:15:00 | 1956.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2061.50 | 2026-03-09 09:15:00 | 1958.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:15:00 | 2059.00 | 2026-03-10 13:15:00 | 2000.10 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2061.50 | 2026-03-10 13:15:00 | 2000.10 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-03-24 10:15:00 | 1848.30 | 2026-03-25 10:15:00 | 1907.00 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2144.90 | 2026-04-28 09:15:00 | 2359.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2132.10 | 2026-04-28 09:15:00 | 2345.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 14:45:00 | 2130.00 | 2026-04-28 09:15:00 | 2343.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 15:15:00 | 2130.20 | 2026-04-28 09:15:00 | 2343.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-23 10:15:00 | 2261.90 | 2026-05-04 09:15:00 | 2488.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-24 14:00:00 | 2261.70 | 2026-05-04 09:15:00 | 2487.87 | TARGET_HIT | 1.00 | 10.00% |
