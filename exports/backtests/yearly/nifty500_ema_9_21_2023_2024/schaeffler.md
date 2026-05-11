# Schaeffler India Ltd. (SCHAEFFLER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4226.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 222 |
| ALERT1 | 145 |
| ALERT2 | 146 |
| ALERT2_SKIP | 95 |
| ALERT3 | 323 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 124 |
| PARTIAL | 3 |
| TARGET_HIT | 9 |
| STOP_HIT | 117 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 129 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 104
- **Target hits / Stop hits / Partials:** 9 / 117 / 3
- **Avg / median % per leg:** -0.49% / -1.14%
- **Sum % (uncompounded):** -63.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 15 | 23.1% | 7 | 56 | 2 | 0.38% | 24.5% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| BUY @ 3rd Alert (retest2) | 61 | 11 | 18.0% | 5 | 56 | 0 | -0.09% | -5.5% |
| SELL (all) | 64 | 10 | 15.6% | 2 | 61 | 1 | -1.38% | -88.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 64 | 10 | 15.6% | 2 | 61 | 1 | -1.38% | -88.1% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 125 | 21 | 16.8% | 7 | 117 | 1 | -0.75% | -93.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 13:15:00 | 2792.45 | 2817.40 | 2820.05 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 14:15:00 | 2834.55 | 2808.22 | 2805.48 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 12:15:00 | 2801.50 | 2813.13 | 2814.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 09:15:00 | 2790.20 | 2807.37 | 2811.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 12:15:00 | 2806.70 | 2804.99 | 2808.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 13:15:00 | 2810.00 | 2805.99 | 2809.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 2810.00 | 2805.99 | 2809.00 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-05-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 10:15:00 | 2816.15 | 2810.39 | 2810.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 13:15:00 | 2817.95 | 2813.24 | 2811.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 12:15:00 | 2816.35 | 2816.75 | 2814.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 12:15:00 | 2816.35 | 2816.75 | 2814.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 12:15:00 | 2816.35 | 2816.75 | 2814.49 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 11:15:00 | 3102.00 | 3149.16 | 3154.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 09:15:00 | 3062.00 | 3100.45 | 3120.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 15:15:00 | 3080.00 | 3074.87 | 3096.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 3144.90 | 3064.17 | 3074.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 3144.90 | 3064.17 | 3074.16 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 10:15:00 | 3184.95 | 3088.33 | 3084.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 10:15:00 | 3221.10 | 3185.92 | 3160.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 10:15:00 | 3211.95 | 3219.96 | 3194.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 11:15:00 | 3203.15 | 3225.32 | 3212.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 3203.15 | 3225.32 | 3212.65 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 10:15:00 | 3187.00 | 3206.63 | 3208.02 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 12:15:00 | 3263.90 | 3215.69 | 3211.75 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 09:15:00 | 3179.00 | 3204.44 | 3207.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 11:15:00 | 3137.05 | 3186.75 | 3198.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 14:15:00 | 3092.15 | 3083.42 | 3110.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 14:15:00 | 3095.80 | 3083.08 | 3096.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 3095.80 | 3083.08 | 3096.71 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 13:15:00 | 3081.95 | 3065.86 | 3064.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 14:15:00 | 3088.30 | 3070.35 | 3066.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 12:15:00 | 3077.95 | 3084.07 | 3075.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 12:15:00 | 3077.95 | 3084.07 | 3075.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 12:15:00 | 3077.95 | 3084.07 | 3075.76 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 3043.90 | 3071.17 | 3073.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 12:15:00 | 3018.50 | 3060.64 | 3068.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 3012.00 | 3004.96 | 3025.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 3012.00 | 3004.96 | 3025.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 3012.00 | 3004.96 | 3025.69 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 14:15:00 | 3066.85 | 3040.41 | 3037.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 15:15:00 | 3074.05 | 3047.14 | 3040.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 12:15:00 | 3053.05 | 3053.87 | 3046.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 12:15:00 | 3053.05 | 3053.87 | 3046.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 3053.05 | 3053.87 | 3046.28 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 09:15:00 | 3110.05 | 3129.17 | 3130.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 10:15:00 | 3101.80 | 3123.70 | 3127.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 14:15:00 | 3082.00 | 3078.15 | 3094.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 15:15:00 | 3102.05 | 3082.93 | 3094.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 3102.05 | 3082.93 | 3094.84 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 3131.00 | 3102.30 | 3102.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 12:15:00 | 3166.90 | 3120.12 | 3110.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 14:15:00 | 3173.00 | 3182.39 | 3156.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 15:15:00 | 3170.00 | 3179.91 | 3157.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 15:15:00 | 3170.00 | 3179.91 | 3157.51 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 10:15:00 | 3095.00 | 3148.67 | 3153.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 12:15:00 | 3090.00 | 3128.15 | 3143.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 10:15:00 | 3119.50 | 3110.84 | 3126.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 3071.00 | 3089.04 | 3107.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 3071.00 | 3089.04 | 3107.98 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 13:15:00 | 3110.00 | 3107.19 | 3106.93 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 15:15:00 | 3101.00 | 3106.71 | 3106.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 09:15:00 | 3093.50 | 3104.06 | 3105.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 11:15:00 | 3103.35 | 3103.22 | 3104.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 12:15:00 | 3097.70 | 3102.12 | 3104.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 3097.70 | 3102.12 | 3104.25 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 14:15:00 | 3047.65 | 3033.49 | 3032.42 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 11:15:00 | 3024.00 | 3030.82 | 3031.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 13:15:00 | 3015.15 | 3026.52 | 3029.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 3016.50 | 2988.13 | 3003.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 3016.50 | 2988.13 | 3003.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 3016.50 | 2988.13 | 3003.64 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 12:15:00 | 2987.50 | 2977.56 | 2976.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 13:15:00 | 3041.00 | 2990.25 | 2982.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 13:15:00 | 3026.25 | 3040.27 | 3026.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 13:15:00 | 3026.25 | 3040.27 | 3026.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 13:15:00 | 3026.25 | 3040.27 | 3026.18 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 3448.00 | 3535.68 | 3544.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 3266.00 | 3453.19 | 3501.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 3362.70 | 3345.05 | 3409.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 3362.70 | 3345.05 | 3409.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 3362.70 | 3345.05 | 3409.62 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 13:15:00 | 3419.50 | 3400.45 | 3399.34 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 14:15:00 | 3289.70 | 3378.30 | 3389.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 09:15:00 | 3261.60 | 3342.43 | 3370.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 13:15:00 | 3176.00 | 3174.32 | 3215.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 09:15:00 | 3163.65 | 3098.43 | 3109.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 3163.65 | 3098.43 | 3109.75 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 11:15:00 | 3143.60 | 3116.97 | 3116.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 13:15:00 | 3179.20 | 3136.83 | 3126.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 3242.95 | 3285.31 | 3254.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 3242.95 | 3285.31 | 3254.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 3242.95 | 3285.31 | 3254.62 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 3213.10 | 3249.01 | 3252.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 13:15:00 | 3195.00 | 3238.21 | 3247.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 3261.40 | 3237.80 | 3244.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 3261.40 | 3237.80 | 3244.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 3261.40 | 3237.80 | 3244.19 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 11:15:00 | 2769.95 | 2752.21 | 2750.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 14:15:00 | 2771.65 | 2760.41 | 2755.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 09:15:00 | 2751.95 | 2760.57 | 2756.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 2751.95 | 2760.57 | 2756.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 2751.95 | 2760.57 | 2756.45 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 11:15:00 | 2732.05 | 2751.58 | 2752.87 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 15:15:00 | 2761.00 | 2754.00 | 2753.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 09:15:00 | 2809.00 | 2765.00 | 2758.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 13:15:00 | 2775.10 | 2775.53 | 2766.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 15:15:00 | 2773.00 | 2774.94 | 2767.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 2773.00 | 2774.94 | 2767.86 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 13:15:00 | 2732.75 | 2762.27 | 2764.40 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 2778.85 | 2766.52 | 2765.30 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 10:15:00 | 2730.50 | 2759.32 | 2762.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 11:15:00 | 2728.25 | 2753.11 | 2759.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 2755.45 | 2743.25 | 2750.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 2755.45 | 2743.25 | 2750.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 2755.45 | 2743.25 | 2750.65 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 09:15:00 | 2772.05 | 2753.60 | 2751.84 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 11:15:00 | 2749.05 | 2752.76 | 2752.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 12:15:00 | 2738.50 | 2749.91 | 2751.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 11:15:00 | 2745.00 | 2732.27 | 2740.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 11:15:00 | 2745.00 | 2732.27 | 2740.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 2745.00 | 2732.27 | 2740.01 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 15:15:00 | 2755.00 | 2746.05 | 2744.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 10:15:00 | 2761.05 | 2751.14 | 2747.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 09:15:00 | 2822.20 | 2831.74 | 2811.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 12:15:00 | 2825.60 | 2833.34 | 2817.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 2825.60 | 2833.34 | 2817.66 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 2773.00 | 2809.57 | 2810.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 09:15:00 | 2763.00 | 2778.57 | 2791.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 10:15:00 | 2812.85 | 2785.43 | 2793.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 10:15:00 | 2812.85 | 2785.43 | 2793.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 2812.85 | 2785.43 | 2793.39 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 14:15:00 | 2828.00 | 2803.44 | 2800.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 11:15:00 | 2830.95 | 2817.92 | 2809.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 09:15:00 | 3019.85 | 3023.59 | 2998.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 3046.20 | 3059.91 | 3041.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 3046.20 | 3059.91 | 3041.39 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 12:15:00 | 3043.15 | 3056.93 | 3057.18 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 14:15:00 | 3068.50 | 3058.82 | 3057.97 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 10:15:00 | 3050.00 | 3057.20 | 3057.57 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 3117.30 | 3065.97 | 3060.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-21 10:15:00 | 3182.95 | 3100.74 | 3081.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 15:15:00 | 3105.60 | 3113.96 | 3097.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 09:15:00 | 3200.70 | 3204.53 | 3186.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 3200.70 | 3204.53 | 3186.74 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 11:15:00 | 3162.20 | 3187.47 | 3188.57 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 13:15:00 | 3201.35 | 3190.86 | 3189.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 15:15:00 | 3215.00 | 3197.24 | 3193.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 09:15:00 | 3175.50 | 3192.89 | 3191.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 09:15:00 | 3175.50 | 3192.89 | 3191.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 3175.50 | 3192.89 | 3191.51 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 13:15:00 | 3194.60 | 3197.27 | 3197.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 3146.05 | 3186.01 | 3192.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 14:15:00 | 3113.00 | 3105.38 | 3130.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 15:15:00 | 3120.75 | 3108.45 | 3129.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 15:15:00 | 3120.75 | 3108.45 | 3129.20 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 11:15:00 | 3225.40 | 3157.71 | 3148.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 12:15:00 | 3239.75 | 3174.12 | 3157.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 10:15:00 | 3194.20 | 3204.33 | 3181.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 11:15:00 | 3186.35 | 3200.74 | 3181.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 3186.35 | 3200.74 | 3181.81 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 15:15:00 | 3301.50 | 3329.38 | 3333.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 3220.00 | 3307.51 | 3322.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 13:15:00 | 3296.05 | 3293.77 | 3309.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 3260.00 | 3284.09 | 3301.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 3260.00 | 3284.09 | 3301.33 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 15:15:00 | 3149.00 | 3121.79 | 3120.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 09:15:00 | 3150.25 | 3135.04 | 3129.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 10:15:00 | 3123.55 | 3132.74 | 3128.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 10:15:00 | 3123.55 | 3132.74 | 3128.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 3123.55 | 3132.74 | 3128.55 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 11:15:00 | 3096.70 | 3125.53 | 3125.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 12:15:00 | 3093.40 | 3119.11 | 3122.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 13:15:00 | 3070.00 | 3065.50 | 3086.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 14:15:00 | 3115.90 | 3075.58 | 3089.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 14:15:00 | 3115.90 | 3075.58 | 3089.15 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 10:15:00 | 3078.10 | 3004.82 | 2997.41 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 13:15:00 | 3021.25 | 3033.93 | 3034.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 14:15:00 | 3007.60 | 3028.66 | 3032.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 13:15:00 | 2872.00 | 2866.69 | 2902.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 2886.15 | 2870.58 | 2901.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 2886.15 | 2870.58 | 2901.03 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 13:15:00 | 2950.05 | 2909.67 | 2908.89 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 09:15:00 | 2909.95 | 2920.60 | 2921.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 2888.40 | 2909.54 | 2915.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 2896.00 | 2895.68 | 2904.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 2905.55 | 2896.86 | 2901.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 2905.55 | 2896.86 | 2901.90 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 2936.75 | 2904.70 | 2904.58 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 12:15:00 | 2884.20 | 2905.04 | 2907.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 13:15:00 | 2859.75 | 2895.98 | 2902.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 15:15:00 | 2869.00 | 2863.43 | 2877.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 2847.45 | 2860.23 | 2874.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 2847.45 | 2860.23 | 2874.38 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 12:15:00 | 2889.90 | 2865.21 | 2864.93 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 2851.00 | 2863.36 | 2864.52 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 13:15:00 | 2870.00 | 2865.65 | 2865.38 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 2846.60 | 2862.05 | 2863.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 2821.00 | 2853.84 | 2859.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 2826.75 | 2795.88 | 2813.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 2826.75 | 2795.88 | 2813.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 2826.75 | 2795.88 | 2813.54 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 2837.90 | 2821.61 | 2821.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 09:15:00 | 2928.00 | 2844.71 | 2831.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 2877.65 | 2886.04 | 2864.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 10:15:00 | 2888.00 | 2886.43 | 2866.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 2888.00 | 2886.43 | 2866.32 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 14:15:00 | 2878.50 | 2920.95 | 2925.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 15:15:00 | 2820.00 | 2883.78 | 2902.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 15:15:00 | 2855.00 | 2853.38 | 2875.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 2961.15 | 2874.93 | 2883.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 2961.15 | 2874.93 | 2883.73 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 2955.45 | 2891.04 | 2890.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 2973.35 | 2931.76 | 2912.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 13:15:00 | 3308.20 | 3309.53 | 3246.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 09:15:00 | 3247.20 | 3298.63 | 3257.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 3247.20 | 3298.63 | 3257.05 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 3200.00 | 3240.28 | 3244.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 10:15:00 | 3191.25 | 3218.05 | 3231.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 12:15:00 | 3219.95 | 3215.90 | 3228.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 13:15:00 | 3210.65 | 3214.85 | 3226.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 3210.65 | 3214.85 | 3226.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 3252.75 | 3211.73 | 3223.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 3296.60 | 3228.70 | 3229.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:00:00 | 3296.60 | 3228.70 | 3229.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 10:15:00 | 3313.05 | 3245.57 | 3237.31 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 15:15:00 | 3224.95 | 3246.58 | 3248.43 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 10:15:00 | 3260.00 | 3249.81 | 3249.61 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 11:15:00 | 3247.00 | 3249.24 | 3249.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 12:15:00 | 3227.75 | 3244.95 | 3247.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 10:15:00 | 3229.95 | 3225.49 | 3235.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 10:15:00 | 3229.95 | 3225.49 | 3235.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 3229.95 | 3225.49 | 3235.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:00:00 | 3229.95 | 3225.49 | 3235.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 3267.00 | 3233.80 | 3238.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 12:00:00 | 3267.00 | 3233.80 | 3238.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 3267.40 | 3240.52 | 3240.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 3259.35 | 3240.52 | 3240.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 13:15:00 | 3260.20 | 3244.45 | 3242.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 13:15:00 | 3260.20 | 3244.45 | 3242.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 15:15:00 | 3269.00 | 3251.91 | 3246.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 3227.10 | 3246.95 | 3244.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 3227.10 | 3246.95 | 3244.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 3227.10 | 3246.95 | 3244.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 12:30:00 | 3269.90 | 3254.00 | 3248.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 13:15:00 | 3234.35 | 3252.73 | 3253.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 13:15:00 | 3234.35 | 3252.73 | 3253.23 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 3289.65 | 3257.27 | 3254.82 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 09:15:00 | 3247.00 | 3276.93 | 3279.17 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 10:15:00 | 3352.35 | 3292.02 | 3285.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 13:15:00 | 3406.25 | 3331.98 | 3306.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 15:15:00 | 3325.55 | 3331.77 | 3311.28 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 09:45:00 | 3365.60 | 3334.95 | 3314.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 11:00:00 | 3374.90 | 3342.94 | 3320.07 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 09:15:00 | 3533.88 | 3459.70 | 3393.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 09:15:00 | 3543.65 | 3459.70 | 3393.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-04-30 10:15:00 | 3702.16 | 3520.20 | 3427.12 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 71 — SELL (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 15:15:00 | 3785.00 | 3823.45 | 3824.23 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-05-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 09:15:00 | 3850.00 | 3828.76 | 3826.58 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-05-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 11:15:00 | 3795.50 | 3822.44 | 3824.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 12:15:00 | 3793.95 | 3816.75 | 3821.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 09:15:00 | 3865.05 | 3806.20 | 3807.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 3865.05 | 3806.20 | 3807.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 3865.05 | 3806.20 | 3807.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 3867.80 | 3806.20 | 3807.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 3845.65 | 3814.09 | 3811.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 3961.80 | 3852.74 | 3832.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 4250.55 | 4305.06 | 4170.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 15:00:00 | 4250.55 | 4305.06 | 4170.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 4591.80 | 4607.17 | 4564.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:30:00 | 4560.25 | 4607.17 | 4564.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 4589.95 | 4599.31 | 4571.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:45:00 | 4579.95 | 4599.31 | 4571.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 4594.00 | 4596.76 | 4575.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 4643.95 | 4582.01 | 4575.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 12:15:00 | 4545.00 | 4573.21 | 4573.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 12:15:00 | 4545.00 | 4573.21 | 4573.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 13:15:00 | 4517.65 | 4562.10 | 4568.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 4496.50 | 4330.85 | 4387.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 4496.50 | 4330.85 | 4387.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 4496.50 | 4330.85 | 4387.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:30:00 | 4503.55 | 4330.85 | 4387.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 4453.40 | 4355.36 | 4393.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:30:00 | 4511.15 | 4355.36 | 4393.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 4524.95 | 4438.45 | 4426.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 09:15:00 | 4631.30 | 4498.36 | 4458.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 4424.75 | 4483.63 | 4455.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 4424.75 | 4483.63 | 4455.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 4424.75 | 4483.63 | 4455.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 4424.75 | 4483.63 | 4455.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 4142.85 | 4415.48 | 4427.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 12:15:00 | 3928.25 | 4318.03 | 4381.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 4224.45 | 4172.46 | 4263.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 4224.45 | 4172.46 | 4263.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 4137.75 | 4165.52 | 4252.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 15:15:00 | 4135.00 | 4161.71 | 4242.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 09:15:00 | 4320.00 | 4189.09 | 4240.73 | SL hit (close>static) qty=1.00 sl=4253.95 alert=retest2 |

### Cycle 78 — BUY (started 2024-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 11:15:00 | 4312.80 | 4231.39 | 4230.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 12:15:00 | 4414.00 | 4267.91 | 4247.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 12:15:00 | 4768.90 | 4782.80 | 4661.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 13:00:00 | 4768.90 | 4782.80 | 4661.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 4653.30 | 4763.21 | 4691.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 4640.00 | 4763.21 | 4691.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 4650.00 | 4740.57 | 4687.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 4648.70 | 4740.57 | 4687.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 10:15:00 | 4635.95 | 4667.71 | 4669.85 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 14:15:00 | 4725.70 | 4668.46 | 4665.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 09:15:00 | 4768.85 | 4694.39 | 4678.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 11:15:00 | 4670.35 | 4696.40 | 4682.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 11:15:00 | 4670.35 | 4696.40 | 4682.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 4670.35 | 4696.40 | 4682.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:00:00 | 4670.35 | 4696.40 | 4682.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 4639.75 | 4685.07 | 4678.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:00:00 | 4639.75 | 4685.07 | 4678.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 4650.80 | 4678.21 | 4675.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:30:00 | 4653.60 | 4678.21 | 4675.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 4652.00 | 4672.97 | 4673.75 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 4700.80 | 4673.97 | 4673.78 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 4644.15 | 4714.65 | 4720.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 12:15:00 | 4625.25 | 4674.43 | 4698.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 09:15:00 | 4652.15 | 4649.19 | 4676.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 4652.15 | 4649.19 | 4676.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 4652.15 | 4649.19 | 4676.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:30:00 | 4627.95 | 4644.74 | 4670.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 09:30:00 | 4627.20 | 4655.35 | 4666.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 11:15:00 | 4709.00 | 4667.50 | 4670.03 | SL hit (close>static) qty=1.00 sl=4695.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 13:15:00 | 4681.00 | 4672.56 | 4672.04 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 14:15:00 | 4654.70 | 4668.99 | 4670.46 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 4688.05 | 4673.84 | 4672.48 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 4622.20 | 4664.80 | 4669.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 10:15:00 | 4549.45 | 4641.73 | 4658.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 4250.00 | 4221.81 | 4310.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 10:00:00 | 4250.00 | 4221.81 | 4310.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 4055.00 | 3994.95 | 4049.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 14:00:00 | 3983.85 | 4033.96 | 4043.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 4055.05 | 3925.35 | 3921.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 4055.05 | 3925.35 | 3921.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 4189.60 | 4092.02 | 4052.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 4126.05 | 4147.66 | 4104.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 10:15:00 | 4123.00 | 4147.66 | 4104.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 4222.90 | 4235.21 | 4207.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 4218.65 | 4235.21 | 4207.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 4201.00 | 4228.37 | 4206.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 4201.00 | 4228.37 | 4206.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 4199.60 | 4222.61 | 4206.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 4188.45 | 4222.61 | 4206.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 4167.05 | 4206.23 | 4201.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:45:00 | 4150.05 | 4206.23 | 4201.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 4050.55 | 4168.98 | 4184.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 3909.45 | 4046.65 | 4106.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 3876.65 | 3849.78 | 3907.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 3886.60 | 3849.78 | 3907.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 3906.65 | 3861.15 | 3907.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 3906.65 | 3861.15 | 3907.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 3896.80 | 3868.28 | 3906.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 3909.00 | 3868.28 | 3906.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 3931.75 | 3880.97 | 3908.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 3931.75 | 3880.97 | 3908.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 3952.35 | 3895.25 | 3912.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 3952.35 | 3895.25 | 3912.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 3955.00 | 3907.20 | 3916.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 3905.30 | 3907.20 | 3916.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 14:15:00 | 3951.25 | 3910.00 | 3906.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 14:15:00 | 3951.25 | 3910.00 | 3906.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 4001.95 | 3948.26 | 3927.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 3930.05 | 3969.44 | 3946.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 3930.05 | 3969.44 | 3946.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 3930.05 | 3969.44 | 3946.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 3930.05 | 3969.44 | 3946.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 3944.80 | 3964.51 | 3946.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:45:00 | 3916.90 | 3964.51 | 3946.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 3946.35 | 3960.88 | 3946.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 13:15:00 | 3999.40 | 3958.70 | 3946.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 09:15:00 | 3916.70 | 3967.75 | 3957.22 | SL hit (close<static) qty=1.00 sl=3930.30 alert=retest2 |

### Cycle 91 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 3875.05 | 3949.21 | 3949.75 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 3961.60 | 3945.11 | 3944.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 3980.05 | 3954.60 | 3949.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 3967.60 | 3986.38 | 3969.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 3967.60 | 3986.38 | 3969.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 3967.60 | 3986.38 | 3969.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 3967.60 | 3986.38 | 3969.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 3954.65 | 3980.04 | 3967.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:45:00 | 3951.00 | 3980.04 | 3967.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 3954.00 | 3974.83 | 3966.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:00:00 | 3954.00 | 3974.83 | 3966.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 3967.75 | 3971.30 | 3966.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 14:45:00 | 3979.95 | 3972.91 | 3967.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 15:15:00 | 3955.50 | 3969.43 | 3966.36 | SL hit (close<static) qty=1.00 sl=3961.60 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 15:15:00 | 3950.00 | 3963.70 | 3965.26 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 3993.75 | 3969.71 | 3967.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 11:15:00 | 4034.20 | 3987.13 | 3976.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 4065.25 | 4066.95 | 4039.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:30:00 | 4069.35 | 4066.95 | 4039.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 4064.00 | 4063.69 | 4042.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 4039.55 | 4063.69 | 4042.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 4044.05 | 4080.85 | 4064.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 4044.05 | 4080.85 | 4064.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 4025.00 | 4069.68 | 4061.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:30:00 | 4030.10 | 4069.68 | 4061.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 3992.95 | 4054.33 | 4054.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 12:15:00 | 3957.00 | 4034.87 | 4046.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 12:15:00 | 3905.25 | 3895.46 | 3925.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 12:15:00 | 3905.25 | 3895.46 | 3925.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 3905.25 | 3895.46 | 3925.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:30:00 | 3949.10 | 3895.46 | 3925.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 3983.20 | 3908.35 | 3921.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 3983.20 | 3908.35 | 3921.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 4013.95 | 3929.47 | 3929.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 4026.20 | 3929.47 | 3929.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 3989.80 | 3941.54 | 3935.07 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 3894.00 | 3936.43 | 3938.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 12:15:00 | 3854.00 | 3912.18 | 3926.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 3864.30 | 3859.45 | 3881.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-04 10:00:00 | 3864.30 | 3859.45 | 3881.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 3907.85 | 3853.04 | 3864.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:45:00 | 3890.05 | 3853.04 | 3864.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 3942.00 | 3870.83 | 3871.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:45:00 | 3954.00 | 3870.83 | 3871.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 3956.25 | 3887.92 | 3879.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 15:15:00 | 3965.00 | 3925.51 | 3901.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 3893.90 | 3919.19 | 3901.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 3893.90 | 3919.19 | 3901.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 3893.90 | 3919.19 | 3901.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 3893.90 | 3919.19 | 3901.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 3885.00 | 3912.35 | 3899.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 3885.00 | 3912.35 | 3899.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 3891.80 | 3908.24 | 3899.01 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 3867.10 | 3890.94 | 3892.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 3815.15 | 3866.37 | 3879.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 13:15:00 | 3868.90 | 3852.64 | 3867.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 13:15:00 | 3868.90 | 3852.64 | 3867.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 3868.90 | 3852.64 | 3867.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 3861.95 | 3852.64 | 3867.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 3861.85 | 3854.48 | 3867.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:15:00 | 3868.00 | 3854.48 | 3867.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 3868.00 | 3857.19 | 3867.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 3908.50 | 3857.19 | 3867.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 3844.90 | 3854.73 | 3865.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:30:00 | 3830.80 | 3847.14 | 3859.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:00:00 | 3832.15 | 3847.14 | 3859.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:30:00 | 3820.20 | 3844.12 | 3852.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 09:45:00 | 3831.80 | 3822.00 | 3836.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 3848.90 | 3827.38 | 3837.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 3848.90 | 3827.38 | 3837.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 3848.50 | 3831.60 | 3838.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:45:00 | 3849.70 | 3831.60 | 3838.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 3850.35 | 3835.35 | 3839.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:45:00 | 3854.55 | 3835.35 | 3839.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-12 15:15:00 | 3859.95 | 3844.69 | 3843.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 3859.95 | 3844.69 | 3843.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 3884.60 | 3852.67 | 3846.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 3881.50 | 3912.79 | 3888.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 10:15:00 | 3881.50 | 3912.79 | 3888.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 3881.50 | 3912.79 | 3888.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 3881.50 | 3912.79 | 3888.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 3926.15 | 3915.46 | 3892.09 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 3860.35 | 3887.72 | 3890.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 3827.00 | 3863.59 | 3877.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 12:15:00 | 3840.85 | 3838.30 | 3860.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 12:30:00 | 3841.10 | 3838.30 | 3860.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 3853.75 | 3844.06 | 3859.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 14:45:00 | 3862.25 | 3844.06 | 3859.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 3850.60 | 3845.37 | 3858.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 3896.35 | 3845.37 | 3858.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 3819.65 | 3840.23 | 3855.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 3850.00 | 3840.23 | 3855.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 3812.25 | 3819.04 | 3835.06 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 3869.40 | 3837.95 | 3837.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 11:15:00 | 3877.30 | 3845.82 | 3841.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 10:15:00 | 3909.90 | 3937.15 | 3912.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 10:15:00 | 3909.90 | 3937.15 | 3912.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 3909.90 | 3937.15 | 3912.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 3909.90 | 3937.15 | 3912.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 3911.75 | 3932.07 | 3912.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 3910.05 | 3932.07 | 3912.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 3900.15 | 3925.69 | 3911.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:30:00 | 3885.95 | 3925.69 | 3911.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 3905.45 | 3921.64 | 3910.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 3922.00 | 3921.71 | 3911.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 3868.05 | 3912.05 | 3909.19 | SL hit (close<static) qty=1.00 sl=3892.05 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 3866.55 | 3902.95 | 3905.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 15:15:00 | 3854.45 | 3874.94 | 3888.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 11:15:00 | 3865.00 | 3863.38 | 3872.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 11:15:00 | 3865.00 | 3863.38 | 3872.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 3865.00 | 3863.38 | 3872.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:00:00 | 3865.00 | 3863.38 | 3872.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 3919.05 | 3872.54 | 3874.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 3919.05 | 3872.54 | 3874.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 15:15:00 | 3897.30 | 3877.49 | 3876.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 10:15:00 | 3937.35 | 3888.06 | 3881.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 3903.85 | 3912.12 | 3899.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 3903.85 | 3912.12 | 3899.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 3903.85 | 3912.12 | 3899.05 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 15:15:00 | 3885.95 | 3892.32 | 3892.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 3870.70 | 3888.00 | 3890.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 14:15:00 | 3874.65 | 3864.21 | 3876.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 14:15:00 | 3874.65 | 3864.21 | 3876.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 3874.65 | 3864.21 | 3876.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:45:00 | 3905.00 | 3864.21 | 3876.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 3847.30 | 3860.82 | 3873.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 3874.35 | 3860.82 | 3873.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 3793.35 | 3847.33 | 3866.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:30:00 | 3761.00 | 3831.02 | 3857.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:00:00 | 3748.35 | 3776.00 | 3813.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 11:15:00 | 3850.20 | 3814.96 | 3810.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 3850.20 | 3814.96 | 3810.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 3886.00 | 3840.81 | 3825.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 3842.15 | 3850.58 | 3834.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 12:45:00 | 3843.25 | 3850.58 | 3834.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 3885.00 | 3857.47 | 3839.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 3896.90 | 3858.49 | 3842.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:30:00 | 3901.40 | 3878.87 | 3860.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:30:00 | 3893.55 | 3880.83 | 3863.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 3972.35 | 3998.45 | 3998.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 3972.35 | 3998.45 | 3998.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 3899.35 | 3954.63 | 3973.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 14:15:00 | 3876.70 | 3872.60 | 3900.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 15:00:00 | 3876.70 | 3872.60 | 3900.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 3502.75 | 3475.08 | 3527.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 10:15:00 | 3439.80 | 3456.08 | 3490.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 10:30:00 | 3442.55 | 3449.57 | 3459.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 3445.00 | 3431.50 | 3445.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 10:15:00 | 3440.15 | 3439.24 | 3446.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 3444.05 | 3440.20 | 3446.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 3505.65 | 3453.29 | 3452.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 3505.65 | 3453.29 | 3452.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 3525.40 | 3475.75 | 3462.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 3502.75 | 3506.82 | 3489.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 14:00:00 | 3502.75 | 3506.82 | 3489.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 3502.70 | 3504.83 | 3492.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:15:00 | 3473.50 | 3504.83 | 3492.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 3448.40 | 3493.55 | 3488.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 3448.40 | 3493.55 | 3488.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 3472.75 | 3489.39 | 3487.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:30:00 | 3448.40 | 3489.39 | 3487.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 3468.60 | 3484.13 | 3485.25 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 09:15:00 | 3513.85 | 3490.41 | 3487.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 10:15:00 | 3567.50 | 3505.83 | 3495.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 14:15:00 | 3496.25 | 3519.92 | 3507.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 14:15:00 | 3496.25 | 3519.92 | 3507.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 3496.25 | 3519.92 | 3507.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 15:00:00 | 3496.25 | 3519.92 | 3507.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 3491.00 | 3514.13 | 3505.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 3505.60 | 3514.13 | 3505.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 3520.00 | 3515.31 | 3507.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 13:45:00 | 3548.00 | 3515.40 | 3509.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 14:15:00 | 3540.25 | 3515.40 | 3509.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 3445.40 | 3506.54 | 3507.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 3445.40 | 3506.54 | 3507.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 10:15:00 | 3437.60 | 3492.75 | 3501.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 14:15:00 | 3436.15 | 3416.75 | 3439.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 15:00:00 | 3436.15 | 3416.75 | 3439.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 3390.00 | 3411.40 | 3435.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:00:00 | 3381.50 | 3404.56 | 3427.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:30:00 | 3387.65 | 3397.80 | 3420.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 3379.80 | 3394.20 | 3416.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 3460.80 | 3412.18 | 3419.93 | SL hit (close>static) qty=1.00 sl=3451.65 alert=retest2 |

### Cycle 112 — BUY (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 10:15:00 | 3468.65 | 3413.52 | 3406.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 14:15:00 | 3503.25 | 3455.60 | 3436.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 3546.55 | 3570.78 | 3542.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 3546.55 | 3570.78 | 3542.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 3546.55 | 3570.78 | 3542.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 3546.55 | 3570.78 | 3542.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 3574.90 | 3571.60 | 3545.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 11:15:00 | 3587.05 | 3571.60 | 3545.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 3582.25 | 3601.27 | 3602.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 3582.25 | 3601.27 | 3602.77 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 3619.20 | 3604.86 | 3604.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 11:15:00 | 3628.15 | 3609.52 | 3606.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 13:15:00 | 3600.40 | 3610.12 | 3607.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 13:15:00 | 3600.40 | 3610.12 | 3607.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 3600.40 | 3610.12 | 3607.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:00:00 | 3600.40 | 3610.12 | 3607.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 3595.80 | 3607.26 | 3606.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:15:00 | 3598.00 | 3607.26 | 3606.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 15:15:00 | 3598.00 | 3605.41 | 3605.58 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 3633.95 | 3611.12 | 3608.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 3640.40 | 3616.97 | 3611.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 12:15:00 | 3665.70 | 3667.12 | 3645.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 12:30:00 | 3660.50 | 3667.12 | 3645.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 3603.80 | 3655.43 | 3643.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 3603.80 | 3655.43 | 3643.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 3661.30 | 3656.60 | 3645.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 15:00:00 | 3677.00 | 3657.54 | 3650.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 3577.80 | 3640.39 | 3643.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 3577.80 | 3640.39 | 3643.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 3560.00 | 3619.90 | 3629.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 3387.60 | 3352.54 | 3383.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 3387.60 | 3352.54 | 3383.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 3387.60 | 3352.54 | 3383.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:15:00 | 3406.35 | 3352.54 | 3383.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 3431.80 | 3368.40 | 3387.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 3431.80 | 3368.40 | 3387.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 3421.55 | 3379.03 | 3390.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 3396.65 | 3379.03 | 3390.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 13:15:00 | 3362.40 | 3339.12 | 3337.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 13:15:00 | 3362.40 | 3339.12 | 3337.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 3400.00 | 3351.30 | 3343.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 3430.20 | 3433.42 | 3404.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 09:30:00 | 3419.30 | 3433.42 | 3404.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 3380.05 | 3427.27 | 3413.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 3380.05 | 3427.27 | 3413.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 3380.10 | 3417.83 | 3410.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 3396.80 | 3417.83 | 3410.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 3371.10 | 3408.49 | 3406.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:30:00 | 3373.90 | 3408.49 | 3406.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 3360.10 | 3398.81 | 3402.53 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 3417.65 | 3401.63 | 3400.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 3436.25 | 3413.81 | 3406.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 3473.15 | 3474.29 | 3454.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:15:00 | 3471.90 | 3474.29 | 3454.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 3453.55 | 3466.75 | 3457.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 3446.40 | 3466.75 | 3457.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 3450.00 | 3463.40 | 3456.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:45:00 | 3454.25 | 3463.40 | 3456.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 3455.50 | 3461.82 | 3456.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 14:45:00 | 3482.10 | 3459.01 | 3456.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 15:15:00 | 3468.00 | 3459.01 | 3456.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 10:45:00 | 3505.90 | 3490.70 | 3478.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 14:15:00 | 3439.05 | 3487.12 | 3489.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 14:15:00 | 3439.05 | 3487.12 | 3489.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 3350.00 | 3453.76 | 3473.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 13:15:00 | 3231.00 | 3230.49 | 3286.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 13:30:00 | 3233.85 | 3230.49 | 3286.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 3262.75 | 3253.01 | 3274.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:30:00 | 3273.05 | 3253.01 | 3274.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 3266.15 | 3249.66 | 3265.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:30:00 | 3294.40 | 3249.66 | 3265.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 3253.40 | 3250.41 | 3264.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:30:00 | 3270.55 | 3250.41 | 3264.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 3273.45 | 3247.68 | 3258.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:45:00 | 3266.00 | 3247.68 | 3258.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 3282.75 | 3254.69 | 3261.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 3282.75 | 3254.69 | 3261.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 3293.80 | 3266.72 | 3265.70 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 3254.90 | 3264.36 | 3264.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 3221.30 | 3250.61 | 3257.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 3235.05 | 3231.64 | 3243.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 13:00:00 | 3235.05 | 3231.64 | 3243.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 3245.90 | 3234.49 | 3243.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 3245.90 | 3234.49 | 3243.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 3222.70 | 3232.13 | 3241.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 15:15:00 | 3251.60 | 3232.13 | 3241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 3251.60 | 3236.03 | 3242.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 3270.20 | 3236.03 | 3242.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 3216.90 | 3232.20 | 3240.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 3207.00 | 3232.20 | 3240.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 12:15:00 | 3274.20 | 3242.58 | 3242.98 | SL hit (close>static) qty=1.00 sl=3272.90 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 13:15:00 | 3267.55 | 3247.57 | 3245.21 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 3221.15 | 3242.48 | 3244.03 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 15:15:00 | 3275.00 | 3249.34 | 3246.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 3354.05 | 3270.28 | 3256.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 3247.90 | 3310.14 | 3291.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 3247.90 | 3310.14 | 3291.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 3247.90 | 3310.14 | 3291.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 3247.90 | 3310.14 | 3291.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 3254.85 | 3299.08 | 3287.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 3254.85 | 3299.08 | 3287.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 3241.70 | 3287.61 | 3283.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:30:00 | 3246.50 | 3287.61 | 3283.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 3249.75 | 3274.97 | 3278.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 3215.75 | 3263.13 | 3272.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 3122.05 | 3117.60 | 3163.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 3122.05 | 3117.60 | 3163.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 3220.30 | 3135.10 | 3156.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 3212.45 | 3135.10 | 3156.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 3232.60 | 3154.60 | 3163.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 3232.60 | 3154.60 | 3163.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 3192.45 | 3173.06 | 3170.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 3226.30 | 3188.90 | 3178.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-29 15:15:00 | 3165.95 | 3184.31 | 3177.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 15:15:00 | 3165.95 | 3184.31 | 3177.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 3165.95 | 3184.31 | 3177.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 09:30:00 | 3256.90 | 3200.03 | 3185.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 12:15:00 | 3284.80 | 3354.89 | 3356.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 3284.80 | 3354.89 | 3356.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 3269.35 | 3337.78 | 3348.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 15:15:00 | 3344.00 | 3332.09 | 3343.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 15:15:00 | 3344.00 | 3332.09 | 3343.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 3344.00 | 3332.09 | 3343.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 3347.20 | 3332.09 | 3343.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 3296.40 | 3324.95 | 3339.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:45:00 | 3276.45 | 3325.27 | 3337.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 14:15:00 | 3382.35 | 3340.07 | 3335.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 14:15:00 | 3382.35 | 3340.07 | 3335.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 15:15:00 | 3404.90 | 3353.04 | 3342.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 3322.20 | 3346.87 | 3340.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 3322.20 | 3346.87 | 3340.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 3322.20 | 3346.87 | 3340.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 3345.00 | 3346.87 | 3340.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 3350.00 | 3347.50 | 3341.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 15:00:00 | 3377.05 | 3354.38 | 3346.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:30:00 | 3360.00 | 3356.33 | 3348.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:00:00 | 3355.60 | 3356.33 | 3348.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 3286.50 | 3342.36 | 3343.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 10:15:00 | 3286.50 | 3342.36 | 3343.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 3267.20 | 3318.41 | 3331.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 3062.60 | 3055.72 | 3115.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:45:00 | 3059.90 | 3055.72 | 3115.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 2975.65 | 2946.31 | 2972.37 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 3050.00 | 2984.32 | 2983.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 3087.90 | 3005.04 | 2992.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 12:15:00 | 3042.20 | 3045.96 | 3021.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 12:15:00 | 3042.20 | 3045.96 | 3021.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 3042.20 | 3045.96 | 3021.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:00:00 | 3042.20 | 3045.96 | 3021.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 3055.30 | 3047.38 | 3026.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 3088.85 | 3049.90 | 3029.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 10:00:00 | 3086.70 | 3113.06 | 3112.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 3064.40 | 3103.33 | 3108.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 3064.40 | 3103.33 | 3108.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 3051.00 | 3076.29 | 3091.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 09:15:00 | 3086.05 | 3078.24 | 3090.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 3086.05 | 3078.24 | 3090.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 3086.05 | 3078.24 | 3090.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:30:00 | 3055.95 | 3072.50 | 3085.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:15:00 | 3041.00 | 3073.05 | 3083.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:30:00 | 3046.30 | 3046.86 | 3069.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 3324.95 | 3097.22 | 3075.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 3324.95 | 3097.22 | 3075.67 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 3229.40 | 3246.59 | 3247.34 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 3261.45 | 3244.13 | 3244.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 11:15:00 | 3312.90 | 3257.88 | 3250.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 13:15:00 | 3399.95 | 3401.24 | 3373.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 13:45:00 | 3390.00 | 3401.24 | 3373.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 3376.85 | 3396.36 | 3373.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 15:00:00 | 3376.85 | 3396.36 | 3373.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 3365.95 | 3390.28 | 3373.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 3406.10 | 3390.28 | 3373.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 3492.70 | 3524.39 | 3527.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 3492.70 | 3524.39 | 3527.98 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 10:15:00 | 3583.95 | 3532.64 | 3530.30 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 3499.95 | 3524.19 | 3527.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 3486.70 | 3512.18 | 3520.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 3298.90 | 3287.17 | 3337.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:30:00 | 3297.60 | 3287.17 | 3337.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 3317.00 | 3299.15 | 3319.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:00:00 | 3317.00 | 3299.15 | 3319.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 3307.25 | 3300.77 | 3318.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:45:00 | 3316.95 | 3300.77 | 3318.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 3194.50 | 3273.37 | 3299.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 3189.45 | 3273.37 | 3299.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 11:45:00 | 3188.45 | 3242.48 | 3280.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 2870.51 | 3140.24 | 3212.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 3128.60 | 3046.70 | 3041.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 3140.00 | 3094.48 | 3068.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 10:15:00 | 3264.30 | 3282.75 | 3243.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 11:00:00 | 3264.30 | 3282.75 | 3243.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 3259.10 | 3297.64 | 3269.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 3259.10 | 3297.64 | 3269.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 3247.00 | 3287.51 | 3267.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 3221.60 | 3287.51 | 3267.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 3262.50 | 3282.51 | 3267.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 15:00:00 | 3286.20 | 3276.53 | 3267.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 11:45:00 | 3278.00 | 3275.59 | 3270.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 3227.10 | 3281.51 | 3277.23 | SL hit (close<static) qty=1.00 sl=3244.80 alert=retest2 |

### Cycle 141 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 3186.10 | 3262.43 | 3268.94 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 3290.00 | 3252.89 | 3248.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 3308.00 | 3268.57 | 3256.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 3275.60 | 3282.40 | 3268.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 14:00:00 | 3275.60 | 3282.40 | 3268.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 3285.30 | 3282.98 | 3269.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:30:00 | 3252.20 | 3282.98 | 3269.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 3281.00 | 3282.58 | 3270.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 3508.30 | 3282.58 | 3270.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-13 10:15:00 | 3859.13 | 3768.67 | 3722.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 12:15:00 | 4022.00 | 4059.66 | 4062.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 4011.50 | 4050.03 | 4057.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 4055.00 | 4047.42 | 4054.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 4055.00 | 4047.42 | 4054.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 4055.00 | 4047.42 | 4054.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:00:00 | 4004.30 | 4031.60 | 4044.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 14:45:00 | 3999.80 | 4023.52 | 4038.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 3998.70 | 4021.62 | 4036.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 12:45:00 | 4006.70 | 4017.91 | 4030.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 4045.30 | 4010.00 | 4019.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 4045.30 | 4010.00 | 4019.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 4040.80 | 4016.16 | 4021.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:45:00 | 4043.90 | 4016.16 | 4021.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 4023.90 | 4019.67 | 4022.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:30:00 | 4030.20 | 4019.67 | 4022.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 4000.90 | 4015.91 | 4020.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 4113.00 | 4037.58 | 4029.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 4113.00 | 4037.58 | 4029.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 4180.00 | 4111.16 | 4076.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 10:15:00 | 4157.50 | 4171.87 | 4135.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 4157.50 | 4171.87 | 4135.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 4157.50 | 4171.87 | 4135.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 4164.90 | 4171.87 | 4135.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 4115.90 | 4160.68 | 4133.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 4115.90 | 4160.68 | 4133.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 4133.00 | 4155.14 | 4133.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:45:00 | 4144.10 | 4152.91 | 4134.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 14:45:00 | 4143.40 | 4146.39 | 4133.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:30:00 | 4139.20 | 4139.62 | 4133.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 12:00:00 | 4145.90 | 4140.88 | 4134.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 4091.40 | 4130.98 | 4130.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 4091.40 | 4130.98 | 4130.32 | SL hit (close<static) qty=1.00 sl=4112.70 alert=retest2 |

### Cycle 145 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 4118.90 | 4128.57 | 4129.29 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 4157.00 | 4131.82 | 4129.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 11:15:00 | 4170.40 | 4139.54 | 4133.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 13:15:00 | 4281.00 | 4285.73 | 4249.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 14:00:00 | 4281.00 | 4285.73 | 4249.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 4248.20 | 4270.95 | 4251.49 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 4220.00 | 4240.84 | 4243.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 4203.80 | 4225.55 | 4234.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 4234.80 | 4217.73 | 4226.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 4234.80 | 4217.73 | 4226.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 4234.80 | 4217.73 | 4226.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:30:00 | 4216.50 | 4218.32 | 4224.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 14:15:00 | 4005.67 | 4065.51 | 4112.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 4068.50 | 4057.38 | 4100.45 | SL hit (close>ema200) qty=0.50 sl=4057.38 alert=retest2 |

### Cycle 148 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 3933.60 | 3920.79 | 3919.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 3960.30 | 3941.55 | 3931.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 3930.20 | 3941.65 | 3933.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 14:15:00 | 3930.20 | 3941.65 | 3933.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 3930.20 | 3941.65 | 3933.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 3930.20 | 3941.65 | 3933.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 3921.00 | 3937.52 | 3932.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 3930.40 | 3937.52 | 3932.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 3890.70 | 3928.16 | 3928.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 3884.40 | 3919.41 | 3924.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 15:15:00 | 3907.40 | 3899.24 | 3910.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-27 09:15:00 | 3914.00 | 3899.24 | 3910.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 3909.70 | 3901.33 | 3910.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:15:00 | 3921.70 | 3901.33 | 3910.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 3915.00 | 3904.06 | 3910.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:30:00 | 3933.80 | 3904.06 | 3910.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 3910.50 | 3905.35 | 3910.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:30:00 | 3913.70 | 3905.35 | 3910.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 3911.10 | 3906.50 | 3910.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:45:00 | 3920.00 | 3906.50 | 3910.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 13:15:00 | 3945.00 | 3914.20 | 3913.66 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 3890.10 | 3909.38 | 3911.52 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 09:15:00 | 3955.50 | 3917.10 | 3914.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 10:15:00 | 4005.10 | 3934.70 | 3922.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 3998.20 | 4002.36 | 3970.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 3998.20 | 4002.36 | 3970.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 3972.00 | 3996.29 | 3970.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 3972.00 | 3996.29 | 3970.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 3964.70 | 3989.97 | 3969.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 3964.70 | 3989.97 | 3969.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 3955.90 | 3983.16 | 3968.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 3956.80 | 3983.16 | 3968.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 3949.30 | 3976.39 | 3966.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:30:00 | 3954.60 | 3976.39 | 3966.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 3983.00 | 3977.71 | 3968.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 3931.80 | 3977.71 | 3968.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 3944.40 | 3971.05 | 3966.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:45:00 | 4008.70 | 3970.46 | 3966.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 4014.30 | 3970.46 | 3966.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 3942.50 | 3982.88 | 3987.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 3942.50 | 3982.88 | 3987.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 12:15:00 | 3930.00 | 3972.31 | 3981.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 10:15:00 | 4000.00 | 3960.86 | 3970.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 10:15:00 | 4000.00 | 3960.86 | 3970.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 4000.00 | 3960.86 | 3970.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 4000.00 | 3960.86 | 3970.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 3997.40 | 3968.16 | 3972.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 3961.80 | 3968.16 | 3972.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 13:15:00 | 4007.00 | 3981.41 | 3978.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 4007.00 | 3981.41 | 3978.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 4107.90 | 4015.83 | 3996.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 4220.20 | 4238.53 | 4175.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 4220.20 | 4238.53 | 4175.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 4174.00 | 4218.78 | 4185.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 4154.30 | 4218.78 | 4185.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 4186.50 | 4212.32 | 4185.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 4186.50 | 4212.32 | 4185.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 4184.80 | 4206.82 | 4185.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 4184.80 | 4206.82 | 4185.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 4184.70 | 4202.39 | 4185.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 4184.70 | 4202.39 | 4185.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 4180.30 | 4197.97 | 4185.25 | EMA400 retest candle locked (from upside) |

### Cycle 155 — SELL (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 09:15:00 | 4130.50 | 4170.99 | 4174.91 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 4205.10 | 4176.00 | 4175.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 15:15:00 | 4245.50 | 4189.90 | 4181.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 15:15:00 | 4202.50 | 4227.79 | 4211.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 15:15:00 | 4202.50 | 4227.79 | 4211.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 4202.50 | 4227.79 | 4211.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 4187.60 | 4227.79 | 4211.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 4170.00 | 4216.23 | 4207.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 4169.20 | 4216.23 | 4207.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 4180.00 | 4208.98 | 4204.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 4202.00 | 4208.98 | 4204.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 12:15:00 | 4185.00 | 4200.56 | 4201.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 4185.00 | 4200.56 | 4201.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 4179.00 | 4193.76 | 4198.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 11:15:00 | 4229.80 | 4196.85 | 4197.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 11:15:00 | 4229.80 | 4196.85 | 4197.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 4229.80 | 4196.85 | 4197.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 4229.80 | 4196.85 | 4197.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 4198.00 | 4197.08 | 4197.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 13:30:00 | 4178.40 | 4192.90 | 4195.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 4182.10 | 4186.21 | 4187.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 4212.00 | 4191.37 | 4189.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 4212.00 | 4191.37 | 4189.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 4282.10 | 4209.52 | 4197.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 12:15:00 | 4282.20 | 4285.86 | 4255.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 13:00:00 | 4282.20 | 4285.86 | 4255.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 4214.20 | 4268.64 | 4257.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 4214.20 | 4268.64 | 4257.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 4226.60 | 4260.23 | 4254.66 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 4199.70 | 4243.88 | 4247.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 4124.10 | 4206.65 | 4227.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 4164.80 | 4164.59 | 4196.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:45:00 | 4166.80 | 4164.59 | 4196.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 4187.70 | 4092.46 | 4126.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 4204.80 | 4092.46 | 4126.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 4087.80 | 4091.53 | 4123.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 4065.10 | 4086.18 | 4117.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 4059.80 | 4080.63 | 4112.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 12:15:00 | 4109.90 | 4075.96 | 4074.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 4109.90 | 4075.96 | 4074.30 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 4062.00 | 4074.19 | 4074.44 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 10:15:00 | 4090.20 | 4077.39 | 4075.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 11:15:00 | 4114.40 | 4084.79 | 4079.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 4089.70 | 4103.22 | 4092.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 4089.70 | 4103.22 | 4092.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 4089.70 | 4103.22 | 4092.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 4091.90 | 4103.22 | 4092.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 4111.90 | 4104.95 | 4094.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:30:00 | 4070.50 | 4104.95 | 4094.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 4100.00 | 4107.33 | 4097.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:30:00 | 4097.80 | 4107.33 | 4097.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 4101.00 | 4106.06 | 4097.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:30:00 | 4100.00 | 4106.06 | 4097.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 4100.00 | 4104.85 | 4098.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 4110.20 | 4104.85 | 4098.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:45:00 | 4110.60 | 4107.62 | 4100.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 4074.20 | 4100.93 | 4098.16 | SL hit (close<static) qty=1.00 sl=4081.10 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 11:15:00 | 4025.10 | 4085.77 | 4091.52 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 4158.50 | 4095.91 | 4089.51 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 4068.00 | 4101.13 | 4103.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 4052.00 | 4086.11 | 4096.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 11:15:00 | 3893.20 | 3882.79 | 3920.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 12:00:00 | 3893.20 | 3882.79 | 3920.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 3921.30 | 3885.27 | 3906.88 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 3937.00 | 3917.98 | 3917.83 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 3900.00 | 3914.66 | 3916.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 3881.10 | 3904.48 | 3911.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 3907.20 | 3887.72 | 3899.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 3907.20 | 3887.72 | 3899.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 3907.20 | 3887.72 | 3899.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 3907.20 | 3887.72 | 3899.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 3919.90 | 3894.16 | 3901.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 3919.90 | 3894.16 | 3901.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 3883.50 | 3892.03 | 3899.73 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 3944.00 | 3910.68 | 3906.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 3954.00 | 3925.17 | 3916.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 3988.00 | 3990.43 | 3962.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:45:00 | 3984.00 | 3990.43 | 3962.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 3990.00 | 3990.35 | 3964.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 3964.80 | 3990.35 | 3964.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 3966.70 | 3996.76 | 3976.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 3966.10 | 3996.76 | 3976.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 3953.00 | 3988.01 | 3974.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 3958.40 | 3988.01 | 3974.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 3949.30 | 3966.90 | 3967.22 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 3976.10 | 3967.93 | 3967.19 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 3954.40 | 3965.22 | 3966.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 3928.50 | 3957.88 | 3962.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 15:15:00 | 3964.00 | 3953.94 | 3959.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 15:15:00 | 3964.00 | 3953.94 | 3959.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 3964.00 | 3953.94 | 3959.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 3925.40 | 3953.94 | 3959.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3937.00 | 3950.55 | 3957.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:45:00 | 3897.70 | 3923.36 | 3937.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 10:30:00 | 3906.00 | 3885.99 | 3886.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 3901.90 | 3889.17 | 3887.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 3901.90 | 3889.17 | 3887.75 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 3855.10 | 3884.15 | 3886.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 09:15:00 | 3851.30 | 3862.11 | 3871.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 3831.00 | 3809.93 | 3826.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 3831.00 | 3809.93 | 3826.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 3831.00 | 3809.93 | 3826.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 3831.00 | 3809.93 | 3826.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 3876.00 | 3823.14 | 3831.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 3876.00 | 3823.14 | 3831.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 3853.20 | 3829.15 | 3833.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:45:00 | 3881.80 | 3829.15 | 3833.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 3880.00 | 3839.32 | 3837.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 3903.10 | 3866.94 | 3854.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 3970.80 | 3984.00 | 3949.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 3970.80 | 3984.00 | 3949.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 3965.00 | 3980.20 | 3950.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 3941.20 | 3972.40 | 3949.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3964.10 | 3970.74 | 3950.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 3995.00 | 3955.32 | 3949.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 4069.50 | 4113.78 | 4118.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 4069.50 | 4113.78 | 4118.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 4062.60 | 4091.08 | 4097.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 4025.40 | 4007.59 | 4036.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 4025.40 | 4007.59 | 4036.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 4108.80 | 4027.83 | 4042.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 4108.80 | 4027.83 | 4042.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 4125.60 | 4047.38 | 4050.38 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 4083.20 | 4054.55 | 4053.37 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 11:15:00 | 4031.90 | 4052.84 | 4052.97 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 13:15:00 | 4107.70 | 4060.17 | 4056.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 4211.20 | 4090.38 | 4070.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 4189.00 | 4215.85 | 4179.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 4189.00 | 4215.85 | 4179.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 4189.00 | 4215.85 | 4179.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 4189.00 | 4215.85 | 4179.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 4180.00 | 4205.18 | 4181.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 4180.00 | 4205.18 | 4181.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 4216.50 | 4207.45 | 4184.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 4293.90 | 4209.18 | 4190.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 4219.80 | 4247.96 | 4238.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:00:00 | 4224.70 | 4243.31 | 4236.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 4192.30 | 4227.70 | 4230.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 4192.30 | 4227.70 | 4230.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 4176.00 | 4203.31 | 4216.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 3992.60 | 3982.33 | 4032.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 3992.60 | 3982.33 | 4032.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 3982.20 | 3980.57 | 4006.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:45:00 | 3949.80 | 3971.49 | 3997.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:30:00 | 3951.00 | 3962.23 | 3988.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 3932.80 | 3952.06 | 3976.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 3945.00 | 3952.06 | 3976.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 3942.00 | 3894.55 | 3910.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 3942.00 | 3894.55 | 3910.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 3961.50 | 3907.94 | 3914.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 10:30:00 | 3944.90 | 3915.13 | 3917.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 3938.70 | 3919.84 | 3919.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 3938.70 | 3919.84 | 3919.43 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 3907.80 | 3919.65 | 3919.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 3900.00 | 3915.72 | 3917.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 3911.00 | 3900.36 | 3907.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 3911.00 | 3900.36 | 3907.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3911.00 | 3900.36 | 3907.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 3911.00 | 3900.36 | 3907.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 3924.00 | 3905.09 | 3909.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 3943.00 | 3905.09 | 3909.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3917.00 | 3907.47 | 3909.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:30:00 | 3902.00 | 3905.70 | 3908.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:30:00 | 3907.00 | 3909.77 | 3910.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 3920.00 | 3911.82 | 3911.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 3920.00 | 3911.82 | 3911.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 15:15:00 | 3934.10 | 3917.66 | 3913.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 3881.80 | 3917.66 | 3915.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 3881.80 | 3917.66 | 3915.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 3881.80 | 3917.66 | 3915.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 3881.80 | 3917.66 | 3915.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 3878.40 | 3909.81 | 3912.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 14:15:00 | 3869.90 | 3897.36 | 3905.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 3897.00 | 3888.02 | 3897.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 11:15:00 | 3897.00 | 3888.02 | 3897.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 3897.00 | 3888.02 | 3897.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 3897.00 | 3888.02 | 3897.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 3939.80 | 3898.38 | 3901.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 3939.80 | 3898.38 | 3901.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 3951.50 | 3909.00 | 3906.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 12:15:00 | 3963.40 | 3938.10 | 3927.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 4209.50 | 4212.97 | 4123.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 11:00:00 | 4209.50 | 4212.97 | 4123.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 4140.20 | 4190.74 | 4135.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 4140.20 | 4190.74 | 4135.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 4121.80 | 4176.95 | 4133.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 4107.70 | 4176.95 | 4133.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 4178.70 | 4177.30 | 4137.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 4121.20 | 4177.30 | 4137.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 4168.00 | 4175.44 | 4140.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:30:00 | 4142.80 | 4175.44 | 4140.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 4146.20 | 4168.77 | 4148.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:00:00 | 4146.20 | 4168.77 | 4148.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 4121.60 | 4159.34 | 4146.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 4121.60 | 4159.34 | 4146.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 4121.00 | 4151.67 | 4143.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 4055.40 | 4151.67 | 4143.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 4062.30 | 4133.80 | 4136.54 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 4193.90 | 4119.84 | 4111.44 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 4081.00 | 4125.33 | 4129.53 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 4140.70 | 4123.55 | 4122.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 4168.50 | 4135.96 | 4128.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 4101.70 | 4133.28 | 4129.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 4101.70 | 4133.28 | 4129.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 4101.70 | 4133.28 | 4129.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 4096.40 | 4133.28 | 4129.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 4128.80 | 4132.38 | 4129.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:00:00 | 4136.00 | 4133.11 | 4129.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:45:00 | 4134.40 | 4133.78 | 4130.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:00:00 | 4139.80 | 4131.77 | 4129.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:00:00 | 4141.20 | 4141.10 | 4135.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 4120.80 | 4137.04 | 4133.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 4123.10 | 4137.04 | 4133.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 4132.90 | 4136.21 | 4133.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 4132.90 | 4136.21 | 4133.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 4100.00 | 4128.97 | 4130.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 4100.00 | 4128.97 | 4130.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 09:15:00 | 4066.40 | 4111.19 | 4121.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 3860.90 | 3859.52 | 3890.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:45:00 | 3872.00 | 3859.52 | 3890.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 3879.90 | 3859.82 | 3880.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 3879.90 | 3859.82 | 3880.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 3898.50 | 3867.56 | 3881.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:45:00 | 3904.40 | 3867.56 | 3881.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 3890.30 | 3872.11 | 3882.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 3910.10 | 3872.11 | 3882.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 3900.50 | 3889.17 | 3888.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 10:15:00 | 3939.00 | 3911.27 | 3900.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 12:15:00 | 3910.30 | 3913.21 | 3903.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:45:00 | 3913.20 | 3913.21 | 3903.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 3903.00 | 3911.17 | 3903.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 3903.00 | 3911.17 | 3903.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 3892.80 | 3907.49 | 3902.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 3892.80 | 3907.49 | 3902.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 3898.20 | 3905.63 | 3902.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 3905.00 | 3905.63 | 3902.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 3905.00 | 3905.45 | 3902.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 3899.90 | 3905.45 | 3902.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 3902.10 | 3904.78 | 3902.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:00:00 | 3902.10 | 3904.78 | 3902.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 3914.20 | 3906.67 | 3903.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:30:00 | 3910.00 | 3906.67 | 3903.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 3884.00 | 3902.78 | 3902.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 3884.00 | 3902.78 | 3902.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 15:15:00 | 3898.80 | 3901.98 | 3902.11 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 3914.80 | 3904.55 | 3903.26 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 3894.80 | 3902.49 | 3902.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 3884.20 | 3897.64 | 3900.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 14:15:00 | 3898.10 | 3897.73 | 3900.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 14:15:00 | 3898.10 | 3897.73 | 3900.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 3898.10 | 3897.73 | 3900.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 3898.10 | 3897.73 | 3900.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 3890.00 | 3896.18 | 3899.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 3876.10 | 3896.41 | 3898.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 3920.60 | 3901.25 | 3900.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 3920.60 | 3901.25 | 3900.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 3936.70 | 3914.26 | 3907.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 3877.10 | 3907.39 | 3905.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 3877.10 | 3907.39 | 3905.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 3877.10 | 3907.39 | 3905.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 3877.10 | 3907.39 | 3905.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 3800.80 | 3886.08 | 3896.11 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 10:15:00 | 3862.10 | 3842.25 | 3839.82 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 3827.30 | 3842.64 | 3842.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 3813.00 | 3836.71 | 3840.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 3791.40 | 3779.14 | 3798.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 3791.40 | 3779.14 | 3798.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 3791.40 | 3779.14 | 3798.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 3805.60 | 3779.14 | 3798.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 3792.00 | 3781.71 | 3797.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 3820.00 | 3786.75 | 3798.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 3802.90 | 3789.98 | 3799.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 3808.50 | 3789.98 | 3799.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 3803.90 | 3792.76 | 3799.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 3804.40 | 3792.76 | 3799.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 3791.10 | 3792.43 | 3798.82 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 3842.80 | 3806.45 | 3804.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 3861.00 | 3817.36 | 3809.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 3825.20 | 3841.45 | 3825.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 12:15:00 | 3825.20 | 3841.45 | 3825.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 3825.20 | 3841.45 | 3825.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:00:00 | 3825.20 | 3841.45 | 3825.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 3834.20 | 3840.00 | 3826.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:30:00 | 3822.20 | 3840.00 | 3826.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 3903.60 | 3852.72 | 3833.59 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 3780.50 | 3831.03 | 3835.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 3779.90 | 3796.49 | 3813.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 3802.60 | 3794.15 | 3809.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:00:00 | 3802.60 | 3794.15 | 3809.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 3809.50 | 3798.32 | 3808.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 3787.10 | 3806.01 | 3808.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 3824.90 | 3808.12 | 3808.80 | SL hit (close>static) qty=1.00 sl=3813.70 alert=retest2 |

### Cycle 200 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 3822.70 | 3811.03 | 3810.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 3861.00 | 3826.06 | 3818.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 14:15:00 | 3877.00 | 3881.89 | 3863.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 15:00:00 | 3877.00 | 3881.89 | 3863.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 3862.40 | 3877.99 | 3863.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 3857.30 | 3877.99 | 3863.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 3861.90 | 3874.78 | 3863.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:45:00 | 3871.50 | 3874.86 | 3864.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 3874.80 | 3875.09 | 3868.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:45:00 | 3872.30 | 3872.17 | 3867.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:30:00 | 3870.70 | 3869.55 | 3867.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 3858.00 | 3867.24 | 3866.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 3860.70 | 3867.24 | 3866.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 3837.20 | 3861.23 | 3863.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 3837.20 | 3861.23 | 3863.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 14:15:00 | 3815.00 | 3851.98 | 3859.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 11:15:00 | 3804.80 | 3794.69 | 3813.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 11:15:00 | 3804.80 | 3794.69 | 3813.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 3804.80 | 3794.69 | 3813.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 3806.00 | 3794.69 | 3813.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 3800.70 | 3796.22 | 3809.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 3797.60 | 3796.22 | 3809.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 3807.90 | 3798.99 | 3808.65 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 13:15:00 | 3869.20 | 3812.92 | 3811.69 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 3805.00 | 3813.78 | 3814.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 14:15:00 | 3755.80 | 3799.82 | 3807.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 3741.40 | 3730.66 | 3762.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 3741.40 | 3730.66 | 3762.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 3817.00 | 3750.51 | 3765.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 3722.50 | 3752.47 | 3765.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 15:15:00 | 3783.90 | 3771.46 | 3770.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 3783.90 | 3771.46 | 3770.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 3803.60 | 3782.34 | 3775.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 3802.00 | 3802.87 | 3789.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 3802.00 | 3802.87 | 3789.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 3802.00 | 3802.87 | 3789.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:45:00 | 3833.80 | 3804.05 | 3794.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 3766.50 | 3796.76 | 3795.97 | SL hit (close<static) qty=1.00 sl=3769.80 alert=retest2 |

### Cycle 205 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 3762.80 | 3789.97 | 3792.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 3703.20 | 3772.62 | 3784.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 11:15:00 | 3583.40 | 3579.73 | 3631.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 11:45:00 | 3583.80 | 3579.73 | 3631.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 3596.90 | 3587.57 | 3610.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:00:00 | 3585.70 | 3587.20 | 3608.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 14:15:00 | 3613.00 | 3592.26 | 3606.77 | SL hit (close>static) qty=1.00 sl=3611.10 alert=retest2 |

### Cycle 206 — BUY (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 13:15:00 | 3637.10 | 3610.29 | 3608.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 3658.90 | 3624.84 | 3616.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 11:15:00 | 3614.40 | 3625.66 | 3618.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 11:15:00 | 3614.40 | 3625.66 | 3618.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 3614.40 | 3625.66 | 3618.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:30:00 | 3611.60 | 3625.66 | 3618.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 3629.80 | 3626.49 | 3619.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 13:45:00 | 3636.50 | 3627.07 | 3620.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 14:30:00 | 3638.00 | 3631.14 | 3623.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 3602.40 | 3624.23 | 3621.31 | SL hit (close<static) qty=1.00 sl=3606.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 3573.10 | 3614.00 | 3616.93 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 3639.10 | 3615.59 | 3614.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 3667.80 | 3626.03 | 3619.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 3603.40 | 3627.08 | 3622.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 3603.40 | 3627.08 | 3622.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 3603.40 | 3627.08 | 3622.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 3592.00 | 3627.08 | 3622.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 3597.50 | 3621.16 | 3620.45 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 3598.20 | 3616.57 | 3618.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 3584.40 | 3605.89 | 3612.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 14:15:00 | 3600.00 | 3599.21 | 3608.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 3600.00 | 3599.21 | 3608.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 3600.00 | 3599.21 | 3608.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 3600.00 | 3599.21 | 3608.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 3601.60 | 3595.81 | 3605.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 3601.60 | 3595.81 | 3605.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 3622.00 | 3601.05 | 3606.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:45:00 | 3621.90 | 3601.05 | 3606.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 3640.00 | 3608.84 | 3609.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:30:00 | 3638.00 | 3608.84 | 3609.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 12:15:00 | 3643.20 | 3615.71 | 3612.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 3688.80 | 3634.87 | 3622.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 3838.60 | 3849.31 | 3784.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 13:00:00 | 3838.60 | 3849.31 | 3784.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 3849.90 | 3843.89 | 3793.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:45:00 | 3788.80 | 3843.89 | 3793.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3772.90 | 3829.01 | 3795.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 3752.70 | 3829.01 | 3795.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 3735.60 | 3810.32 | 3789.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 3730.10 | 3810.32 | 3789.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 3805.90 | 3813.60 | 3798.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 3795.40 | 3813.60 | 3798.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 3791.90 | 3809.26 | 3797.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 3791.90 | 3809.26 | 3797.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 3784.50 | 3804.31 | 3796.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 3784.00 | 3804.31 | 3796.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 3823.50 | 3808.15 | 3799.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 15:15:00 | 3850.00 | 3803.82 | 3799.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 13:15:00 | 3789.80 | 3876.91 | 3888.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 3789.80 | 3876.91 | 3888.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 3712.40 | 3799.31 | 3833.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 3786.50 | 3777.39 | 3810.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 14:00:00 | 3786.50 | 3777.39 | 3810.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 3813.10 | 3785.30 | 3805.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 3813.10 | 3785.30 | 3805.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 3829.20 | 3794.08 | 3807.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 3829.20 | 3794.08 | 3807.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 3824.60 | 3800.18 | 3809.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 14:00:00 | 3804.00 | 3804.08 | 3809.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 15:15:00 | 3767.00 | 3805.31 | 3809.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:00:00 | 3804.50 | 3798.91 | 3804.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:45:00 | 3789.50 | 3797.83 | 3803.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 3762.30 | 3785.80 | 3795.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 3800.00 | 3785.80 | 3795.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 3885.30 | 3793.32 | 3792.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 3885.30 | 3793.32 | 3792.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 15:15:00 | 3900.00 | 3814.65 | 3802.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 3833.20 | 3851.49 | 3830.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 3833.20 | 3851.49 | 3830.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 3820.00 | 3845.19 | 3829.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 3829.60 | 3845.19 | 3829.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 3838.00 | 3843.75 | 3830.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 3868.00 | 3848.30 | 3833.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 3963.60 | 3874.44 | 3855.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-25 09:15:00 | 4254.80 | 4068.09 | 3995.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 4146.00 | 4279.29 | 4286.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 4120.30 | 4247.49 | 4271.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 4170.30 | 4165.31 | 4210.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:15:00 | 4201.10 | 4165.31 | 4210.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 4238.30 | 4179.91 | 4212.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 4238.30 | 4179.91 | 4212.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 4230.00 | 4189.93 | 4214.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:15:00 | 4244.40 | 4189.93 | 4214.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 4291.70 | 4229.62 | 4228.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 4300.00 | 4243.70 | 4234.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 4325.00 | 4342.52 | 4297.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 14:30:00 | 4319.40 | 4342.52 | 4297.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 4182.00 | 4305.24 | 4288.09 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 4151.00 | 4274.39 | 4275.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 4122.20 | 4163.25 | 4193.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3807.60 | 3798.73 | 3867.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 3807.60 | 3798.73 | 3867.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 3906.50 | 3831.09 | 3866.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 3906.50 | 3831.09 | 3866.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 3907.60 | 3846.39 | 3869.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:30:00 | 3901.70 | 3846.39 | 3869.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 3989.00 | 3894.41 | 3887.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 4000.00 | 3936.84 | 3910.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 3960.10 | 3988.67 | 3951.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 11:00:00 | 3960.10 | 3988.67 | 3951.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 3960.10 | 3977.31 | 3957.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:45:00 | 3960.00 | 3977.31 | 3957.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 3960.10 | 3973.86 | 3957.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 4079.10 | 3973.86 | 3957.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 3954.90 | 4016.61 | 4017.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 3954.90 | 4016.61 | 4017.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 15:15:00 | 3937.00 | 4000.69 | 4010.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 13:15:00 | 3978.30 | 3958.17 | 3981.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 13:15:00 | 3978.30 | 3958.17 | 3981.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 3978.30 | 3958.17 | 3981.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 3978.30 | 3958.17 | 3981.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 3981.00 | 3962.74 | 3981.18 | EMA400 retest candle locked (from downside) |

### Cycle 218 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 4089.00 | 3991.55 | 3991.30 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 3972.10 | 4006.04 | 4010.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 3865.00 | 3977.83 | 3997.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3978.20 | 3902.36 | 3937.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3978.20 | 3902.36 | 3937.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3978.20 | 3902.36 | 3937.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 3978.20 | 3902.36 | 3937.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 3882.20 | 3898.32 | 3932.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:15:00 | 3865.10 | 3898.32 | 3932.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 3873.30 | 3889.15 | 3919.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 3919.50 | 3793.94 | 3787.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 3919.50 | 3793.94 | 3787.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 3970.50 | 3829.25 | 3803.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 10:15:00 | 3863.00 | 3866.48 | 3838.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:45:00 | 3858.50 | 3866.48 | 3838.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 3849.90 | 3861.82 | 3843.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 3842.00 | 3861.82 | 3843.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 3847.20 | 3858.90 | 3843.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 3847.20 | 3858.90 | 3843.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 3844.50 | 3856.02 | 3843.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 3925.00 | 3856.02 | 3843.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 3864.50 | 3883.32 | 3869.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 09:15:00 | 4250.95 | 4151.40 | 4094.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 4163.40 | 4292.45 | 4300.91 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 4252.30 | 4180.64 | 4172.17 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-18 13:15:00 | 3259.35 | 2024-04-18 13:15:00 | 3260.20 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-04-19 12:30:00 | 3269.90 | 2024-04-22 13:15:00 | 3234.35 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest1 | 2024-04-29 09:45:00 | 3365.60 | 2024-04-30 09:15:00 | 3533.88 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-04-29 11:00:00 | 3374.90 | 2024-04-30 09:15:00 | 3543.65 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-04-29 09:45:00 | 3365.60 | 2024-04-30 10:15:00 | 3702.16 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-04-29 11:00:00 | 3374.90 | 2024-04-30 10:15:00 | 3712.39 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-05-08 09:15:00 | 3837.50 | 2024-05-08 09:15:00 | 3797.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-05-08 11:30:00 | 3836.25 | 2024-05-08 13:15:00 | 3792.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-05-08 15:00:00 | 3848.00 | 2024-05-09 14:15:00 | 3789.85 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-05-29 09:15:00 | 4643.95 | 2024-05-29 12:15:00 | 4545.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-06-05 15:15:00 | 4135.00 | 2024-06-06 09:15:00 | 4320.00 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2024-07-02 11:30:00 | 4627.95 | 2024-07-03 11:15:00 | 4709.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-07-03 09:30:00 | 4627.20 | 2024-07-03 11:15:00 | 4709.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-07-18 14:00:00 | 3983.85 | 2024-07-24 09:15:00 | 4055.05 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-08-08 09:15:00 | 3905.30 | 2024-08-09 14:15:00 | 3951.25 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-13 13:15:00 | 3999.40 | 2024-08-14 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-08-19 14:45:00 | 3979.95 | 2024-08-19 15:15:00 | 3955.50 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-08-20 10:30:00 | 3980.30 | 2024-08-20 13:15:00 | 3959.90 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-08-20 11:45:00 | 3978.15 | 2024-08-20 13:15:00 | 3959.90 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-08-20 12:45:00 | 3979.80 | 2024-08-20 13:15:00 | 3959.90 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-09-10 11:30:00 | 3830.80 | 2024-09-12 15:15:00 | 3859.95 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-09-10 12:00:00 | 3832.15 | 2024-09-12 15:15:00 | 3859.95 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-09-11 11:30:00 | 3820.20 | 2024-09-12 15:15:00 | 3859.95 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-09-12 09:45:00 | 3831.80 | 2024-09-12 15:15:00 | 3859.95 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-09-25 15:00:00 | 3922.00 | 2024-09-26 09:15:00 | 3868.05 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-10-07 10:30:00 | 3761.00 | 2024-10-09 11:15:00 | 3850.20 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-10-08 10:00:00 | 3748.35 | 2024-10-09 11:15:00 | 3850.20 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-10-11 09:15:00 | 3896.90 | 2024-10-21 09:15:00 | 3972.35 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2024-10-11 13:30:00 | 3901.40 | 2024-10-21 09:15:00 | 3972.35 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2024-10-11 14:30:00 | 3893.55 | 2024-10-21 09:15:00 | 3972.35 | STOP_HIT | 1.00 | 2.02% |
| SELL | retest2 | 2024-10-31 10:15:00 | 3439.80 | 2024-11-06 11:15:00 | 3505.65 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-11-05 10:30:00 | 3442.55 | 2024-11-06 11:15:00 | 3505.65 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-11-05 15:15:00 | 3445.00 | 2024-11-06 11:15:00 | 3505.65 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-11-06 10:15:00 | 3440.15 | 2024-11-06 11:15:00 | 3505.65 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-11-12 13:45:00 | 3548.00 | 2024-11-13 09:15:00 | 3445.40 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-11-12 14:15:00 | 3540.25 | 2024-11-13 09:15:00 | 3445.40 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-11-18 11:00:00 | 3381.50 | 2024-11-19 09:15:00 | 3460.80 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-11-18 12:30:00 | 3387.65 | 2024-11-19 09:15:00 | 3460.80 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-11-18 14:00:00 | 3379.80 | 2024-11-19 09:15:00 | 3460.80 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-11-19 15:15:00 | 3387.90 | 2024-11-26 10:15:00 | 3468.65 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-11-21 09:15:00 | 3379.70 | 2024-11-26 10:15:00 | 3468.65 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-11-21 10:30:00 | 3377.00 | 2024-11-26 10:15:00 | 3468.65 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-11-21 11:45:00 | 3355.00 | 2024-11-26 10:15:00 | 3468.65 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-11-21 12:15:00 | 3380.35 | 2024-11-26 10:15:00 | 3468.65 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-11-26 09:45:00 | 3375.00 | 2024-11-26 10:15:00 | 3468.65 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-12-02 11:15:00 | 3587.05 | 2024-12-06 09:15:00 | 3582.25 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-12-11 15:00:00 | 3677.00 | 2024-12-12 09:15:00 | 3577.80 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-12-20 12:15:00 | 3396.65 | 2024-12-26 13:15:00 | 3362.40 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2025-01-06 14:45:00 | 3482.10 | 2025-01-09 14:15:00 | 3439.05 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-01-06 15:15:00 | 3468.00 | 2025-01-09 14:15:00 | 3439.05 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-01-08 10:45:00 | 3505.90 | 2025-01-09 14:15:00 | 3439.05 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-01-21 10:15:00 | 3207.00 | 2025-01-21 12:15:00 | 3274.20 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-01-30 09:30:00 | 3256.90 | 2025-02-03 12:15:00 | 3284.80 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-02-04 10:45:00 | 3276.45 | 2025-02-05 14:15:00 | 3382.35 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2025-02-06 15:00:00 | 3377.05 | 2025-02-07 10:15:00 | 3286.50 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-02-07 09:30:00 | 3360.00 | 2025-02-07 10:15:00 | 3286.50 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-02-07 10:00:00 | 3355.60 | 2025-02-07 10:15:00 | 3286.50 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-02-20 09:15:00 | 3088.85 | 2025-02-27 10:15:00 | 3064.40 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-02-27 10:00:00 | 3086.70 | 2025-02-27 10:15:00 | 3064.40 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-02-28 12:30:00 | 3055.95 | 2025-03-04 09:15:00 | 3324.95 | STOP_HIT | 1.00 | -8.80% |
| SELL | retest2 | 2025-02-28 15:15:00 | 3041.00 | 2025-03-04 09:15:00 | 3324.95 | STOP_HIT | 1.00 | -9.34% |
| SELL | retest2 | 2025-03-03 09:30:00 | 3046.30 | 2025-03-04 09:15:00 | 3324.95 | STOP_HIT | 1.00 | -9.15% |
| BUY | retest2 | 2025-03-19 09:15:00 | 3406.10 | 2025-03-25 14:15:00 | 3492.70 | STOP_HIT | 1.00 | 2.54% |
| SELL | retest2 | 2025-04-04 10:15:00 | 3189.45 | 2025-04-07 09:15:00 | 2870.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 11:45:00 | 3188.45 | 2025-04-07 09:15:00 | 2869.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 15:00:00 | 3286.20 | 2025-04-25 09:15:00 | 3227.10 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-04-24 11:45:00 | 3278.00 | 2025-04-25 09:15:00 | 3227.10 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-04-30 09:15:00 | 3508.30 | 2025-05-13 10:15:00 | 3859.13 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-26 13:00:00 | 4004.30 | 2025-05-29 09:15:00 | 4113.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-05-26 14:45:00 | 3999.80 | 2025-05-29 09:15:00 | 4113.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-05-27 09:15:00 | 3998.70 | 2025-05-29 09:15:00 | 4113.00 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-05-27 12:45:00 | 4006.70 | 2025-05-29 09:15:00 | 4113.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-06-02 13:45:00 | 4144.10 | 2025-06-03 12:15:00 | 4091.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-06-02 14:45:00 | 4143.40 | 2025-06-03 12:15:00 | 4091.40 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-06-03 10:30:00 | 4139.20 | 2025-06-03 12:15:00 | 4091.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-03 12:00:00 | 4145.90 | 2025-06-03 12:15:00 | 4091.40 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-11 13:30:00 | 4216.50 | 2025-06-16 14:15:00 | 4005.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 13:30:00 | 4216.50 | 2025-06-17 09:15:00 | 4068.50 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-07-02 13:45:00 | 4008.70 | 2025-07-04 11:15:00 | 3942.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-02 14:15:00 | 4014.30 | 2025-07-04 11:15:00 | 3942.50 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-07-07 12:15:00 | 3961.80 | 2025-07-07 13:15:00 | 4007.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-07-16 11:15:00 | 4202.00 | 2025-07-16 12:15:00 | 4185.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-17 13:30:00 | 4178.40 | 2025-07-21 09:15:00 | 4212.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-21 09:15:00 | 4182.10 | 2025-07-21 09:15:00 | 4212.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-28 11:30:00 | 4065.10 | 2025-07-30 12:15:00 | 4109.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-07-28 12:30:00 | 4059.80 | 2025-07-30 12:15:00 | 4109.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-08-01 15:15:00 | 4110.20 | 2025-08-04 10:15:00 | 4074.20 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-08-04 09:45:00 | 4110.60 | 2025-08-04 10:15:00 | 4074.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-28 11:45:00 | 3897.70 | 2025-09-02 11:15:00 | 3901.90 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-09-02 10:30:00 | 3906.00 | 2025-09-02 11:15:00 | 3901.90 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-09-15 09:15:00 | 3995.00 | 2025-09-23 10:15:00 | 4069.50 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-10-07 09:15:00 | 4293.90 | 2025-10-09 09:15:00 | 4192.30 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-10-08 12:45:00 | 4219.80 | 2025-10-09 09:15:00 | 4192.30 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-10-08 14:00:00 | 4224.70 | 2025-10-09 09:15:00 | 4192.30 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-16 11:45:00 | 3949.80 | 2025-10-23 11:15:00 | 3938.70 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-10-16 13:30:00 | 3951.00 | 2025-10-23 11:15:00 | 3938.70 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-10-17 09:30:00 | 3932.80 | 2025-10-23 11:15:00 | 3938.70 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-10-17 10:15:00 | 3945.00 | 2025-10-23 11:15:00 | 3938.70 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-10-23 10:30:00 | 3944.90 | 2025-10-23 11:15:00 | 3938.70 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-10-27 10:30:00 | 3902.00 | 2025-10-27 13:15:00 | 3920.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-10-27 12:30:00 | 3907.00 | 2025-10-27 13:15:00 | 3920.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-11-18 12:00:00 | 4136.00 | 2025-11-19 13:15:00 | 4100.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-11-18 12:45:00 | 4134.40 | 2025-11-19 13:15:00 | 4100.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-11-18 15:00:00 | 4139.80 | 2025-11-19 13:15:00 | 4100.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-11-19 11:00:00 | 4141.20 | 2025-11-19 13:15:00 | 4100.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-05 10:15:00 | 3876.10 | 2025-12-05 10:15:00 | 3920.60 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-29 11:15:00 | 3787.10 | 2025-12-29 13:15:00 | 3824.90 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-01-02 10:45:00 | 3871.50 | 2026-01-05 13:15:00 | 3837.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-01-02 14:30:00 | 3874.80 | 2026-01-05 13:15:00 | 3837.20 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-05 09:45:00 | 3872.30 | 2026-01-05 13:15:00 | 3837.20 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-05 11:30:00 | 3870.70 | 2026-01-05 13:15:00 | 3837.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-01-13 09:30:00 | 3722.50 | 2026-01-13 15:15:00 | 3783.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-01-16 14:45:00 | 3833.80 | 2026-01-19 14:15:00 | 3766.50 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-01-23 13:00:00 | 3585.70 | 2026-01-23 14:15:00 | 3613.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-27 09:15:00 | 3572.20 | 2026-01-27 11:15:00 | 3632.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2026-01-28 13:45:00 | 3636.50 | 2026-01-29 09:15:00 | 3602.40 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-01-28 14:30:00 | 3638.00 | 2026-01-29 09:15:00 | 3602.40 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-02-06 15:15:00 | 3850.00 | 2026-02-11 13:15:00 | 3789.80 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-02-16 14:00:00 | 3804.00 | 2026-02-18 14:15:00 | 3885.30 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-02-16 15:15:00 | 3767.00 | 2026-02-18 14:15:00 | 3885.30 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-02-17 11:00:00 | 3804.50 | 2026-02-18 14:15:00 | 3885.30 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-02-17 11:45:00 | 3789.50 | 2026-02-18 14:15:00 | 3885.30 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-02-20 10:30:00 | 3868.00 | 2026-02-25 09:15:00 | 4254.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-23 09:15:00 | 3963.60 | 2026-02-26 09:15:00 | 4359.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-20 09:15:00 | 4079.10 | 2026-03-23 14:15:00 | 3954.90 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2026-04-01 11:15:00 | 3865.10 | 2026-04-08 10:15:00 | 3919.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-04-01 13:45:00 | 3873.30 | 2026-04-08 10:15:00 | 3919.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-04-10 09:15:00 | 3925.00 | 2026-04-22 09:15:00 | 4250.95 | TARGET_HIT | 1.00 | 8.30% |
| BUY | retest2 | 2026-04-13 10:15:00 | 3864.50 | 2026-04-23 09:15:00 | 4317.50 | TARGET_HIT | 1.00 | 11.72% |
