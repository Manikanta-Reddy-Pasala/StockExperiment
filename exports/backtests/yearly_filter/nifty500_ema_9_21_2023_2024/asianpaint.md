# Asian Paints Ltd. (ASIANPAINT)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 2600.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 222 |
| ALERT1 | 147 |
| ALERT2 | 145 |
| ALERT2_SKIP | 72 |
| ALERT3 | 390 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 170 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 174 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 183 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 127
- **Target hits / Stop hits / Partials:** 2 / 173 / 8
- **Avg / median % per leg:** 0.20% / -0.55%
- **Sum % (uncompounded):** 37.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 90 | 25 | 27.8% | 1 | 89 | 0 | -0.03% | -2.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 90 | 25 | 27.8% | 1 | 89 | 0 | -0.03% | -2.3% |
| SELL (all) | 93 | 31 | 33.3% | 1 | 84 | 8 | 0.42% | 39.5% |
| SELL @ 2nd Alert (retest1) | 9 | 8 | 88.9% | 0 | 5 | 4 | 4.79% | 43.2% |
| SELL @ 3rd Alert (retest2) | 84 | 23 | 27.4% | 1 | 79 | 4 | -0.04% | -3.7% |
| retest1 (combined) | 9 | 8 | 88.9% | 0 | 5 | 4 | 4.79% | 43.2% |
| retest2 (combined) | 174 | 48 | 27.6% | 2 | 168 | 4 | -0.03% | -5.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 3100.00 | 3123.11 | 3125.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 10:15:00 | 3085.00 | 3101.07 | 3107.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 10:15:00 | 3096.00 | 3089.81 | 3097.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 10:15:00 | 3096.00 | 3089.81 | 3097.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 3096.00 | 3089.81 | 3097.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 11:00:00 | 3096.00 | 3089.81 | 3097.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 11:15:00 | 3100.00 | 3091.85 | 3097.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 12:00:00 | 3100.00 | 3091.85 | 3097.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 12:15:00 | 3102.80 | 3094.04 | 3097.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 13:30:00 | 3085.00 | 3094.42 | 3097.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 14:15:00 | 3087.80 | 3094.42 | 3097.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 09:15:00 | 3107.00 | 3095.16 | 3097.14 | SL hit (close>static) qty=1.00 sl=3105.60 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 3119.80 | 3100.09 | 3099.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 12:15:00 | 3121.60 | 3107.27 | 3102.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 3115.50 | 3116.88 | 3110.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 11:30:00 | 3114.20 | 3116.88 | 3110.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 3112.25 | 3115.61 | 3110.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 14:00:00 | 3112.25 | 3115.61 | 3110.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 14:15:00 | 3102.80 | 3113.05 | 3110.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 14:45:00 | 3101.55 | 3113.05 | 3110.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 15:15:00 | 3101.00 | 3110.64 | 3109.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:15:00 | 3091.05 | 3110.64 | 3109.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 09:15:00 | 3097.55 | 3108.02 | 3108.31 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 3127.65 | 3111.53 | 3109.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 13:15:00 | 3129.75 | 3121.33 | 3115.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 11:15:00 | 3121.10 | 3124.58 | 3119.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 12:00:00 | 3121.10 | 3124.58 | 3119.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 12:15:00 | 3128.55 | 3125.37 | 3120.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 12:30:00 | 3123.00 | 3125.37 | 3120.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 3185.00 | 3150.83 | 3139.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 10:45:00 | 3189.35 | 3161.74 | 3145.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 14:30:00 | 3204.00 | 3173.82 | 3156.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 15:15:00 | 3196.95 | 3173.82 | 3156.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-05 15:15:00 | 3200.65 | 3211.33 | 3212.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 15:15:00 | 3200.65 | 3211.33 | 3212.17 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 09:15:00 | 3233.00 | 3215.67 | 3214.06 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 13:15:00 | 3200.00 | 3211.86 | 3212.78 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 10:15:00 | 3219.30 | 3214.08 | 3213.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 15:15:00 | 3226.05 | 3219.55 | 3216.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 10:15:00 | 3220.55 | 3220.64 | 3217.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 11:00:00 | 3220.55 | 3220.64 | 3217.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 3207.00 | 3217.92 | 3216.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 3207.00 | 3217.92 | 3216.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 3198.05 | 3213.94 | 3215.03 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 09:15:00 | 3224.20 | 3215.08 | 3215.00 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 10:15:00 | 3181.10 | 3208.29 | 3211.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 12:15:00 | 3178.20 | 3198.38 | 3206.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 13:15:00 | 3190.10 | 3181.54 | 3190.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 13:15:00 | 3190.10 | 3181.54 | 3190.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 13:15:00 | 3190.10 | 3181.54 | 3190.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 14:00:00 | 3190.10 | 3181.54 | 3190.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 14:15:00 | 3193.20 | 3183.87 | 3191.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 14:45:00 | 3202.25 | 3183.87 | 3191.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 15:15:00 | 3197.40 | 3186.58 | 3191.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:15:00 | 3248.15 | 3186.58 | 3191.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 09:15:00 | 3267.95 | 3202.85 | 3198.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 12:15:00 | 3270.65 | 3256.92 | 3237.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 13:15:00 | 3307.85 | 3308.76 | 3295.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-19 13:45:00 | 3307.85 | 3308.76 | 3295.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 3289.55 | 3305.54 | 3297.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:45:00 | 3283.35 | 3305.54 | 3297.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 3288.15 | 3302.06 | 3296.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 3285.25 | 3302.06 | 3296.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 3288.95 | 3299.44 | 3295.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 12:00:00 | 3288.95 | 3299.44 | 3295.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 12:15:00 | 3304.25 | 3300.40 | 3296.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 13:15:00 | 3309.00 | 3300.40 | 3296.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 15:00:00 | 3319.55 | 3304.13 | 3298.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 10:45:00 | 3308.40 | 3306.81 | 3301.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 11:15:00 | 3316.15 | 3306.81 | 3301.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 14:15:00 | 3316.85 | 3310.84 | 3305.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 15:15:00 | 3322.00 | 3310.84 | 3305.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 10:15:00 | 3296.20 | 3309.33 | 3306.25 | SL hit (close<static) qty=1.00 sl=3300.05 alert=retest2 |

### Cycle 13 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 3281.85 | 3303.84 | 3304.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 3274.75 | 3298.02 | 3301.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 11:15:00 | 3283.75 | 3276.55 | 3286.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 11:15:00 | 3283.75 | 3276.55 | 3286.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 3283.75 | 3276.55 | 3286.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 12:00:00 | 3283.75 | 3276.55 | 3286.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 12:15:00 | 3284.90 | 3278.22 | 3286.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 12:30:00 | 3288.60 | 3278.22 | 3286.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 13:15:00 | 3300.00 | 3282.57 | 3287.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 13:30:00 | 3298.10 | 3282.57 | 3287.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 14:15:00 | 3297.55 | 3285.57 | 3288.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 14:30:00 | 3296.90 | 3285.57 | 3288.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 09:15:00 | 3304.50 | 3291.18 | 3290.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 13:15:00 | 3309.00 | 3296.97 | 3293.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 3352.20 | 3357.55 | 3346.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 3352.20 | 3357.55 | 3346.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 3352.20 | 3357.55 | 3346.03 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 11:15:00 | 3334.10 | 3346.54 | 3346.68 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 09:15:00 | 3388.90 | 3354.58 | 3350.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 14:15:00 | 3401.00 | 3382.42 | 3371.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 3381.15 | 3385.71 | 3375.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 10:15:00 | 3381.15 | 3385.71 | 3375.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 3381.15 | 3385.71 | 3375.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:00:00 | 3381.15 | 3385.71 | 3375.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 3360.80 | 3380.73 | 3374.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:00:00 | 3360.80 | 3380.73 | 3374.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 3350.65 | 3374.72 | 3372.17 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 14:15:00 | 3346.05 | 3365.62 | 3368.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 10:15:00 | 3326.75 | 3352.69 | 3361.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 13:15:00 | 3342.45 | 3337.91 | 3351.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-10 14:00:00 | 3342.45 | 3337.91 | 3351.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 3346.05 | 3339.53 | 3350.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 14:45:00 | 3344.00 | 3339.53 | 3350.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 3374.50 | 3346.07 | 3351.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:45:00 | 3388.00 | 3346.07 | 3351.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 3394.65 | 3355.78 | 3355.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 13:15:00 | 3397.00 | 3380.46 | 3372.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 3391.50 | 3395.85 | 3385.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 14:00:00 | 3391.50 | 3395.85 | 3385.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 3402.65 | 3397.21 | 3387.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:15:00 | 3406.55 | 3397.97 | 3388.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 11:15:00 | 3406.65 | 3397.91 | 3390.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 12:00:00 | 3407.95 | 3399.92 | 3391.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 11:15:00 | 3478.75 | 3509.97 | 3512.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 11:15:00 | 3478.75 | 3509.97 | 3512.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 12:15:00 | 3445.10 | 3496.99 | 3506.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 15:15:00 | 3383.00 | 3379.28 | 3405.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-28 09:15:00 | 3383.25 | 3379.28 | 3405.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 3385.50 | 3381.41 | 3394.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 15:15:00 | 3394.40 | 3381.41 | 3394.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 3394.40 | 3384.01 | 3394.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-31 09:15:00 | 3340.00 | 3384.01 | 3394.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 14:15:00 | 3351.50 | 3344.49 | 3343.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 14:15:00 | 3351.50 | 3344.49 | 3343.92 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 09:15:00 | 3327.00 | 3340.59 | 3342.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 11:15:00 | 3317.60 | 3333.72 | 3338.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 14:15:00 | 3330.15 | 3325.82 | 3333.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 14:15:00 | 3330.15 | 3325.82 | 3333.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 3330.15 | 3325.82 | 3333.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 15:00:00 | 3330.15 | 3325.82 | 3333.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 15:15:00 | 3325.20 | 3325.69 | 3332.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 09:15:00 | 3295.95 | 3325.69 | 3332.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 11:15:00 | 3189.05 | 3179.81 | 3179.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 3189.05 | 3179.81 | 3179.19 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 15:15:00 | 3178.00 | 3179.03 | 3179.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 10:15:00 | 3174.70 | 3178.07 | 3178.64 | Break + close below crossover candle low |

### Cycle 24 — BUY (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 09:15:00 | 3201.50 | 3179.39 | 3178.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 10:15:00 | 3206.95 | 3184.90 | 3181.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 15:15:00 | 3284.00 | 3287.52 | 3275.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-31 09:15:00 | 3276.55 | 3287.52 | 3275.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 3271.80 | 3284.38 | 3274.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:45:00 | 3274.00 | 3284.38 | 3274.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 3271.45 | 3281.79 | 3274.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:45:00 | 3267.10 | 3281.79 | 3274.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 3277.10 | 3280.85 | 3274.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:30:00 | 3271.40 | 3280.85 | 3274.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 3277.00 | 3280.08 | 3274.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 12:30:00 | 3271.05 | 3280.08 | 3274.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 13:15:00 | 3273.65 | 3278.80 | 3274.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 13:45:00 | 3271.10 | 3278.80 | 3274.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 3269.75 | 3276.99 | 3274.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 15:00:00 | 3269.75 | 3276.99 | 3274.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 15:15:00 | 3255.00 | 3272.59 | 3272.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 09:15:00 | 3211.60 | 3227.22 | 3236.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-06 11:15:00 | 3227.00 | 3225.67 | 3234.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-06 12:00:00 | 3227.00 | 3225.67 | 3234.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 14:15:00 | 3225.40 | 3219.41 | 3228.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-06 15:00:00 | 3225.40 | 3219.41 | 3228.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 15:15:00 | 3223.75 | 3220.28 | 3228.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-07 09:15:00 | 3207.10 | 3220.28 | 3228.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 13:15:00 | 3241.60 | 3221.14 | 3225.04 | SL hit (close>static) qty=1.00 sl=3230.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-09-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 15:15:00 | 3246.00 | 3229.77 | 3228.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 11:15:00 | 3261.80 | 3241.23 | 3234.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 14:15:00 | 3238.95 | 3244.73 | 3238.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 14:15:00 | 3238.95 | 3244.73 | 3238.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 3238.95 | 3244.73 | 3238.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 15:00:00 | 3238.95 | 3244.73 | 3238.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 3236.95 | 3243.17 | 3238.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 3258.35 | 3243.17 | 3238.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 10:45:00 | 3240.00 | 3252.71 | 3251.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-14 11:15:00 | 3232.25 | 3251.41 | 3253.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-09-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 11:15:00 | 3232.25 | 3251.41 | 3253.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 09:15:00 | 3204.60 | 3235.89 | 3244.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-18 09:15:00 | 3215.95 | 3209.51 | 3223.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 09:15:00 | 3215.95 | 3209.51 | 3223.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 3215.95 | 3209.51 | 3223.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 10:15:00 | 3186.00 | 3206.59 | 3212.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 14:15:00 | 3242.10 | 3214.45 | 3213.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 14:15:00 | 3242.10 | 3214.45 | 3213.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-21 15:15:00 | 3248.00 | 3221.16 | 3216.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 09:15:00 | 3274.15 | 3301.32 | 3279.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 3274.15 | 3301.32 | 3279.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 3274.15 | 3301.32 | 3279.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:30:00 | 3262.55 | 3301.32 | 3279.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 3290.85 | 3299.23 | 3280.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-26 13:00:00 | 3293.20 | 3296.07 | 3282.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 11:45:00 | 3295.50 | 3293.57 | 3286.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 09:15:00 | 3254.20 | 3289.99 | 3288.05 | SL hit (close<static) qty=1.00 sl=3273.75 alert=retest2 |

### Cycle 29 — SELL (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 10:15:00 | 3230.25 | 3278.04 | 3282.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 11:15:00 | 3211.00 | 3264.64 | 3276.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-03 10:15:00 | 3196.20 | 3177.37 | 3200.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-03 11:00:00 | 3196.20 | 3177.37 | 3200.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 11:15:00 | 3194.05 | 3180.71 | 3200.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 12:15:00 | 3202.95 | 3180.71 | 3200.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 12:15:00 | 3201.60 | 3184.89 | 3200.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 12:45:00 | 3199.00 | 3184.89 | 3200.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 3172.30 | 3182.37 | 3197.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 13:30:00 | 3199.05 | 3182.37 | 3197.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 3173.85 | 3176.05 | 3188.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 11:30:00 | 3183.55 | 3176.05 | 3188.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 3187.55 | 3175.56 | 3182.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:30:00 | 3182.60 | 3181.72 | 3184.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 12:15:00 | 3200.35 | 3188.53 | 3187.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 12:15:00 | 3200.35 | 3188.53 | 3187.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 14:15:00 | 3207.65 | 3194.67 | 3190.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 13:15:00 | 3199.20 | 3199.40 | 3195.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-06 14:00:00 | 3199.20 | 3199.40 | 3195.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 3197.00 | 3198.71 | 3195.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 09:15:00 | 3149.95 | 3198.71 | 3195.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 3152.95 | 3189.56 | 3191.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 10:15:00 | 3132.00 | 3153.20 | 3158.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 12:15:00 | 3150.25 | 3150.23 | 3155.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-13 13:00:00 | 3150.25 | 3150.23 | 3155.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 3150.10 | 3150.10 | 3154.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 09:15:00 | 3116.20 | 3149.68 | 3154.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 13:15:00 | 2960.39 | 3027.49 | 3055.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-30 09:15:00 | 2971.00 | 2966.73 | 2996.00 | SL hit (close>ema200) qty=0.50 sl=2966.73 alert=retest2 |

### Cycle 32 — BUY (started 2023-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 15:15:00 | 2999.00 | 2991.69 | 2990.93 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 09:15:00 | 2961.35 | 2985.62 | 2988.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 10:15:00 | 2953.90 | 2979.28 | 2985.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 10:15:00 | 2956.45 | 2953.95 | 2966.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-02 11:00:00 | 2956.45 | 2953.95 | 2966.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 2962.90 | 2954.25 | 2960.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 10:15:00 | 2966.45 | 2954.25 | 2960.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 10:15:00 | 2972.10 | 2957.82 | 2961.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 11:00:00 | 2972.10 | 2957.82 | 2961.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 11:15:00 | 2975.25 | 2961.30 | 2962.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 12:00:00 | 2975.25 | 2961.30 | 2962.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 12:15:00 | 2975.05 | 2964.05 | 2963.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 3005.00 | 2977.50 | 2970.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 09:15:00 | 3008.25 | 3011.40 | 2994.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 09:45:00 | 3006.10 | 3011.40 | 2994.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 3044.90 | 3076.68 | 3065.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:00:00 | 3044.90 | 3076.68 | 3065.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 3058.00 | 3072.94 | 3064.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 11:15:00 | 3068.75 | 3072.94 | 3064.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 11:15:00 | 3133.50 | 3142.65 | 3142.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 11:15:00 | 3133.50 | 3142.65 | 3142.81 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 10:15:00 | 3153.10 | 3141.95 | 3141.60 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 3137.00 | 3141.29 | 3141.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 13:15:00 | 3121.50 | 3137.33 | 3139.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 3149.10 | 3138.07 | 3139.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 3149.10 | 3138.07 | 3139.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 3149.10 | 3138.07 | 3139.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 10:30:00 | 3140.10 | 3138.86 | 3139.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 11:30:00 | 3141.60 | 3139.65 | 3139.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 12:00:00 | 3142.80 | 3139.65 | 3139.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 13:00:00 | 3138.70 | 3139.46 | 3139.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 13:15:00 | 3132.00 | 3137.97 | 3138.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 15:00:00 | 3125.60 | 3135.49 | 3137.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 09:30:00 | 3123.00 | 3129.87 | 3134.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 10:45:00 | 3125.10 | 3127.37 | 3133.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 09:15:00 | 3145.85 | 3132.27 | 3132.80 | SL hit (close>static) qty=1.00 sl=3141.45 alert=retest2 |

### Cycle 38 — BUY (started 2023-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 10:15:00 | 3152.35 | 3136.29 | 3134.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 11:15:00 | 3156.65 | 3140.36 | 3136.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 14:15:00 | 3145.95 | 3154.03 | 3148.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 14:15:00 | 3145.95 | 3154.03 | 3148.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 3145.95 | 3154.03 | 3148.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 15:00:00 | 3145.95 | 3154.03 | 3148.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 3149.55 | 3153.13 | 3148.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:15:00 | 3126.95 | 3153.13 | 3148.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 3123.95 | 3147.29 | 3146.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 3122.25 | 3147.29 | 3146.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 10:15:00 | 3132.35 | 3144.31 | 3144.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 13:15:00 | 3118.00 | 3135.58 | 3140.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 09:15:00 | 3151.00 | 3134.89 | 3138.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 09:15:00 | 3151.00 | 3134.89 | 3138.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 3151.00 | 3134.89 | 3138.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 09:30:00 | 3166.00 | 3134.89 | 3138.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 11:15:00 | 3167.30 | 3144.20 | 3142.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 10:15:00 | 3185.30 | 3167.55 | 3156.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 13:15:00 | 3256.55 | 3259.08 | 3240.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-07 14:00:00 | 3256.55 | 3259.08 | 3240.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 3260.00 | 3257.36 | 3244.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 09:30:00 | 3248.20 | 3257.36 | 3244.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 3250.05 | 3254.96 | 3245.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:45:00 | 3243.75 | 3254.96 | 3245.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 3242.90 | 3252.55 | 3245.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 3242.90 | 3252.55 | 3245.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 3237.90 | 3249.62 | 3244.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:45:00 | 3232.95 | 3249.62 | 3244.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 3231.80 | 3246.06 | 3243.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 15:00:00 | 3231.80 | 3246.06 | 3243.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 09:15:00 | 3216.25 | 3237.37 | 3239.66 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 15:15:00 | 3237.50 | 3227.25 | 3226.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 10:15:00 | 3245.05 | 3230.97 | 3228.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 12:15:00 | 3229.75 | 3233.13 | 3229.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 12:15:00 | 3229.75 | 3233.13 | 3229.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 12:15:00 | 3229.75 | 3233.13 | 3229.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 13:00:00 | 3229.75 | 3233.13 | 3229.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 13:15:00 | 3234.20 | 3233.35 | 3230.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 15:15:00 | 3245.75 | 3234.82 | 3231.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-21 09:15:00 | 3289.65 | 3311.26 | 3312.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 09:15:00 | 3289.65 | 3311.26 | 3312.35 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 3321.00 | 3311.41 | 3311.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 11:15:00 | 3337.05 | 3316.53 | 3313.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 09:15:00 | 3357.30 | 3364.58 | 3348.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 10:00:00 | 3357.30 | 3364.58 | 3348.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 13:15:00 | 3379.45 | 3385.69 | 3373.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 14:00:00 | 3379.45 | 3385.69 | 3373.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 12:15:00 | 3388.60 | 3395.17 | 3384.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 13:00:00 | 3388.60 | 3395.17 | 3384.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 3390.25 | 3395.83 | 3390.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:00:00 | 3390.25 | 3395.83 | 3390.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 3396.50 | 3395.96 | 3390.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 14:30:00 | 3398.55 | 3395.41 | 3391.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 15:15:00 | 3390.00 | 3394.33 | 3391.02 | SL hit (close<static) qty=1.00 sl=3390.70 alert=retest2 |

### Cycle 45 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 3362.00 | 3387.86 | 3388.38 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 11:15:00 | 3392.90 | 3386.91 | 3386.71 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 12:15:00 | 3385.00 | 3386.53 | 3386.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 14:15:00 | 3371.40 | 3383.11 | 3384.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 09:15:00 | 3390.65 | 3383.31 | 3384.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 3390.65 | 3383.31 | 3384.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 3390.65 | 3383.31 | 3384.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:45:00 | 3392.50 | 3383.31 | 3384.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 3394.00 | 3385.45 | 3385.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:00:00 | 3394.00 | 3385.45 | 3385.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 3383.15 | 3384.99 | 3385.29 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-01-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 13:15:00 | 3391.15 | 3386.52 | 3385.96 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 14:15:00 | 3375.95 | 3384.41 | 3385.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 09:15:00 | 3372.50 | 3380.58 | 3383.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 14:15:00 | 3284.75 | 3279.81 | 3297.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-10 15:00:00 | 3284.75 | 3279.81 | 3297.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 3288.60 | 3282.40 | 3295.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 10:30:00 | 3281.50 | 3281.25 | 3293.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 12:00:00 | 3279.95 | 3280.99 | 3292.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 12:30:00 | 3282.35 | 3280.82 | 3291.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 13:00:00 | 3280.15 | 3280.82 | 3291.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 3285.00 | 3282.94 | 3289.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:15:00 | 3265.40 | 3282.94 | 3289.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 3249.85 | 3276.33 | 3286.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-16 10:15:00 | 3300.40 | 3281.22 | 3279.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 10:15:00 | 3300.40 | 3281.22 | 3279.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 12:15:00 | 3308.50 | 3289.46 | 3283.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 3283.05 | 3291.63 | 3286.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 3283.05 | 3291.63 | 3286.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 3283.05 | 3291.63 | 3286.71 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 12:15:00 | 3267.60 | 3281.59 | 3282.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 14:15:00 | 3248.00 | 3272.78 | 3278.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 3186.85 | 3186.74 | 3218.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 10:45:00 | 3163.80 | 3180.58 | 3212.52 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 12:45:00 | 3169.75 | 3176.90 | 3205.12 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 14:00:00 | 3163.75 | 3174.27 | 3201.36 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:45:00 | 3159.20 | 3170.14 | 3190.55 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-24 09:15:00 | 3005.61 | 3063.04 | 3107.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-24 09:15:00 | 3011.26 | 3063.04 | 3107.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-24 09:15:00 | 3005.56 | 3063.04 | 3107.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-24 09:15:00 | 3001.24 | 3063.04 | 3107.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 2974.00 | 2969.55 | 3007.41 | SL hit (close>ema200) qty=0.50 sl=2969.55 alert=retest1 |

### Cycle 52 — BUY (started 2024-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 10:15:00 | 2947.55 | 2937.32 | 2937.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 12:15:00 | 2965.10 | 2945.16 | 2941.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 2961.50 | 2961.97 | 2951.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-08 09:30:00 | 2964.00 | 2961.97 | 2951.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 2934.80 | 2956.54 | 2950.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 2934.80 | 2956.54 | 2950.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 2941.50 | 2953.53 | 2949.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 12:15:00 | 2945.15 | 2953.53 | 2949.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 13:00:00 | 2947.90 | 2952.40 | 2949.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 13:15:00 | 2922.95 | 2946.51 | 2946.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 13:15:00 | 2922.95 | 2946.51 | 2946.74 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 15:15:00 | 2955.95 | 2944.99 | 2944.12 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 2930.00 | 2941.99 | 2942.84 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 13:15:00 | 2954.35 | 2943.55 | 2943.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 15:15:00 | 2959.00 | 2947.91 | 2945.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 09:15:00 | 2943.85 | 2961.10 | 2955.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 09:15:00 | 2943.85 | 2961.10 | 2955.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 2943.85 | 2961.10 | 2955.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 14:45:00 | 2973.45 | 2958.64 | 2955.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-20 10:15:00 | 2988.20 | 2995.77 | 2996.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 2988.20 | 2995.77 | 2996.23 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 12:15:00 | 3007.60 | 2998.62 | 2997.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-21 11:15:00 | 3023.20 | 3010.29 | 3004.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 3010.40 | 3011.58 | 3006.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 13:15:00 | 3010.40 | 3011.58 | 3006.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 3010.40 | 3011.58 | 3006.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 3010.40 | 3011.58 | 3006.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 3007.30 | 3010.72 | 3006.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:45:00 | 2995.05 | 3010.72 | 3006.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 2990.00 | 3006.58 | 3004.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 2951.85 | 3006.58 | 3004.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 2953.00 | 2995.86 | 2999.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 09:15:00 | 2868.15 | 2961.45 | 2978.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 2811.55 | 2808.41 | 2841.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 11:00:00 | 2811.55 | 2808.41 | 2841.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 2834.70 | 2817.72 | 2833.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 09:15:00 | 2820.05 | 2817.72 | 2833.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-02 09:15:00 | 2853.00 | 2834.63 | 2835.07 | SL hit (close>static) qty=1.00 sl=2838.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 11:15:00 | 2851.00 | 2837.90 | 2836.52 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 14:15:00 | 2833.60 | 2836.65 | 2836.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 2816.15 | 2831.47 | 2834.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 2825.00 | 2816.72 | 2821.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 14:15:00 | 2825.00 | 2816.72 | 2821.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 2825.00 | 2816.72 | 2821.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 2825.00 | 2816.72 | 2821.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 2834.00 | 2820.17 | 2823.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 2857.65 | 2820.17 | 2823.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 09:15:00 | 2855.30 | 2827.20 | 2826.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 11:15:00 | 2878.95 | 2843.90 | 2834.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 10:15:00 | 2861.90 | 2863.67 | 2850.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-11 11:00:00 | 2861.90 | 2863.67 | 2850.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 15:15:00 | 2878.00 | 2879.70 | 2870.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 09:15:00 | 2874.50 | 2879.70 | 2870.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 2871.40 | 2878.04 | 2870.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 09:45:00 | 2871.95 | 2878.04 | 2870.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 2870.05 | 2876.44 | 2870.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 11:00:00 | 2870.05 | 2876.44 | 2870.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 2854.85 | 2872.12 | 2869.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 11:30:00 | 2853.80 | 2872.12 | 2869.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 12:15:00 | 2857.20 | 2869.14 | 2867.94 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 13:15:00 | 2847.80 | 2864.87 | 2866.11 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-03-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 10:15:00 | 2880.90 | 2868.13 | 2866.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 14:15:00 | 2891.05 | 2879.08 | 2873.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 09:15:00 | 2875.40 | 2881.37 | 2875.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 09:15:00 | 2875.40 | 2881.37 | 2875.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 2875.40 | 2881.37 | 2875.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:30:00 | 2870.80 | 2881.37 | 2875.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 2882.85 | 2881.67 | 2876.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 11:30:00 | 2885.90 | 2881.46 | 2876.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 12:30:00 | 2889.55 | 2883.39 | 2877.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-15 14:15:00 | 2862.65 | 2879.08 | 2876.81 | SL hit (close<static) qty=1.00 sl=2872.15 alert=retest2 |

### Cycle 65 — SELL (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 15:15:00 | 2859.30 | 2875.12 | 2875.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 09:15:00 | 2827.65 | 2865.63 | 2870.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 2820.25 | 2817.87 | 2829.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 13:45:00 | 2810.20 | 2816.71 | 2827.91 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 15:15:00 | 2835.00 | 2821.88 | 2828.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-20 15:15:00 | 2835.00 | 2821.88 | 2828.41 | SL hit (close>ema400) qty=1.00 sl=2828.41 alert=retest1 |

### Cycle 66 — BUY (started 2024-03-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 14:15:00 | 2835.10 | 2829.51 | 2828.96 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 11:15:00 | 2820.70 | 2828.32 | 2828.84 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 13:15:00 | 2841.50 | 2829.72 | 2828.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 11:15:00 | 2843.80 | 2833.93 | 2830.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 14:15:00 | 2837.40 | 2840.41 | 2835.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-28 15:00:00 | 2837.40 | 2840.41 | 2835.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 2856.65 | 2843.66 | 2837.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 2877.75 | 2843.66 | 2837.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 11:45:00 | 2866.00 | 2852.66 | 2843.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 12:30:00 | 2867.15 | 2857.25 | 2846.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 14:15:00 | 2869.55 | 2858.76 | 2848.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 2858.00 | 2869.34 | 2862.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 11:00:00 | 2863.30 | 2868.13 | 2862.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 11:00:00 | 2865.95 | 2866.25 | 2864.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 11:15:00 | 2875.00 | 2888.87 | 2889.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 11:15:00 | 2875.00 | 2888.87 | 2889.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 2868.60 | 2884.81 | 2887.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 10:15:00 | 2869.25 | 2869.11 | 2877.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-10 10:30:00 | 2869.50 | 2869.11 | 2877.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 2889.35 | 2873.16 | 2878.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:00:00 | 2889.35 | 2873.16 | 2878.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 2886.00 | 2875.73 | 2878.85 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 14:15:00 | 2897.40 | 2883.04 | 2881.82 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 09:15:00 | 2852.40 | 2878.90 | 2880.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 2836.35 | 2857.78 | 2867.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 10:15:00 | 2850.50 | 2837.15 | 2844.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 10:15:00 | 2850.50 | 2837.15 | 2844.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 2850.50 | 2837.15 | 2844.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:00:00 | 2850.50 | 2837.15 | 2844.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 2854.30 | 2840.58 | 2844.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:30:00 | 2854.60 | 2840.58 | 2844.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 2821.80 | 2838.70 | 2843.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 15:00:00 | 2805.05 | 2831.97 | 2839.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 14:30:00 | 2802.80 | 2808.23 | 2820.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 12:15:00 | 2839.20 | 2824.78 | 2824.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 2839.20 | 2824.78 | 2824.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 2866.00 | 2839.30 | 2831.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 13:15:00 | 2866.50 | 2866.88 | 2856.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 2841.20 | 2862.69 | 2856.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 2841.20 | 2862.69 | 2856.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:45:00 | 2837.65 | 2862.69 | 2856.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 2840.30 | 2858.22 | 2855.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 2838.80 | 2858.22 | 2855.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 2861.40 | 2858.61 | 2856.47 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 2844.20 | 2854.81 | 2855.59 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 09:15:00 | 2867.50 | 2856.60 | 2856.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 09:15:00 | 2894.50 | 2869.27 | 2863.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 2872.65 | 2882.52 | 2873.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 14:15:00 | 2872.65 | 2882.52 | 2873.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 2872.65 | 2882.52 | 2873.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 2872.65 | 2882.52 | 2873.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 2875.00 | 2881.02 | 2873.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 2919.00 | 2881.02 | 2873.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 13:15:00 | 2918.40 | 2926.28 | 2926.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 13:15:00 | 2918.40 | 2926.28 | 2926.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 14:15:00 | 2913.10 | 2923.64 | 2925.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 13:15:00 | 2779.00 | 2776.56 | 2809.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 14:00:00 | 2779.00 | 2776.56 | 2809.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 2838.80 | 2788.71 | 2804.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 11:00:00 | 2838.80 | 2788.71 | 2804.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 11:15:00 | 2866.70 | 2804.31 | 2809.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:00:00 | 2866.70 | 2804.31 | 2809.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 2871.65 | 2817.78 | 2815.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 2882.45 | 2830.71 | 2821.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 09:15:00 | 2847.20 | 2847.75 | 2832.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 09:30:00 | 2855.10 | 2847.75 | 2832.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 2843.10 | 2861.01 | 2849.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 2843.10 | 2861.01 | 2849.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 2838.95 | 2856.60 | 2848.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:45:00 | 2834.45 | 2856.60 | 2848.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 13:15:00 | 2827.00 | 2843.93 | 2844.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 14:15:00 | 2815.40 | 2838.23 | 2841.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 2810.80 | 2809.21 | 2822.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 2810.80 | 2809.21 | 2822.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 2810.80 | 2809.21 | 2822.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 2810.80 | 2809.21 | 2822.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 2794.95 | 2807.22 | 2819.00 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 09:15:00 | 2847.15 | 2821.43 | 2819.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 11:15:00 | 2854.55 | 2832.71 | 2825.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 2883.25 | 2890.35 | 2876.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 10:00:00 | 2883.25 | 2890.35 | 2876.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 2873.40 | 2886.15 | 2878.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:00:00 | 2873.40 | 2886.15 | 2878.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 2879.05 | 2884.73 | 2878.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:15:00 | 2868.00 | 2884.73 | 2878.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 2874.20 | 2882.63 | 2877.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:30:00 | 2885.00 | 2877.67 | 2876.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:30:00 | 2883.30 | 2880.33 | 2877.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 10:30:00 | 2883.00 | 2878.22 | 2877.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 11:30:00 | 2887.05 | 2880.67 | 2878.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 2900.35 | 2897.04 | 2888.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:00:00 | 2916.45 | 2903.58 | 2893.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 14:15:00 | 2881.40 | 2894.13 | 2894.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 2881.40 | 2894.13 | 2894.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 12:15:00 | 2873.00 | 2883.49 | 2887.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 09:15:00 | 2889.35 | 2879.99 | 2883.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 2889.35 | 2879.99 | 2883.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 2889.35 | 2879.99 | 2883.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:45:00 | 2887.25 | 2879.99 | 2883.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 2842.20 | 2872.43 | 2880.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 12:15:00 | 2791.50 | 2868.34 | 2877.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 2991.90 | 2891.69 | 2884.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 2991.90 | 2891.69 | 2884.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 2997.30 | 2912.81 | 2894.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 2930.00 | 2942.59 | 2921.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 10:00:00 | 2930.00 | 2942.59 | 2921.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 2931.85 | 2940.44 | 2922.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 2926.00 | 2940.44 | 2922.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 2928.40 | 2938.03 | 2922.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:45:00 | 2929.10 | 2938.03 | 2922.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 2916.00 | 2933.63 | 2922.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:00:00 | 2916.00 | 2933.63 | 2922.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 2916.80 | 2930.26 | 2921.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:45:00 | 2913.30 | 2930.26 | 2921.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 2903.40 | 2924.89 | 2920.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 2895.40 | 2924.89 | 2920.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 2934.35 | 2923.73 | 2920.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 11:45:00 | 2936.25 | 2925.88 | 2921.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 2938.10 | 2925.92 | 2922.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 13:00:00 | 2936.45 | 2928.61 | 2925.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 14:30:00 | 2939.95 | 2933.22 | 2927.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 2910.05 | 2929.67 | 2927.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-11 10:15:00 | 2910.00 | 2925.74 | 2925.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 10:15:00 | 2910.00 | 2925.74 | 2925.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 14:15:00 | 2902.50 | 2914.98 | 2920.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 12:15:00 | 2904.70 | 2899.23 | 2909.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 12:15:00 | 2904.70 | 2899.23 | 2909.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 2904.70 | 2899.23 | 2909.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:30:00 | 2910.00 | 2899.23 | 2909.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 2901.15 | 2899.62 | 2908.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:30:00 | 2906.30 | 2899.62 | 2908.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 2907.35 | 2901.16 | 2908.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 2907.35 | 2901.16 | 2908.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 2904.15 | 2901.76 | 2907.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:15:00 | 2911.15 | 2901.76 | 2907.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 2913.00 | 2904.01 | 2908.37 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 15:15:00 | 2913.00 | 2910.30 | 2910.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 2921.40 | 2912.52 | 2911.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 2912.90 | 2917.41 | 2915.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 2912.90 | 2917.41 | 2915.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 2912.90 | 2917.41 | 2915.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 2915.60 | 2917.41 | 2915.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 2915.80 | 2917.09 | 2915.22 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 2889.00 | 2912.02 | 2913.75 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 2912.85 | 2907.24 | 2907.03 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 2896.95 | 2906.39 | 2906.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 2890.00 | 2903.11 | 2905.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 13:15:00 | 2906.20 | 2902.59 | 2904.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 13:15:00 | 2906.20 | 2902.59 | 2904.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 2906.20 | 2902.59 | 2904.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:00:00 | 2906.20 | 2902.59 | 2904.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 2886.30 | 2899.33 | 2902.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:30:00 | 2878.75 | 2892.90 | 2899.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 12:45:00 | 2879.90 | 2890.90 | 2896.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:45:00 | 2880.75 | 2890.56 | 2894.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:00:00 | 2877.60 | 2887.97 | 2893.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 2887.15 | 2868.15 | 2872.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 2887.15 | 2868.15 | 2872.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 2885.70 | 2871.66 | 2873.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:45:00 | 2889.45 | 2871.66 | 2873.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 2876.45 | 2872.56 | 2873.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:45:00 | 2878.50 | 2872.56 | 2873.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-27 15:15:00 | 2885.80 | 2875.21 | 2875.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 2885.80 | 2875.21 | 2875.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 2912.00 | 2882.57 | 2878.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 2920.00 | 2920.87 | 2910.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 09:30:00 | 2915.75 | 2920.87 | 2910.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 2927.85 | 2933.09 | 2925.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:00:00 | 2927.85 | 2933.09 | 2925.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 2924.50 | 2931.37 | 2925.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:30:00 | 2919.95 | 2931.37 | 2925.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 2929.90 | 2931.08 | 2925.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 2928.55 | 2931.08 | 2925.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 2942.55 | 2933.37 | 2927.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:30:00 | 2965.05 | 2937.64 | 2929.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 2898.75 | 2928.77 | 2931.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 2898.75 | 2928.77 | 2931.30 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 2930.50 | 2916.61 | 2915.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 13:15:00 | 2995.00 | 2932.29 | 2922.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 09:15:00 | 2985.00 | 3000.46 | 2975.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 10:00:00 | 2985.00 | 3000.46 | 2975.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 2990.25 | 2998.42 | 2976.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:30:00 | 2982.00 | 2998.42 | 2976.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 2991.40 | 3002.98 | 2988.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 2964.10 | 3002.98 | 2988.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2963.25 | 2995.04 | 2985.79 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 12:15:00 | 2958.00 | 2979.54 | 2980.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 13:15:00 | 2951.65 | 2973.96 | 2977.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 10:15:00 | 2978.00 | 2968.73 | 2973.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 10:15:00 | 2978.00 | 2968.73 | 2973.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 2978.00 | 2968.73 | 2973.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:45:00 | 2976.80 | 2968.73 | 2973.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 2998.65 | 2974.71 | 2975.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 2998.65 | 2974.71 | 2975.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 12:15:00 | 3005.95 | 2980.96 | 2978.39 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 2943.00 | 2970.82 | 2974.37 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 2943.80 | 2917.49 | 2916.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 2959.80 | 2929.47 | 2922.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 3100.05 | 3107.83 | 3089.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 3100.05 | 3107.83 | 3089.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 3110.20 | 3108.30 | 3091.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:45:00 | 3123.85 | 3105.83 | 3095.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 10:45:00 | 3125.00 | 3108.40 | 3097.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 3089.90 | 3097.33 | 3098.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 11:15:00 | 3089.90 | 3097.33 | 3098.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 3063.70 | 3090.55 | 3094.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 14:15:00 | 3041.50 | 3015.44 | 3037.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 14:15:00 | 3041.50 | 3015.44 | 3037.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 3041.50 | 3015.44 | 3037.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:45:00 | 3033.25 | 3015.44 | 3037.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 3040.00 | 3020.35 | 3037.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 3059.05 | 3020.35 | 3037.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 3016.85 | 3019.65 | 3035.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:30:00 | 3062.00 | 3019.65 | 3035.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 3044.85 | 3024.69 | 3036.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:00:00 | 3044.85 | 3024.69 | 3036.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 3054.30 | 3030.61 | 3037.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:00:00 | 3054.30 | 3030.61 | 3037.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 3056.20 | 3040.16 | 3040.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 3056.20 | 3040.16 | 3040.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 15:15:00 | 3050.00 | 3042.13 | 3041.68 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 3034.85 | 3040.45 | 3040.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 3021.95 | 3034.54 | 3037.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 13:15:00 | 3026.00 | 3021.43 | 3028.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 13:15:00 | 3026.00 | 3021.43 | 3028.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 3026.00 | 3021.43 | 3028.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:00:00 | 3026.00 | 3021.43 | 3028.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 3025.50 | 3022.25 | 3027.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:45:00 | 3027.10 | 3022.25 | 3027.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 3027.00 | 3023.20 | 3027.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 3036.00 | 3023.20 | 3027.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 3015.00 | 3021.56 | 3026.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 3008.15 | 3021.56 | 3026.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 14:15:00 | 3048.55 | 3031.65 | 3029.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 3048.55 | 3031.65 | 3029.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 3059.00 | 3040.05 | 3034.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 3170.65 | 3173.46 | 3147.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 14:15:00 | 3154.20 | 3163.88 | 3152.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 3154.20 | 3163.88 | 3152.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 3154.20 | 3163.88 | 3152.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 3155.00 | 3162.11 | 3152.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 3151.25 | 3162.11 | 3152.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 3164.40 | 3162.57 | 3153.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:30:00 | 3170.80 | 3164.26 | 3156.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 3171.00 | 3168.73 | 3161.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 10:15:00 | 3137.10 | 3158.96 | 3161.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 3137.10 | 3158.96 | 3161.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 3124.00 | 3139.30 | 3149.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 3140.35 | 3139.51 | 3148.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 10:00:00 | 3140.35 | 3139.51 | 3148.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 3146.15 | 3125.91 | 3135.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 3162.85 | 3125.91 | 3135.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 3146.00 | 3129.93 | 3136.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:30:00 | 3143.45 | 3129.93 | 3136.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 3141.00 | 3134.49 | 3137.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:30:00 | 3135.00 | 3133.93 | 3136.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 09:15:00 | 3156.30 | 3137.89 | 3137.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 3156.30 | 3137.89 | 3137.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 3231.75 | 3170.38 | 3158.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 11:15:00 | 3262.50 | 3267.49 | 3248.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-09 12:00:00 | 3262.50 | 3267.49 | 3248.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 3338.00 | 3361.73 | 3344.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:30:00 | 3337.75 | 3361.73 | 3344.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 3349.00 | 3359.18 | 3344.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:30:00 | 3353.65 | 3358.54 | 3345.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 12:30:00 | 3356.95 | 3358.10 | 3346.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:30:00 | 3355.30 | 3358.30 | 3347.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 10:00:00 | 3360.30 | 3358.59 | 3350.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 3344.00 | 3354.87 | 3350.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 3344.00 | 3354.87 | 3350.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 3335.85 | 3351.06 | 3348.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:00:00 | 3335.85 | 3351.06 | 3348.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-16 14:15:00 | 3338.20 | 3347.31 | 3347.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 14:15:00 | 3338.20 | 3347.31 | 3347.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 3334.00 | 3344.64 | 3346.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 3305.70 | 3289.24 | 3305.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 3305.70 | 3289.24 | 3305.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 3305.70 | 3289.24 | 3305.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 3305.70 | 3289.24 | 3305.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 3318.00 | 3294.99 | 3306.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:30:00 | 3318.00 | 3294.99 | 3306.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 3308.30 | 3297.65 | 3306.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:15:00 | 3294.50 | 3297.65 | 3306.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 14:45:00 | 3297.05 | 3293.77 | 3302.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 3278.30 | 3295.19 | 3302.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:00:00 | 3298.25 | 3295.81 | 3301.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 3307.85 | 3298.21 | 3302.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:30:00 | 3310.95 | 3298.21 | 3302.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 3306.80 | 3299.93 | 3302.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 3286.15 | 3304.51 | 3304.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 3305.45 | 3304.70 | 3304.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 3305.45 | 3304.70 | 3304.64 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 10:15:00 | 3289.00 | 3302.72 | 3303.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 11:15:00 | 3284.35 | 3299.05 | 3302.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 09:15:00 | 3291.30 | 3288.86 | 3295.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 3291.30 | 3288.86 | 3295.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 3291.30 | 3288.86 | 3295.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:45:00 | 3290.95 | 3288.86 | 3295.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 3281.30 | 3287.35 | 3293.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:15:00 | 3270.00 | 3287.35 | 3293.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 15:00:00 | 3276.85 | 3251.74 | 3254.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 3311.90 | 3267.75 | 3261.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 3311.90 | 3267.75 | 3261.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 3327.00 | 3279.60 | 3267.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 3320.00 | 3327.78 | 3310.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:15:00 | 3284.00 | 3327.78 | 3310.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 3266.40 | 3315.51 | 3306.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 3266.40 | 3315.51 | 3306.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 3252.95 | 3303.00 | 3301.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:45:00 | 3258.20 | 3303.00 | 3301.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 11:15:00 | 3273.25 | 3297.05 | 3299.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 3188.70 | 3263.31 | 3280.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 3080.35 | 3073.39 | 3106.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 3080.35 | 3073.39 | 3106.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 3112.20 | 3090.24 | 3101.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:15:00 | 3094.85 | 3096.56 | 3101.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 3087.65 | 3097.65 | 3101.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 11:15:00 | 3084.95 | 3058.91 | 3057.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 3084.95 | 3058.91 | 3057.29 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 3040.05 | 3067.19 | 3067.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 3033.55 | 3048.50 | 3055.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 3028.50 | 3025.03 | 3040.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 10:00:00 | 3028.50 | 3025.03 | 3040.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 3042.35 | 3027.89 | 3036.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 3042.35 | 3027.89 | 3036.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 3050.05 | 3032.32 | 3038.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 3034.40 | 3034.45 | 3038.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 3015.55 | 2991.96 | 2991.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 3015.55 | 2991.96 | 2991.61 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 2967.95 | 2993.21 | 2993.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 2956.75 | 2985.92 | 2990.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 12:15:00 | 2985.85 | 2984.33 | 2988.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 13:00:00 | 2985.85 | 2984.33 | 2988.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 2988.05 | 2985.07 | 2988.45 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 2995.15 | 2989.58 | 2989.51 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 2980.45 | 2987.76 | 2988.69 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 2999.30 | 2990.07 | 2989.65 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 2976.10 | 2988.27 | 2988.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 2948.80 | 2975.41 | 2982.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 2894.25 | 2888.72 | 2904.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 13:15:00 | 2894.25 | 2888.72 | 2904.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 2894.25 | 2888.72 | 2904.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:30:00 | 2896.00 | 2888.72 | 2904.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 2893.45 | 2889.67 | 2903.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 2893.45 | 2889.67 | 2903.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 2871.15 | 2888.10 | 2900.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 12:45:00 | 2845.90 | 2869.58 | 2888.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-11 09:15:00 | 2561.31 | 2744.05 | 2804.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 2482.85 | 2467.44 | 2467.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 10:15:00 | 2512.00 | 2481.74 | 2475.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 2494.95 | 2498.14 | 2488.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:00:00 | 2494.95 | 2498.14 | 2488.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 2484.30 | 2495.38 | 2487.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 2487.30 | 2495.38 | 2487.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 2482.20 | 2492.74 | 2487.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 2482.50 | 2492.74 | 2487.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 2490.65 | 2492.32 | 2487.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:30:00 | 2497.45 | 2492.09 | 2488.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 2471.80 | 2487.76 | 2487.16 | SL hit (close<static) qty=1.00 sl=2481.40 alert=retest2 |

### Cycle 113 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 2465.40 | 2483.28 | 2485.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 2459.90 | 2478.61 | 2482.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 2478.50 | 2471.48 | 2476.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 2478.50 | 2471.48 | 2476.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 2478.50 | 2471.48 | 2476.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 2478.50 | 2471.48 | 2476.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 2490.00 | 2475.19 | 2478.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 2490.00 | 2475.19 | 2478.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 2487.25 | 2477.60 | 2478.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:00:00 | 2487.25 | 2477.60 | 2478.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 2480.05 | 2478.54 | 2479.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:30:00 | 2484.90 | 2478.54 | 2479.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 2478.00 | 2478.43 | 2479.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 2466.80 | 2478.43 | 2479.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2470.25 | 2476.79 | 2478.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 11:45:00 | 2459.40 | 2469.16 | 2472.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 14:15:00 | 2458.85 | 2467.74 | 2471.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 2435.80 | 2450.76 | 2458.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 14:15:00 | 2418.10 | 2411.72 | 2411.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 2418.10 | 2411.72 | 2411.36 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 2396.70 | 2409.72 | 2410.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 2395.45 | 2406.87 | 2409.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 2391.80 | 2388.14 | 2396.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 2391.80 | 2388.14 | 2396.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 2391.80 | 2388.14 | 2396.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 2391.80 | 2388.14 | 2396.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 2392.35 | 2388.99 | 2396.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 2392.35 | 2388.99 | 2396.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 2399.50 | 2391.09 | 2396.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 2399.50 | 2391.09 | 2396.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 2406.35 | 2394.14 | 2397.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 2406.35 | 2394.14 | 2397.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 2396.95 | 2394.58 | 2396.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 2396.95 | 2394.58 | 2396.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 2397.00 | 2395.06 | 2396.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 2393.00 | 2396.70 | 2397.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 2273.35 | 2341.53 | 2361.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 2315.50 | 2304.77 | 2325.83 | SL hit (close>ema200) qty=0.50 sl=2304.77 alert=retest2 |

### Cycle 116 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 2291.95 | 2279.10 | 2277.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 2296.85 | 2285.45 | 2281.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 2301.20 | 2303.20 | 2295.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 2301.20 | 2303.20 | 2295.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 2301.20 | 2303.20 | 2295.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:30:00 | 2301.15 | 2303.20 | 2295.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2310.40 | 2330.50 | 2323.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2301.80 | 2330.50 | 2323.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 2289.00 | 2322.20 | 2320.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 2289.00 | 2322.20 | 2320.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 2276.90 | 2313.14 | 2316.25 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 13:15:00 | 2330.85 | 2301.08 | 2300.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 14:15:00 | 2336.05 | 2308.07 | 2303.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 2323.40 | 2338.39 | 2326.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 2323.40 | 2338.39 | 2326.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 2323.40 | 2338.39 | 2326.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 2323.40 | 2338.39 | 2326.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 2334.30 | 2337.57 | 2327.64 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 2290.10 | 2319.87 | 2322.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 2275.25 | 2304.85 | 2314.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 2244.65 | 2224.91 | 2235.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 2244.65 | 2224.91 | 2235.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 2244.65 | 2224.91 | 2235.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 2244.65 | 2224.91 | 2235.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 2245.40 | 2229.01 | 2236.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:45:00 | 2244.15 | 2229.01 | 2236.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 13:15:00 | 2259.65 | 2243.69 | 2242.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 2265.10 | 2252.27 | 2246.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 13:15:00 | 2280.45 | 2282.95 | 2271.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 13:15:00 | 2280.45 | 2282.95 | 2271.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 2280.45 | 2282.95 | 2271.59 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 2256.95 | 2265.85 | 2266.33 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 2288.55 | 2266.61 | 2265.78 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 2263.50 | 2270.85 | 2271.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 2251.15 | 2264.83 | 2268.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 10:15:00 | 2257.90 | 2257.33 | 2263.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 11:00:00 | 2257.90 | 2257.33 | 2263.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 2258.00 | 2257.46 | 2262.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:30:00 | 2263.50 | 2257.46 | 2262.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 2260.40 | 2258.05 | 2262.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:45:00 | 2260.60 | 2258.05 | 2262.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 2252.90 | 2257.02 | 2261.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 2241.90 | 2254.25 | 2259.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 2249.05 | 2240.30 | 2243.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 15:15:00 | 2255.00 | 2245.13 | 2244.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 15:15:00 | 2255.00 | 2245.13 | 2244.99 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 09:15:00 | 2236.75 | 2243.46 | 2244.24 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 2261.35 | 2247.03 | 2245.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 2280.25 | 2253.68 | 2248.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 10:15:00 | 2327.00 | 2329.76 | 2305.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 11:15:00 | 2302.60 | 2324.33 | 2305.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 2302.60 | 2324.33 | 2305.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:00:00 | 2302.60 | 2324.33 | 2305.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 2295.05 | 2318.48 | 2304.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 2295.05 | 2318.48 | 2304.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 2293.40 | 2297.58 | 2298.12 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 2333.00 | 2304.63 | 2301.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 13:15:00 | 2396.00 | 2322.90 | 2309.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 09:15:00 | 2262.65 | 2319.62 | 2312.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 2262.65 | 2319.62 | 2312.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 2262.65 | 2319.62 | 2312.45 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 11:15:00 | 2276.05 | 2302.55 | 2305.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 2261.90 | 2277.55 | 2288.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 13:15:00 | 2269.75 | 2265.74 | 2275.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 14:00:00 | 2269.75 | 2265.74 | 2275.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 2273.80 | 2268.35 | 2275.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 2270.95 | 2268.35 | 2275.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 2262.90 | 2267.26 | 2273.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 2262.90 | 2267.26 | 2273.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 2266.50 | 2267.11 | 2273.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:30:00 | 2271.50 | 2267.11 | 2273.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 2270.60 | 2266.22 | 2270.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 2270.60 | 2266.22 | 2270.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 2270.00 | 2266.97 | 2270.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 2268.10 | 2266.97 | 2270.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 2253.45 | 2264.27 | 2269.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:30:00 | 2244.30 | 2259.53 | 2266.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:45:00 | 2247.75 | 2253.86 | 2262.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:45:00 | 2247.75 | 2230.97 | 2232.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 2254.70 | 2237.68 | 2235.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 12:15:00 | 2254.70 | 2237.68 | 2235.40 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 12:15:00 | 2228.85 | 2235.06 | 2235.88 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 2245.10 | 2236.94 | 2236.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 2247.65 | 2240.90 | 2239.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 2238.70 | 2246.00 | 2243.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 2238.70 | 2246.00 | 2243.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 2238.70 | 2246.00 | 2243.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 2238.70 | 2246.00 | 2243.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 2247.65 | 2246.33 | 2243.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 2254.15 | 2249.48 | 2245.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 13:15:00 | 2250.90 | 2249.16 | 2245.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:15:00 | 2251.00 | 2248.73 | 2245.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:30:00 | 2256.70 | 2252.06 | 2248.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 2246.50 | 2250.95 | 2248.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:00:00 | 2246.50 | 2250.95 | 2248.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 2249.10 | 2250.58 | 2248.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:30:00 | 2246.00 | 2250.58 | 2248.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 2251.90 | 2250.84 | 2248.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 2247.10 | 2250.84 | 2248.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 2246.95 | 2250.06 | 2248.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 2246.95 | 2250.06 | 2248.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 2243.50 | 2248.75 | 2248.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 2248.10 | 2248.75 | 2248.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 2249.50 | 2249.74 | 2248.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:00:00 | 2249.50 | 2249.74 | 2248.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 2248.80 | 2249.55 | 2248.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:30:00 | 2248.95 | 2249.55 | 2248.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 2247.70 | 2249.18 | 2248.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:30:00 | 2246.25 | 2249.18 | 2248.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-25 13:15:00 | 2233.10 | 2245.97 | 2247.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 2233.10 | 2245.97 | 2247.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 2226.10 | 2241.99 | 2245.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 15:15:00 | 2180.40 | 2165.72 | 2181.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 15:15:00 | 2180.40 | 2165.72 | 2181.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 2180.40 | 2165.72 | 2181.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 2148.65 | 2165.72 | 2181.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 2217.30 | 2172.88 | 2168.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 2217.30 | 2172.88 | 2168.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 2244.00 | 2187.10 | 2174.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 15:15:00 | 2273.60 | 2276.81 | 2257.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-11 09:15:00 | 2271.55 | 2276.81 | 2257.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 2282.60 | 2277.96 | 2260.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:30:00 | 2287.55 | 2279.61 | 2262.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:00:00 | 2286.95 | 2281.84 | 2267.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 15:00:00 | 2288.80 | 2283.24 | 2269.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 12:15:00 | 2256.20 | 2270.35 | 2268.11 | SL hit (close<static) qty=1.00 sl=2257.90 alert=retest2 |

### Cycle 135 — SELL (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 14:15:00 | 2252.55 | 2264.28 | 2265.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 15:15:00 | 2250.00 | 2261.43 | 2264.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 15:15:00 | 2222.00 | 2221.02 | 2233.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:15:00 | 2225.80 | 2221.02 | 2233.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 2244.90 | 2225.80 | 2234.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 2244.90 | 2225.80 | 2234.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 2273.20 | 2235.28 | 2237.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:00:00 | 2273.20 | 2235.28 | 2237.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 2279.00 | 2244.02 | 2241.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 11:15:00 | 2290.75 | 2281.68 | 2270.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 13:15:00 | 2281.45 | 2283.54 | 2273.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 13:45:00 | 2279.40 | 2283.54 | 2273.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 2320.60 | 2322.12 | 2313.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 2319.25 | 2322.12 | 2313.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 2307.10 | 2319.26 | 2313.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 2307.10 | 2319.26 | 2313.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 2311.25 | 2317.66 | 2313.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 14:15:00 | 2325.00 | 2318.28 | 2314.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 15:00:00 | 2325.15 | 2319.66 | 2315.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:30:00 | 2325.90 | 2318.51 | 2315.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 11:15:00 | 2323.00 | 2318.51 | 2315.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 2335.10 | 2338.51 | 2330.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:45:00 | 2331.15 | 2338.51 | 2330.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 2326.25 | 2336.69 | 2331.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 2345.00 | 2336.69 | 2331.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 2345.25 | 2338.40 | 2332.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-01 13:15:00 | 2310.25 | 2329.01 | 2329.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 2310.25 | 2329.01 | 2329.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 2301.55 | 2319.93 | 2325.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 15:15:00 | 2308.95 | 2308.52 | 2316.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 09:15:00 | 2310.70 | 2308.52 | 2316.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 2325.00 | 2311.82 | 2316.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 2325.00 | 2311.82 | 2316.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 2334.40 | 2316.33 | 2318.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:30:00 | 2335.95 | 2316.33 | 2318.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 2339.60 | 2320.99 | 2320.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 2348.25 | 2326.44 | 2322.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 2308.85 | 2331.01 | 2327.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 2308.85 | 2331.01 | 2327.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 2308.85 | 2331.01 | 2327.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 2308.85 | 2331.01 | 2327.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 2321.40 | 2329.08 | 2326.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:45:00 | 2330.55 | 2328.81 | 2326.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 10:15:00 | 2338.30 | 2336.31 | 2332.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 12:30:00 | 2338.20 | 2333.30 | 2331.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 13:30:00 | 2333.55 | 2331.62 | 2331.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 2347.00 | 2334.69 | 2332.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 2383.00 | 2336.56 | 2333.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-22 12:15:00 | 2438.90 | 2445.14 | 2445.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 12:15:00 | 2438.90 | 2445.14 | 2445.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 15:15:00 | 2428.00 | 2438.59 | 2442.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 09:15:00 | 2439.00 | 2438.68 | 2441.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 2439.00 | 2438.68 | 2441.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2439.00 | 2438.68 | 2441.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:30:00 | 2451.70 | 2438.68 | 2441.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 2433.30 | 2437.60 | 2441.18 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 2450.00 | 2443.33 | 2442.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 2487.40 | 2452.15 | 2446.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 11:15:00 | 2455.40 | 2456.52 | 2449.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 12:00:00 | 2455.40 | 2456.52 | 2449.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 2458.00 | 2456.81 | 2450.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:30:00 | 2451.20 | 2456.81 | 2450.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 2451.80 | 2455.81 | 2450.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:30:00 | 2450.80 | 2455.81 | 2450.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 2465.70 | 2457.79 | 2452.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:45:00 | 2451.00 | 2457.79 | 2452.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2430.90 | 2453.79 | 2451.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2430.90 | 2453.79 | 2451.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 2420.10 | 2447.05 | 2448.52 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 2454.80 | 2444.57 | 2443.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 2460.70 | 2448.06 | 2445.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 14:15:00 | 2451.00 | 2454.44 | 2450.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 14:15:00 | 2451.00 | 2454.44 | 2450.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 2451.00 | 2454.44 | 2450.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 2451.00 | 2454.44 | 2450.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 2456.10 | 2454.78 | 2450.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 2448.80 | 2454.78 | 2450.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 2454.20 | 2454.66 | 2451.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 12:00:00 | 2457.50 | 2455.15 | 2451.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 12:15:00 | 2436.30 | 2451.38 | 2450.53 | SL hit (close<static) qty=1.00 sl=2436.90 alert=retest2 |

### Cycle 143 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 2413.10 | 2443.73 | 2447.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 2399.00 | 2426.03 | 2435.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 2471.00 | 2428.68 | 2433.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 2471.00 | 2428.68 | 2433.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 2471.00 | 2428.68 | 2433.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 2471.00 | 2428.68 | 2433.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 2469.80 | 2436.91 | 2436.67 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 2409.60 | 2433.09 | 2435.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 2396.90 | 2421.86 | 2429.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 14:15:00 | 2417.80 | 2416.73 | 2424.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-06 15:00:00 | 2417.80 | 2416.73 | 2424.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2363.80 | 2319.65 | 2335.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 2363.80 | 2319.65 | 2335.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 2368.00 | 2329.32 | 2338.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:30:00 | 2371.30 | 2329.32 | 2338.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 2364.10 | 2345.37 | 2344.24 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 10:15:00 | 2334.10 | 2343.22 | 2343.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 09:15:00 | 2292.90 | 2322.83 | 2332.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 10:15:00 | 2293.00 | 2291.26 | 2307.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 2293.00 | 2291.26 | 2307.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 2314.50 | 2296.84 | 2307.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 2314.50 | 2296.84 | 2307.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 2321.40 | 2301.76 | 2308.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 2319.10 | 2301.76 | 2308.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 2315.90 | 2312.64 | 2312.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 2346.80 | 2322.30 | 2317.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 2341.30 | 2342.05 | 2331.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:00:00 | 2341.30 | 2342.05 | 2331.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 2331.20 | 2339.38 | 2332.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 2331.20 | 2339.38 | 2332.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 2331.00 | 2337.71 | 2332.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 2338.00 | 2337.71 | 2332.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:00:00 | 2334.10 | 2336.99 | 2332.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 11:15:00 | 2320.80 | 2332.71 | 2331.51 | SL hit (close<static) qty=1.00 sl=2326.60 alert=retest2 |

### Cycle 149 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 2311.00 | 2328.37 | 2329.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 2304.40 | 2323.58 | 2327.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 2311.30 | 2309.68 | 2315.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 2311.30 | 2309.68 | 2315.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 2311.30 | 2309.68 | 2315.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 2311.30 | 2309.68 | 2315.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2315.00 | 2302.55 | 2306.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 2315.00 | 2302.55 | 2306.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2322.80 | 2306.60 | 2308.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 2322.50 | 2306.60 | 2308.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 2322.10 | 2309.70 | 2309.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 2336.00 | 2318.08 | 2314.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 2308.60 | 2316.40 | 2313.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 11:15:00 | 2308.60 | 2316.40 | 2313.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 2308.60 | 2316.40 | 2313.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 2308.60 | 2316.40 | 2313.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 2307.90 | 2314.70 | 2313.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 2305.80 | 2314.70 | 2313.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2324.70 | 2321.65 | 2317.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 2324.90 | 2321.65 | 2317.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 2326.50 | 2330.88 | 2324.85 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 2304.20 | 2318.61 | 2320.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 2299.00 | 2307.92 | 2313.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 2268.60 | 2267.77 | 2279.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:45:00 | 2265.40 | 2267.77 | 2279.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 2264.90 | 2266.96 | 2275.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 2260.00 | 2264.72 | 2273.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 2244.10 | 2230.54 | 2230.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 2244.10 | 2230.54 | 2230.04 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 2215.20 | 2227.29 | 2228.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 2199.90 | 2220.71 | 2225.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 2213.00 | 2212.09 | 2218.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 2214.10 | 2212.09 | 2218.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 2224.70 | 2215.33 | 2218.80 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 2237.20 | 2224.03 | 2222.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 2244.30 | 2228.08 | 2224.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 2272.40 | 2274.49 | 2265.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 2272.40 | 2274.49 | 2265.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 2264.00 | 2271.76 | 2266.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 2278.50 | 2271.76 | 2266.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 2273.50 | 2272.11 | 2266.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 2285.90 | 2276.68 | 2271.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 2262.60 | 2269.43 | 2269.42 | SL hit (close<static) qty=1.00 sl=2263.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 2264.30 | 2268.40 | 2268.95 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 2295.00 | 2272.58 | 2270.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 2317.60 | 2291.90 | 2285.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 2334.90 | 2336.50 | 2318.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 2334.90 | 2336.50 | 2318.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2353.90 | 2358.49 | 2345.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:45:00 | 2370.30 | 2360.55 | 2347.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 15:15:00 | 2447.20 | 2469.58 | 2470.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 2447.20 | 2469.58 | 2470.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 2444.90 | 2459.46 | 2465.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 11:15:00 | 2418.60 | 2396.67 | 2409.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 2418.60 | 2396.67 | 2409.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 2418.60 | 2396.67 | 2409.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 2418.60 | 2396.67 | 2409.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 2411.70 | 2399.67 | 2409.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 2415.80 | 2399.67 | 2409.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 2411.20 | 2404.73 | 2410.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 2416.90 | 2404.73 | 2410.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2403.00 | 2404.39 | 2409.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 2406.00 | 2404.39 | 2409.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 2398.20 | 2403.15 | 2408.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 2395.30 | 2402.15 | 2406.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 15:15:00 | 2359.50 | 2352.22 | 2352.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 15:15:00 | 2359.50 | 2352.22 | 2352.02 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 2349.20 | 2351.62 | 2351.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 2342.30 | 2349.75 | 2350.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 2350.50 | 2349.51 | 2350.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 2350.50 | 2349.51 | 2350.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2350.50 | 2349.51 | 2350.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 2346.60 | 2349.51 | 2350.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 2398.10 | 2359.23 | 2354.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 2406.10 | 2375.09 | 2363.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 10:15:00 | 2397.40 | 2406.61 | 2392.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 10:45:00 | 2399.00 | 2406.61 | 2392.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 2386.50 | 2402.59 | 2391.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:00:00 | 2386.50 | 2402.59 | 2391.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 2403.90 | 2402.85 | 2393.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 2406.00 | 2402.85 | 2393.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 2411.20 | 2399.34 | 2393.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 2479.00 | 2485.44 | 2485.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 2479.00 | 2485.44 | 2485.70 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 2490.00 | 2486.35 | 2486.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 2500.00 | 2491.94 | 2489.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 2575.40 | 2576.51 | 2561.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 2562.40 | 2570.36 | 2565.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 2562.40 | 2570.36 | 2565.07 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 2538.50 | 2562.22 | 2563.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 2521.20 | 2554.01 | 2559.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 2505.10 | 2496.57 | 2511.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 12:00:00 | 2505.10 | 2496.57 | 2511.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 2510.50 | 2499.36 | 2511.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:45:00 | 2519.10 | 2499.36 | 2511.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 2511.70 | 2501.83 | 2511.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 2511.70 | 2501.83 | 2511.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 2473.80 | 2496.22 | 2508.11 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 2521.30 | 2507.64 | 2506.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 15:15:00 | 2523.50 | 2511.76 | 2508.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 11:15:00 | 2554.10 | 2557.75 | 2543.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:00:00 | 2554.10 | 2557.75 | 2543.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 2537.90 | 2553.82 | 2543.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 2537.90 | 2553.82 | 2543.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 2541.50 | 2551.36 | 2543.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 2538.50 | 2551.36 | 2543.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 2538.00 | 2548.68 | 2543.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 2544.60 | 2548.68 | 2543.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 2550.60 | 2549.74 | 2544.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 2550.60 | 2549.74 | 2544.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 2553.00 | 2551.70 | 2546.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 2553.00 | 2551.70 | 2546.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 2560.00 | 2553.61 | 2548.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 2566.00 | 2557.43 | 2550.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 2566.00 | 2558.54 | 2551.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:00:00 | 2568.00 | 2572.98 | 2568.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 13:15:00 | 2542.00 | 2563.23 | 2564.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 2542.00 | 2563.23 | 2564.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 2530.20 | 2556.62 | 2561.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 2541.30 | 2540.58 | 2548.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 15:15:00 | 2541.30 | 2540.58 | 2548.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 2541.30 | 2540.58 | 2548.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 2546.00 | 2540.58 | 2548.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2531.50 | 2538.76 | 2547.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 2527.00 | 2538.76 | 2547.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:00:00 | 2530.00 | 2536.50 | 2544.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 2553.60 | 2546.60 | 2545.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 2553.60 | 2546.60 | 2545.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 2556.00 | 2548.48 | 2546.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 2539.20 | 2550.76 | 2549.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 12:15:00 | 2539.20 | 2550.76 | 2549.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 2539.20 | 2550.76 | 2549.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 2539.20 | 2550.76 | 2549.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 2551.80 | 2550.97 | 2549.32 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 2540.50 | 2548.27 | 2548.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 2514.80 | 2541.58 | 2545.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 2492.80 | 2489.21 | 2499.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 15:00:00 | 2492.80 | 2489.21 | 2499.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2495.10 | 2491.24 | 2498.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:15:00 | 2491.00 | 2492.60 | 2498.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 2507.00 | 2496.32 | 2496.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 2507.00 | 2496.32 | 2496.05 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 2477.90 | 2494.11 | 2495.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 2434.40 | 2477.67 | 2486.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 2460.80 | 2458.38 | 2470.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 2460.80 | 2458.38 | 2470.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 2469.20 | 2460.55 | 2470.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 2469.20 | 2460.55 | 2470.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 2470.70 | 2462.58 | 2470.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 2470.70 | 2462.58 | 2470.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 2467.00 | 2463.46 | 2469.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:15:00 | 2459.20 | 2464.29 | 2469.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2336.24 | 2363.74 | 2393.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 2350.10 | 2349.94 | 2363.16 | SL hit (close>ema200) qty=0.50 sl=2349.94 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 2354.40 | 2349.59 | 2349.16 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 2338.10 | 2347.29 | 2348.16 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 2353.90 | 2349.52 | 2349.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 2369.90 | 2356.50 | 2352.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 2348.80 | 2356.91 | 2353.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 2348.80 | 2356.91 | 2353.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 2348.80 | 2356.91 | 2353.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 2348.80 | 2356.91 | 2353.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 2354.70 | 2356.47 | 2353.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 2359.20 | 2356.47 | 2353.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 2342.40 | 2353.66 | 2352.83 | SL hit (close<static) qty=1.00 sl=2348.80 alert=retest2 |

### Cycle 173 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 2337.10 | 2350.35 | 2351.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 12:15:00 | 2332.10 | 2344.40 | 2348.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 2334.10 | 2332.47 | 2338.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 13:15:00 | 2334.10 | 2332.47 | 2338.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 2334.10 | 2332.47 | 2338.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:45:00 | 2334.50 | 2332.47 | 2338.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 2340.00 | 2334.46 | 2338.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 2353.00 | 2334.46 | 2338.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 2348.20 | 2337.21 | 2339.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 2349.30 | 2337.21 | 2339.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 2348.50 | 2339.47 | 2339.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 2348.50 | 2339.47 | 2339.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 2347.40 | 2341.05 | 2340.60 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 2326.50 | 2339.61 | 2341.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 2321.00 | 2335.89 | 2339.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 2353.50 | 2331.30 | 2334.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 2353.50 | 2331.30 | 2334.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2353.50 | 2331.30 | 2334.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 2347.60 | 2331.30 | 2334.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 2366.50 | 2338.34 | 2337.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 2379.00 | 2355.65 | 2346.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 13:15:00 | 2509.00 | 2510.86 | 2481.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 2506.90 | 2510.86 | 2481.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2498.70 | 2516.18 | 2499.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 2498.70 | 2516.18 | 2499.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 2499.70 | 2512.89 | 2499.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 2501.90 | 2512.89 | 2499.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 2494.60 | 2509.23 | 2499.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:15:00 | 2488.30 | 2509.23 | 2499.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 2490.70 | 2505.52 | 2498.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 2489.40 | 2505.52 | 2498.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 2497.00 | 2503.82 | 2498.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 2499.90 | 2503.82 | 2498.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 2499.40 | 2501.83 | 2498.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 2500.90 | 2501.64 | 2498.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 2506.30 | 2509.52 | 2507.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 2506.90 | 2508.99 | 2507.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 2501.20 | 2506.63 | 2506.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 2501.20 | 2506.63 | 2506.82 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 2524.60 | 2510.22 | 2508.44 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 2509.50 | 2519.87 | 2520.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 2499.10 | 2513.83 | 2517.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 2514.50 | 2508.80 | 2513.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 2514.50 | 2508.80 | 2513.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 2514.50 | 2508.80 | 2513.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 2514.50 | 2508.80 | 2513.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 2506.50 | 2508.34 | 2512.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 2503.10 | 2508.34 | 2512.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2509.90 | 2508.65 | 2512.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:15:00 | 2494.10 | 2507.40 | 2510.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 2618.00 | 2521.76 | 2515.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 2618.00 | 2521.76 | 2515.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 2652.50 | 2611.13 | 2584.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 2639.00 | 2641.78 | 2617.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:30:00 | 2638.30 | 2641.78 | 2617.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 2789.10 | 2684.67 | 2656.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 2660.50 | 2684.67 | 2656.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2890.00 | 2891.05 | 2865.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 2904.30 | 2896.44 | 2872.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:00:00 | 2906.20 | 2901.23 | 2880.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 2909.80 | 2900.74 | 2884.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 2902.50 | 2901.17 | 2885.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2892.00 | 2899.54 | 2891.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2891.40 | 2899.54 | 2891.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2874.90 | 2894.61 | 2889.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 2874.90 | 2894.61 | 2889.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 2872.60 | 2890.21 | 2888.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 2870.00 | 2890.21 | 2888.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 2858.80 | 2883.93 | 2885.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 2858.80 | 2883.93 | 2885.46 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 09:15:00 | 2894.50 | 2878.09 | 2877.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 09:15:00 | 2898.50 | 2884.37 | 2880.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 12:15:00 | 2886.30 | 2886.46 | 2882.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 13:00:00 | 2886.30 | 2886.46 | 2882.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 2888.90 | 2886.94 | 2883.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 2885.10 | 2886.94 | 2883.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2876.90 | 2884.94 | 2882.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 2876.90 | 2884.94 | 2882.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 2877.20 | 2883.39 | 2882.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 2886.10 | 2883.39 | 2882.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 13:00:00 | 2879.20 | 2883.64 | 2882.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 2864.40 | 2879.79 | 2881.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 13:15:00 | 2864.40 | 2879.79 | 2881.26 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 2899.70 | 2881.76 | 2881.61 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 2873.00 | 2881.08 | 2881.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 2856.20 | 2871.63 | 2875.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 2867.90 | 2864.22 | 2869.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 2867.90 | 2864.22 | 2869.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 2861.50 | 2863.67 | 2869.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 2907.70 | 2863.67 | 2869.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 2890.00 | 2868.94 | 2870.99 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 2904.90 | 2876.13 | 2874.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 12:15:00 | 2919.00 | 2889.80 | 2880.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 12:15:00 | 2952.90 | 2956.90 | 2937.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 13:00:00 | 2952.90 | 2956.90 | 2937.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 2950.00 | 2953.36 | 2944.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 2959.90 | 2953.36 | 2944.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 2935.40 | 2952.64 | 2948.70 | SL hit (close<static) qty=1.00 sl=2941.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 2920.80 | 2943.06 | 2944.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 2811.30 | 2911.62 | 2929.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 2782.30 | 2770.84 | 2787.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 2782.30 | 2770.84 | 2787.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2782.30 | 2770.84 | 2787.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 2781.40 | 2770.84 | 2787.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2793.90 | 2775.45 | 2788.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 2793.90 | 2775.45 | 2788.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 2796.80 | 2779.72 | 2788.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 2794.10 | 2779.72 | 2788.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 2782.30 | 2780.24 | 2788.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 14:15:00 | 2777.00 | 2780.43 | 2787.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 2777.80 | 2780.72 | 2787.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 2804.60 | 2785.63 | 2787.80 | SL hit (close>static) qty=1.00 sl=2802.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 2801.10 | 2790.48 | 2789.49 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 2778.90 | 2788.10 | 2788.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 2770.30 | 2782.50 | 2785.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 2781.40 | 2770.49 | 2776.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 2781.40 | 2770.49 | 2776.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2781.40 | 2770.49 | 2776.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 2781.40 | 2770.49 | 2776.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2784.90 | 2773.37 | 2777.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 2788.90 | 2773.37 | 2777.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 2788.00 | 2780.24 | 2779.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2818.50 | 2793.01 | 2786.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 2792.40 | 2795.28 | 2789.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:30:00 | 2792.60 | 2795.28 | 2789.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2796.10 | 2799.34 | 2793.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 2786.00 | 2799.34 | 2793.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2789.90 | 2797.45 | 2793.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 2788.50 | 2797.45 | 2793.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2802.40 | 2798.44 | 2794.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 12:30:00 | 2804.60 | 2799.71 | 2795.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 14:15:00 | 2787.60 | 2797.46 | 2797.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 2787.60 | 2797.46 | 2797.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 2762.70 | 2788.67 | 2793.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 12:15:00 | 2760.00 | 2754.72 | 2767.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 13:00:00 | 2760.00 | 2754.72 | 2767.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2787.90 | 2761.36 | 2769.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 2787.90 | 2761.36 | 2769.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2774.50 | 2763.98 | 2769.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 2764.10 | 2766.19 | 2770.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 2788.30 | 2773.99 | 2773.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 11:15:00 | 2788.30 | 2773.99 | 2773.06 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 2758.50 | 2770.73 | 2771.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 15:15:00 | 2753.50 | 2765.18 | 2768.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 2781.90 | 2768.53 | 2770.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 2781.90 | 2768.53 | 2770.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2781.90 | 2768.53 | 2770.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 2780.80 | 2768.53 | 2770.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2775.00 | 2769.82 | 2770.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 2773.60 | 2769.82 | 2770.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 2778.40 | 2771.54 | 2771.25 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 2749.70 | 2768.67 | 2770.89 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 2802.10 | 2776.17 | 2773.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 2823.20 | 2791.31 | 2780.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 2821.80 | 2833.25 | 2819.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 12:00:00 | 2821.80 | 2833.25 | 2819.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 2813.80 | 2829.36 | 2819.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 2814.70 | 2829.36 | 2819.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 2812.20 | 2825.93 | 2818.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:45:00 | 2809.20 | 2825.93 | 2818.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2780.80 | 2811.93 | 2813.51 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 2824.80 | 2808.33 | 2806.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 12:15:00 | 2833.60 | 2813.38 | 2809.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 2840.20 | 2870.67 | 2859.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 2840.20 | 2870.67 | 2859.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2840.20 | 2870.67 | 2859.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 2840.20 | 2870.67 | 2859.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2838.60 | 2864.26 | 2857.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 2843.40 | 2864.26 | 2857.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 13:15:00 | 2820.10 | 2847.43 | 2851.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 14:15:00 | 2810.90 | 2840.12 | 2847.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2706.50 | 2680.27 | 2705.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2706.50 | 2680.27 | 2705.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2706.50 | 2680.27 | 2705.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 2724.00 | 2680.27 | 2705.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2696.40 | 2683.49 | 2704.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 2691.90 | 2683.49 | 2704.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 2691.20 | 2683.91 | 2701.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2737.90 | 2702.25 | 2705.13 | SL hit (close>static) qty=1.00 sl=2710.30 alert=retest2 |

### Cycle 200 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 2782.00 | 2718.20 | 2712.12 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 2680.00 | 2712.11 | 2714.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 2629.70 | 2686.33 | 2701.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 2438.40 | 2436.20 | 2481.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 14:00:00 | 2438.40 | 2436.20 | 2481.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2407.50 | 2392.30 | 2426.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 2417.10 | 2392.30 | 2426.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2417.10 | 2401.29 | 2415.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2402.70 | 2401.29 | 2415.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 2445.70 | 2423.29 | 2420.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 2445.70 | 2423.29 | 2420.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 2460.60 | 2440.16 | 2430.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2421.50 | 2439.85 | 2433.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 2421.50 | 2439.85 | 2433.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 2421.50 | 2439.85 | 2433.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 2418.00 | 2439.85 | 2433.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 2409.70 | 2433.82 | 2431.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 2409.70 | 2433.82 | 2431.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 2433.50 | 2430.70 | 2429.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 2404.50 | 2430.70 | 2429.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2401.00 | 2424.76 | 2427.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 2390.50 | 2410.42 | 2419.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2417.00 | 2408.01 | 2415.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 2417.00 | 2408.01 | 2415.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2417.00 | 2408.01 | 2415.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 2417.00 | 2408.01 | 2415.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 2417.70 | 2409.95 | 2415.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 2422.60 | 2409.95 | 2415.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 2416.50 | 2411.51 | 2414.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 2416.50 | 2411.51 | 2414.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 2417.00 | 2412.61 | 2414.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 2411.00 | 2412.61 | 2414.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2383.90 | 2396.46 | 2403.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 2380.40 | 2394.28 | 2402.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 2379.70 | 2387.77 | 2394.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 2410.40 | 2395.86 | 2396.55 | SL hit (close>static) qty=1.00 sl=2405.70 alert=retest2 |

### Cycle 204 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 2410.90 | 2398.87 | 2397.86 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 2384.60 | 2395.74 | 2396.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 2365.00 | 2387.95 | 2393.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 2380.90 | 2379.70 | 2386.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 2380.90 | 2379.70 | 2386.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 2390.40 | 2381.84 | 2387.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 2390.40 | 2381.84 | 2387.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 2397.60 | 2384.99 | 2388.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 2396.20 | 2384.99 | 2388.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 2397.50 | 2387.49 | 2388.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 2397.50 | 2387.49 | 2388.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 2400.00 | 2390.00 | 2389.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 2433.60 | 2398.72 | 2393.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 11:15:00 | 2419.10 | 2425.90 | 2415.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 12:00:00 | 2419.10 | 2425.90 | 2415.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2407.20 | 2425.02 | 2419.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 2403.30 | 2425.02 | 2419.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 2400.00 | 2420.01 | 2417.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 2400.00 | 2420.01 | 2417.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 2409.10 | 2415.21 | 2415.79 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 2421.80 | 2414.00 | 2413.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 2436.90 | 2427.85 | 2421.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 2424.10 | 2427.10 | 2422.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 14:00:00 | 2424.10 | 2427.10 | 2422.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 2425.60 | 2426.80 | 2422.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 2423.50 | 2426.80 | 2422.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 2433.30 | 2429.06 | 2424.31 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 2412.70 | 2421.87 | 2421.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 10:15:00 | 2398.20 | 2410.32 | 2414.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 2388.10 | 2384.01 | 2394.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 14:00:00 | 2388.10 | 2384.01 | 2394.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 2375.90 | 2382.39 | 2392.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 2394.00 | 2382.39 | 2392.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2284.80 | 2273.61 | 2293.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 2297.10 | 2273.61 | 2293.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2286.00 | 2277.96 | 2290.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 2286.00 | 2277.96 | 2290.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 2304.60 | 2283.29 | 2292.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 2304.60 | 2283.29 | 2292.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 2300.00 | 2286.63 | 2292.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 2295.10 | 2286.46 | 2292.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 2180.34 | 2261.00 | 2278.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 2255.80 | 2233.45 | 2251.41 | SL hit (close>ema200) qty=0.50 sl=2233.45 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 2279.00 | 2258.79 | 2258.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 2295.00 | 2266.03 | 2261.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 2268.70 | 2269.09 | 2264.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 2268.70 | 2269.09 | 2264.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 2262.10 | 2267.69 | 2263.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 2265.80 | 2267.69 | 2263.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 2244.80 | 2263.11 | 2262.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 2244.80 | 2263.11 | 2262.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 2242.40 | 2258.97 | 2260.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 2230.90 | 2253.36 | 2257.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 2248.20 | 2244.55 | 2251.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 11:00:00 | 2248.20 | 2244.55 | 2251.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 2221.80 | 2203.28 | 2212.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 2225.60 | 2203.28 | 2212.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 2215.00 | 2205.62 | 2212.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 2232.90 | 2205.62 | 2212.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2253.60 | 2215.22 | 2216.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 2253.60 | 2215.22 | 2216.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 2236.90 | 2219.56 | 2218.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 2263.90 | 2243.86 | 2234.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2202.60 | 2238.19 | 2233.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2202.60 | 2238.19 | 2233.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2202.60 | 2238.19 | 2233.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 2202.20 | 2238.19 | 2233.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 2195.70 | 2229.69 | 2230.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 2186.00 | 2207.48 | 2218.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2210.90 | 2206.15 | 2215.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2210.90 | 2206.15 | 2215.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2210.90 | 2206.15 | 2215.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:00:00 | 2199.80 | 2206.55 | 2213.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 2159.70 | 2204.29 | 2210.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:45:00 | 2203.50 | 2167.26 | 2174.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 2220.30 | 2177.87 | 2178.50 | SL hit (close>static) qty=1.00 sl=2219.60 alert=retest2 |

### Cycle 214 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 2219.20 | 2186.13 | 2182.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2248.40 | 2203.96 | 2191.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2250.50 | 2255.43 | 2230.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 2230.20 | 2246.54 | 2232.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 2230.20 | 2246.54 | 2232.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 2230.20 | 2246.54 | 2232.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 2213.90 | 2240.01 | 2230.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 2213.90 | 2240.01 | 2230.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 2208.60 | 2233.73 | 2228.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 2208.60 | 2233.73 | 2228.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 2181.80 | 2219.42 | 2222.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 2170.50 | 2209.64 | 2217.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2231.20 | 2191.14 | 2202.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2231.20 | 2191.14 | 2202.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2231.20 | 2191.14 | 2202.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 2231.20 | 2191.14 | 2202.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 2211.90 | 2195.29 | 2203.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 2207.00 | 2200.27 | 2204.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 2240.00 | 2208.22 | 2207.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 2240.00 | 2208.22 | 2207.82 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 2170.00 | 2207.69 | 2208.63 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 2197.00 | 2181.58 | 2181.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2280.80 | 2201.43 | 2190.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 2265.00 | 2266.39 | 2243.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 2265.00 | 2266.39 | 2243.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2328.10 | 2338.90 | 2305.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2343.50 | 2338.90 | 2305.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 2577.85 | 2512.77 | 2479.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 2509.40 | 2528.21 | 2528.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 2484.30 | 2519.42 | 2524.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 2507.20 | 2497.27 | 2508.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 2507.20 | 2497.27 | 2508.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 2507.20 | 2497.27 | 2508.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 2518.00 | 2497.27 | 2508.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 2496.60 | 2497.14 | 2507.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 2491.40 | 2497.14 | 2507.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 2482.00 | 2494.11 | 2504.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:45:00 | 2491.40 | 2493.40 | 2502.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:00:00 | 2487.70 | 2492.26 | 2501.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 2488.00 | 2487.30 | 2496.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:30:00 | 2494.20 | 2487.30 | 2496.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 2477.00 | 2482.49 | 2491.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:45:00 | 2484.30 | 2482.49 | 2491.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 2454.60 | 2465.03 | 2476.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 2449.20 | 2465.03 | 2476.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 2447.00 | 2461.52 | 2473.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:45:00 | 2448.90 | 2435.55 | 2450.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 2486.60 | 2447.88 | 2453.82 | SL hit (close>static) qty=1.00 sl=2477.90 alert=retest2 |

### Cycle 220 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 2458.00 | 2457.54 | 2457.51 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 14:15:00 | 2450.00 | 2456.03 | 2456.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 2435.90 | 2450.72 | 2454.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 2449.40 | 2437.89 | 2443.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 2449.40 | 2437.89 | 2443.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 2449.40 | 2437.89 | 2443.83 | EMA400 retest candle locked (from downside) |

### Cycle 222 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 2478.90 | 2451.26 | 2449.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 2494.90 | 2463.76 | 2455.47 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 13:30:00 | 3085.00 | 2023-05-23 09:15:00 | 3107.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-05-22 14:15:00 | 3087.80 | 2023-05-23 09:15:00 | 3107.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-05-31 10:45:00 | 3189.35 | 2023-06-05 15:15:00 | 3200.65 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2023-05-31 14:30:00 | 3204.00 | 2023-06-05 15:15:00 | 3200.65 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2023-05-31 15:15:00 | 3196.95 | 2023-06-05 15:15:00 | 3200.65 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2023-06-20 13:15:00 | 3309.00 | 2023-06-22 10:15:00 | 3296.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-06-20 15:00:00 | 3319.55 | 2023-06-22 11:15:00 | 3281.85 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-06-21 10:45:00 | 3308.40 | 2023-06-22 11:15:00 | 3281.85 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-06-21 11:15:00 | 3316.15 | 2023-06-22 11:15:00 | 3281.85 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-06-21 15:15:00 | 3322.00 | 2023-06-22 11:15:00 | 3281.85 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-07-14 09:15:00 | 3406.55 | 2023-07-25 11:15:00 | 3478.75 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2023-07-14 11:15:00 | 3406.65 | 2023-07-25 11:15:00 | 3478.75 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2023-07-14 12:00:00 | 3407.95 | 2023-07-25 11:15:00 | 3478.75 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2023-07-31 09:15:00 | 3340.00 | 2023-08-08 14:15:00 | 3351.50 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2023-08-10 09:15:00 | 3295.95 | 2023-08-22 11:15:00 | 3189.05 | STOP_HIT | 1.00 | 3.24% |
| SELL | retest2 | 2023-09-07 09:15:00 | 3207.10 | 2023-09-07 13:15:00 | 3241.60 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-09-11 09:15:00 | 3258.35 | 2023-09-14 11:15:00 | 3232.25 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-09-13 10:45:00 | 3240.00 | 2023-09-14 11:15:00 | 3232.25 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2023-09-21 10:15:00 | 3186.00 | 2023-09-21 14:15:00 | 3242.10 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2023-09-26 13:00:00 | 3293.20 | 2023-09-28 09:15:00 | 3254.20 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-09-27 11:45:00 | 3295.50 | 2023-09-28 09:15:00 | 3254.20 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2023-10-05 10:30:00 | 3182.60 | 2023-10-05 12:15:00 | 3200.35 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-10-16 09:15:00 | 3116.20 | 2023-10-26 13:15:00 | 2960.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-16 09:15:00 | 3116.20 | 2023-10-30 09:15:00 | 2971.00 | STOP_HIT | 0.50 | 4.66% |
| BUY | retest2 | 2023-11-10 11:15:00 | 3068.75 | 2023-11-21 11:15:00 | 3133.50 | STOP_HIT | 1.00 | 2.11% |
| SELL | retest2 | 2023-11-23 10:30:00 | 3140.10 | 2023-11-28 09:15:00 | 3145.85 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2023-11-23 11:30:00 | 3141.60 | 2023-11-28 09:15:00 | 3145.85 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2023-11-23 12:00:00 | 3142.80 | 2023-11-28 09:15:00 | 3145.85 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2023-11-23 13:00:00 | 3138.70 | 2023-11-28 10:15:00 | 3152.35 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2023-11-23 15:00:00 | 3125.60 | 2023-11-28 10:15:00 | 3152.35 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-11-24 09:30:00 | 3123.00 | 2023-11-28 10:15:00 | 3152.35 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-11-24 10:45:00 | 3125.10 | 2023-11-28 10:15:00 | 3152.35 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-12-14 15:15:00 | 3245.75 | 2023-12-21 09:15:00 | 3289.65 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2024-01-01 14:30:00 | 3398.55 | 2024-01-01 15:15:00 | 3390.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-01-11 10:30:00 | 3281.50 | 2024-01-16 10:15:00 | 3300.40 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-01-11 12:00:00 | 3279.95 | 2024-01-16 10:15:00 | 3300.40 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-01-11 12:30:00 | 3282.35 | 2024-01-16 10:15:00 | 3300.40 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-01-11 13:00:00 | 3280.15 | 2024-01-16 10:15:00 | 3300.40 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2024-01-19 10:45:00 | 3163.80 | 2024-01-24 09:15:00 | 3005.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-01-19 12:45:00 | 3169.75 | 2024-01-24 09:15:00 | 3011.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-01-19 14:00:00 | 3163.75 | 2024-01-24 09:15:00 | 3005.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-01-20 10:45:00 | 3159.20 | 2024-01-24 09:15:00 | 3001.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-01-19 10:45:00 | 3163.80 | 2024-01-29 09:15:00 | 2974.00 | STOP_HIT | 0.50 | 6.00% |
| SELL | retest1 | 2024-01-19 12:45:00 | 3169.75 | 2024-01-29 09:15:00 | 2974.00 | STOP_HIT | 0.50 | 6.18% |
| SELL | retest1 | 2024-01-19 14:00:00 | 3163.75 | 2024-01-29 09:15:00 | 2974.00 | STOP_HIT | 0.50 | 6.00% |
| SELL | retest1 | 2024-01-20 10:45:00 | 3159.20 | 2024-01-29 09:15:00 | 2974.00 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2024-01-30 14:30:00 | 2962.60 | 2024-02-07 10:15:00 | 2947.55 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-01-31 11:30:00 | 2962.65 | 2024-02-07 10:15:00 | 2947.55 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2024-02-08 12:15:00 | 2945.15 | 2024-02-08 13:15:00 | 2922.95 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-02-08 13:00:00 | 2947.90 | 2024-02-08 13:15:00 | 2922.95 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-02-14 14:45:00 | 2973.45 | 2024-02-20 10:15:00 | 2988.20 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2024-03-01 09:15:00 | 2820.05 | 2024-03-02 09:15:00 | 2853.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-03-15 11:30:00 | 2885.90 | 2024-03-15 14:15:00 | 2862.65 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-03-15 12:30:00 | 2889.55 | 2024-03-15 14:15:00 | 2862.65 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest1 | 2024-03-20 13:45:00 | 2810.20 | 2024-03-20 15:15:00 | 2835.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-03-22 09:15:00 | 2812.25 | 2024-03-22 14:15:00 | 2835.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-04-01 09:15:00 | 2877.75 | 2024-04-09 11:15:00 | 2875.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-04-01 11:45:00 | 2866.00 | 2024-04-09 11:15:00 | 2875.00 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2024-04-01 12:30:00 | 2867.15 | 2024-04-09 11:15:00 | 2875.00 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2024-04-01 14:15:00 | 2869.55 | 2024-04-09 11:15:00 | 2875.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-04-03 11:00:00 | 2863.30 | 2024-04-09 11:15:00 | 2875.00 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2024-04-04 11:00:00 | 2865.95 | 2024-04-09 11:15:00 | 2875.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-04-18 15:00:00 | 2805.05 | 2024-04-22 12:15:00 | 2839.20 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-04-19 14:30:00 | 2802.80 | 2024-04-22 12:15:00 | 2839.20 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-05-02 09:15:00 | 2919.00 | 2024-05-07 13:15:00 | 2918.40 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-05-27 10:30:00 | 2885.00 | 2024-05-30 14:15:00 | 2881.40 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-05-27 11:30:00 | 2883.30 | 2024-05-30 14:15:00 | 2881.40 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-05-28 10:30:00 | 2883.00 | 2024-05-30 14:15:00 | 2881.40 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-05-28 11:30:00 | 2887.05 | 2024-05-30 14:15:00 | 2881.40 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-05-29 13:00:00 | 2916.45 | 2024-05-30 14:15:00 | 2881.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-06-04 12:15:00 | 2791.50 | 2024-06-05 09:15:00 | 2991.90 | STOP_HIT | 1.00 | -7.18% |
| BUY | retest2 | 2024-06-07 11:45:00 | 2936.25 | 2024-06-11 10:15:00 | 2910.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-06-10 09:15:00 | 2938.10 | 2024-06-11 10:15:00 | 2910.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-06-10 13:00:00 | 2936.45 | 2024-06-11 10:15:00 | 2910.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-06-10 14:30:00 | 2939.95 | 2024-06-11 10:15:00 | 2910.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-06-24 09:30:00 | 2878.75 | 2024-06-27 15:15:00 | 2885.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-06-24 12:45:00 | 2879.90 | 2024-06-27 15:15:00 | 2885.80 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-06-25 09:45:00 | 2880.75 | 2024-06-27 15:15:00 | 2885.80 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-06-25 11:00:00 | 2877.60 | 2024-06-27 15:15:00 | 2885.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-07-04 10:30:00 | 2965.05 | 2024-07-08 09:15:00 | 2898.75 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-08-06 09:45:00 | 3123.85 | 2024-08-07 11:15:00 | 3089.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-08-06 10:45:00 | 3125.00 | 2024-08-07 11:15:00 | 3089.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-08-16 10:15:00 | 3008.15 | 2024-08-16 14:15:00 | 3048.55 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-08-26 12:30:00 | 3170.80 | 2024-08-28 10:15:00 | 3137.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-08-27 09:45:00 | 3171.00 | 2024-08-28 10:15:00 | 3137.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-08-30 13:30:00 | 3135.00 | 2024-09-02 09:15:00 | 3156.30 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-09-13 11:30:00 | 3353.65 | 2024-09-16 14:15:00 | 3338.20 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-09-13 12:30:00 | 3356.95 | 2024-09-16 14:15:00 | 3338.20 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-09-13 13:30:00 | 3355.30 | 2024-09-16 14:15:00 | 3338.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-09-16 10:00:00 | 3360.30 | 2024-09-16 14:15:00 | 3338.20 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-09-19 12:15:00 | 3294.50 | 2024-09-20 14:15:00 | 3305.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-09-19 14:45:00 | 3297.05 | 2024-09-20 14:15:00 | 3305.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-09-20 09:15:00 | 3278.30 | 2024-09-20 14:15:00 | 3305.45 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-09-20 10:00:00 | 3298.25 | 2024-09-20 14:15:00 | 3305.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-09-20 13:30:00 | 3286.15 | 2024-09-20 14:15:00 | 3305.45 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-09-24 11:15:00 | 3270.00 | 2024-09-27 09:15:00 | 3311.90 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-09-26 15:00:00 | 3276.85 | 2024-09-27 09:15:00 | 3311.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-10-09 13:15:00 | 3094.85 | 2024-10-15 11:15:00 | 3084.95 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-10-09 14:15:00 | 3087.65 | 2024-10-15 11:15:00 | 3084.95 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2024-10-22 10:00:00 | 3034.40 | 2024-10-28 11:15:00 | 3015.55 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2024-11-07 12:45:00 | 2845.90 | 2024-11-11 09:15:00 | 2561.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 14:30:00 | 2497.45 | 2024-11-28 10:15:00 | 2471.80 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-12-04 11:45:00 | 2459.40 | 2024-12-11 14:15:00 | 2418.10 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2024-12-04 14:15:00 | 2458.85 | 2024-12-11 14:15:00 | 2418.10 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2024-12-05 14:45:00 | 2435.80 | 2024-12-11 14:15:00 | 2418.10 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-12-17 09:15:00 | 2393.00 | 2024-12-19 09:15:00 | 2273.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 2393.00 | 2024-12-20 10:15:00 | 2315.50 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2025-01-28 14:45:00 | 2241.90 | 2025-01-30 15:15:00 | 2255.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-01-30 13:15:00 | 2249.05 | 2025-01-30 15:15:00 | 2255.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-02-11 10:30:00 | 2244.30 | 2025-02-17 12:15:00 | 2254.70 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-02-11 12:45:00 | 2247.75 | 2025-02-17 12:15:00 | 2254.70 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-02-17 10:45:00 | 2247.75 | 2025-02-17 12:15:00 | 2254.70 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-02-21 11:30:00 | 2254.15 | 2025-02-25 13:15:00 | 2233.10 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-02-21 13:15:00 | 2250.90 | 2025-02-25 13:15:00 | 2233.10 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-02-21 14:15:00 | 2251.00 | 2025-02-25 13:15:00 | 2233.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-02-24 10:30:00 | 2256.70 | 2025-02-25 13:15:00 | 2233.10 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-03-04 09:15:00 | 2148.65 | 2025-03-06 09:15:00 | 2217.30 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-03-11 10:30:00 | 2287.55 | 2025-03-12 12:15:00 | 2256.20 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-03-11 14:00:00 | 2286.95 | 2025-03-12 12:15:00 | 2256.20 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-03-11 15:00:00 | 2288.80 | 2025-03-12 12:15:00 | 2256.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-03-26 14:15:00 | 2325.00 | 2025-04-01 13:15:00 | 2310.25 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-03-26 15:00:00 | 2325.15 | 2025-04-01 13:15:00 | 2310.25 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-03-27 10:30:00 | 2325.90 | 2025-04-01 13:15:00 | 2310.25 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-03-27 11:15:00 | 2323.00 | 2025-04-01 13:15:00 | 2310.25 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-04-04 12:45:00 | 2330.55 | 2025-04-22 12:15:00 | 2438.90 | STOP_HIT | 1.00 | 4.65% |
| BUY | retest2 | 2025-04-07 10:15:00 | 2338.30 | 2025-04-22 12:15:00 | 2438.90 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2025-04-07 12:30:00 | 2338.20 | 2025-04-22 12:15:00 | 2438.90 | STOP_HIT | 1.00 | 4.31% |
| BUY | retest2 | 2025-04-07 13:30:00 | 2333.55 | 2025-04-22 12:15:00 | 2438.90 | STOP_HIT | 1.00 | 4.51% |
| BUY | retest2 | 2025-04-08 09:15:00 | 2383.00 | 2025-04-22 12:15:00 | 2438.90 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-04-30 12:00:00 | 2457.50 | 2025-04-30 12:15:00 | 2436.30 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-20 09:15:00 | 2338.00 | 2025-05-20 11:15:00 | 2320.80 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-05-20 10:00:00 | 2334.10 | 2025-05-20 11:15:00 | 2320.80 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-06-03 10:45:00 | 2260.00 | 2025-06-12 11:15:00 | 2244.10 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2025-06-20 15:00:00 | 2285.90 | 2025-06-23 12:15:00 | 2262.60 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-02 10:45:00 | 2370.30 | 2025-07-10 15:15:00 | 2447.20 | STOP_HIT | 1.00 | 3.24% |
| SELL | retest2 | 2025-07-18 09:15:00 | 2395.30 | 2025-07-28 15:15:00 | 2359.50 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2025-07-31 13:15:00 | 2406.00 | 2025-08-12 15:15:00 | 2479.00 | STOP_HIT | 1.00 | 3.03% |
| BUY | retest2 | 2025-08-01 09:15:00 | 2411.20 | 2025-08-12 15:15:00 | 2479.00 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2025-09-04 09:30:00 | 2566.00 | 2025-09-08 13:15:00 | 2542.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-04 11:15:00 | 2566.00 | 2025-09-08 13:15:00 | 2542.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-08 11:00:00 | 2568.00 | 2025-09-08 13:15:00 | 2542.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-10 10:15:00 | 2527.00 | 2025-09-11 13:15:00 | 2553.60 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-10 12:00:00 | 2530.00 | 2025-09-11 13:15:00 | 2553.60 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-18 12:15:00 | 2491.00 | 2025-09-19 12:15:00 | 2507.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-24 14:15:00 | 2459.20 | 2025-09-29 09:15:00 | 2336.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:15:00 | 2459.20 | 2025-09-30 14:15:00 | 2350.10 | STOP_HIT | 0.50 | 4.44% |
| BUY | retest2 | 2025-10-08 09:15:00 | 2359.20 | 2025-10-08 09:15:00 | 2342.40 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-24 12:15:00 | 2499.90 | 2025-10-28 15:15:00 | 2501.20 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-10-24 14:00:00 | 2499.40 | 2025-10-28 15:15:00 | 2501.20 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-10-24 15:00:00 | 2500.90 | 2025-10-28 15:15:00 | 2501.20 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-10-28 12:15:00 | 2506.30 | 2025-10-28 15:15:00 | 2501.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-04 13:15:00 | 2494.10 | 2025-11-06 09:15:00 | 2618.00 | STOP_HIT | 1.00 | -4.97% |
| BUY | retest2 | 2025-11-18 11:30:00 | 2904.30 | 2025-11-20 11:15:00 | 2858.80 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-18 15:00:00 | 2906.20 | 2025-11-20 11:15:00 | 2858.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-19 09:30:00 | 2909.80 | 2025-11-20 11:15:00 | 2858.80 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-11-19 10:30:00 | 2902.50 | 2025-11-20 11:15:00 | 2858.80 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-11-26 09:15:00 | 2886.10 | 2025-11-26 13:15:00 | 2864.40 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-11-26 13:00:00 | 2879.20 | 2025-11-26 13:15:00 | 2864.40 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-05 12:15:00 | 2959.90 | 2025-12-08 10:15:00 | 2935.40 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-08 10:30:00 | 2950.20 | 2025-12-08 11:15:00 | 2937.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-12-15 14:15:00 | 2777.00 | 2025-12-16 10:15:00 | 2804.60 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-15 15:15:00 | 2777.80 | 2025-12-16 10:15:00 | 2804.60 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-23 12:30:00 | 2804.60 | 2025-12-24 14:15:00 | 2787.60 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-12-30 09:15:00 | 2764.10 | 2025-12-30 11:15:00 | 2788.30 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-01-22 11:15:00 | 2691.90 | 2026-01-23 09:15:00 | 2737.90 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-01-22 12:30:00 | 2691.20 | 2026-01-23 09:15:00 | 2737.90 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-02-03 10:15:00 | 2402.70 | 2026-02-04 09:15:00 | 2445.70 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-02-11 11:15:00 | 2380.40 | 2026-02-12 13:15:00 | 2410.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-12 09:45:00 | 2379.70 | 2026-02-12 13:15:00 | 2410.40 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2295.10 | 2026-03-09 09:15:00 | 2180.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2295.10 | 2026-03-10 09:15:00 | 2255.80 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2026-03-20 14:00:00 | 2199.80 | 2026-03-24 13:15:00 | 2220.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2159.70 | 2026-03-24 13:15:00 | 2220.30 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-03-24 12:45:00 | 2203.50 | 2026-03-24 13:15:00 | 2220.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-04-01 11:30:00 | 2207.00 | 2026-04-01 12:15:00 | 2240.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2343.50 | 2026-04-21 09:15:00 | 2577.85 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 2491.40 | 2026-05-04 09:15:00 | 2486.60 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2026-04-27 12:00:00 | 2482.00 | 2026-05-04 09:15:00 | 2486.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2026-04-27 13:45:00 | 2491.40 | 2026-05-04 09:15:00 | 2486.60 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2026-04-27 15:00:00 | 2487.70 | 2026-05-04 13:15:00 | 2458.00 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2026-04-29 14:15:00 | 2449.20 | 2026-05-04 13:15:00 | 2458.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-04-29 14:45:00 | 2447.00 | 2026-05-04 13:15:00 | 2458.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-04-30 14:45:00 | 2448.90 | 2026-05-04 13:15:00 | 2458.00 | STOP_HIT | 1.00 | -0.37% |
