# Mazagoan Dock Shipbuilders Ltd. (MAZDOCK)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2656.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 57 |
| ALERT1 | 43 |
| ALERT2 | 43 |
| ALERT2_SKIP | 23 |
| ALERT3 | 110 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 54 |
| PARTIAL | 20 |
| TARGET_HIT | 9 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 28
- **Target hits / Stop hits / Partials:** 9 / 47 / 20
- **Avg / median % per leg:** 2.91% / 3.87%
- **Sum % (uncompounded):** 221.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 8 | 42.1% | 5 | 14 | 0 | 1.85% | 35.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 8 | 42.1% | 5 | 14 | 0 | 1.85% | 35.1% |
| SELL (all) | 57 | 40 | 70.2% | 4 | 33 | 20 | 3.27% | 186.2% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.84% | 11.3% |
| SELL @ 3rd Alert (retest2) | 53 | 36 | 67.9% | 4 | 31 | 18 | 3.30% | 174.9% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.84% | 11.3% |
| retest2 (combined) | 72 | 44 | 61.1% | 9 | 45 | 18 | 2.92% | 210.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2917.00 | 2906.36 | 2906.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 3184.30 | 3027.35 | 2977.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 14:15:00 | 3073.00 | 3089.57 | 3034.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 15:00:00 | 3073.00 | 3089.57 | 3034.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 3323.80 | 3406.69 | 3354.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:15:00 | 3310.20 | 3406.69 | 3354.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 3368.40 | 3399.03 | 3355.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 13:15:00 | 3431.00 | 3388.94 | 3358.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:15:00 | 3410.90 | 3368.30 | 3356.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:30:00 | 3396.70 | 3373.18 | 3361.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 3394.80 | 3373.18 | 3361.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 09:15:00 | 3736.37 | 3663.64 | 3622.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 3506.20 | 3626.71 | 3636.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 3484.00 | 3564.31 | 3602.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 3448.50 | 3430.35 | 3491.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:30:00 | 3470.00 | 3430.35 | 3491.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 3474.30 | 3448.66 | 3473.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 3506.00 | 3448.66 | 3473.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 3455.20 | 3449.97 | 3471.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 3455.20 | 3449.97 | 3471.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 3411.90 | 3426.43 | 3448.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 12:00:00 | 3375.90 | 3394.62 | 3408.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 3378.20 | 3394.70 | 3397.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 13:15:00 | 3207.11 | 3272.40 | 3319.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 13:15:00 | 3209.29 | 3272.40 | 3319.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 3170.90 | 3167.74 | 3207.94 | SL hit (close>ema200) qty=0.50 sl=3167.74 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 3290.50 | 3235.24 | 3229.68 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 3228.30 | 3255.08 | 3257.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 3215.30 | 3247.12 | 3253.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 3242.30 | 3235.44 | 3245.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 3242.30 | 3235.44 | 3245.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 3242.30 | 3235.44 | 3245.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 3242.30 | 3235.44 | 3245.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 3253.20 | 3237.17 | 3243.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 3253.20 | 3237.17 | 3243.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 3266.30 | 3243.00 | 3245.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 3305.00 | 3243.00 | 3245.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 3311.00 | 3256.60 | 3251.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 3355.30 | 3286.50 | 3266.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 3271.40 | 3310.71 | 3289.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 3271.40 | 3310.71 | 3289.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 3271.40 | 3310.71 | 3289.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 3271.40 | 3310.71 | 3289.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 3277.20 | 3304.01 | 3288.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:30:00 | 3284.80 | 3301.11 | 3288.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 3233.40 | 3280.38 | 3282.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 09:15:00 | 3233.40 | 3280.38 | 3282.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 13:15:00 | 3200.00 | 3245.27 | 3263.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 14:15:00 | 3172.80 | 3132.85 | 3162.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 3172.80 | 3132.85 | 3162.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 3172.80 | 3132.85 | 3162.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 3172.80 | 3132.85 | 3162.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 3185.30 | 3143.34 | 3164.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 3205.00 | 3143.34 | 3164.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 11:15:00 | 3268.00 | 3191.71 | 3183.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 09:15:00 | 3292.10 | 3236.61 | 3210.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 12:15:00 | 3260.10 | 3267.19 | 3247.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 12:45:00 | 3271.00 | 3267.19 | 3247.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 3260.00 | 3265.75 | 3248.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 3265.40 | 3265.75 | 3248.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 3267.00 | 3293.02 | 3295.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 3267.00 | 3293.02 | 3295.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 3263.00 | 3287.01 | 3292.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 3302.80 | 3275.57 | 3283.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 3302.80 | 3275.57 | 3283.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 3302.80 | 3275.57 | 3283.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 3302.80 | 3275.57 | 3283.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 3294.00 | 3279.26 | 3284.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 3306.00 | 3279.26 | 3284.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 3302.50 | 3289.78 | 3288.45 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 3278.50 | 3288.75 | 3289.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 3262.70 | 3281.02 | 3285.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 3139.10 | 3137.19 | 3174.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 09:30:00 | 3129.00 | 3137.19 | 3174.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 3118.00 | 3114.30 | 3129.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 3107.90 | 3115.32 | 3128.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 2952.51 | 2994.45 | 3039.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 2984.00 | 2968.53 | 3000.52 | SL hit (close>ema200) qty=0.50 sl=2968.53 alert=retest2 |

### Cycle 11 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 2779.10 | 2756.34 | 2755.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 2833.90 | 2771.85 | 2762.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 2801.20 | 2814.33 | 2792.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 2801.20 | 2814.33 | 2792.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2801.20 | 2814.33 | 2792.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 2801.20 | 2814.33 | 2792.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 2779.70 | 2807.40 | 2791.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 2779.80 | 2807.40 | 2791.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 2803.10 | 2806.54 | 2792.78 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 2762.00 | 2785.90 | 2786.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2743.90 | 2768.89 | 2777.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2780.20 | 2767.28 | 2775.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 2780.20 | 2767.28 | 2775.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2780.20 | 2767.28 | 2775.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2780.20 | 2767.28 | 2775.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2795.00 | 2772.82 | 2776.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 2750.00 | 2772.82 | 2776.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 2733.00 | 2716.61 | 2714.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 2733.00 | 2716.61 | 2714.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 2762.50 | 2729.85 | 2721.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 2726.80 | 2733.95 | 2725.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 2726.80 | 2733.95 | 2725.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2726.80 | 2733.95 | 2725.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 2726.80 | 2733.95 | 2725.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2721.50 | 2731.46 | 2725.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:15:00 | 2718.40 | 2731.46 | 2725.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 2717.80 | 2728.73 | 2724.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 2717.80 | 2728.73 | 2724.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 2722.00 | 2726.93 | 2724.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 2722.00 | 2726.93 | 2724.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 2722.30 | 2726.00 | 2724.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 2720.40 | 2726.00 | 2724.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 2718.00 | 2724.40 | 2723.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 2733.20 | 2724.40 | 2723.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 12:45:00 | 2726.80 | 2730.12 | 2727.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 2725.80 | 2728.90 | 2727.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:15:00 | 2752.20 | 2728.90 | 2727.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 2738.00 | 2730.72 | 2728.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 15:00:00 | 2772.10 | 2743.71 | 2738.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 2730.50 | 2758.20 | 2759.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 2730.50 | 2758.20 | 2759.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 2718.00 | 2745.97 | 2753.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 2759.20 | 2735.35 | 2745.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 2759.20 | 2735.35 | 2745.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 2759.20 | 2735.35 | 2745.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 2742.20 | 2738.98 | 2746.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 2694.10 | 2749.18 | 2749.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 2605.09 | 2653.30 | 2682.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 11:15:00 | 2653.20 | 2650.83 | 2676.38 | SL hit (close>ema200) qty=0.50 sl=2650.83 alert=retest2 |

### Cycle 15 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 2698.70 | 2658.48 | 2655.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 2714.20 | 2669.63 | 2661.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 2716.40 | 2720.77 | 2699.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 2716.40 | 2720.77 | 2699.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 2716.00 | 2718.81 | 2703.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 2723.70 | 2715.75 | 2703.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 2700.10 | 2712.62 | 2703.31 | SL hit (close<static) qty=1.00 sl=2703.50 alert=retest2 |

### Cycle 16 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 2678.00 | 2695.38 | 2697.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 2669.00 | 2688.49 | 2693.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 2685.00 | 2675.62 | 2683.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 15:15:00 | 2685.00 | 2675.62 | 2683.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 2685.00 | 2675.62 | 2683.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 2667.60 | 2675.62 | 2683.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2676.70 | 2675.83 | 2682.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 2677.60 | 2675.83 | 2682.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 2680.00 | 2676.67 | 2682.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:15:00 | 2684.00 | 2676.67 | 2682.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 2683.60 | 2678.05 | 2682.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 2683.60 | 2678.05 | 2682.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 2683.80 | 2679.20 | 2682.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 2683.80 | 2679.20 | 2682.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 2676.30 | 2678.62 | 2681.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 2664.80 | 2675.86 | 2680.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 2686.50 | 2677.96 | 2680.18 | SL hit (close>static) qty=1.00 sl=2685.90 alert=retest2 |

### Cycle 17 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 2689.00 | 2681.86 | 2681.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2722.00 | 2689.89 | 2684.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 2975.00 | 2981.01 | 2952.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:00:00 | 2975.00 | 2981.01 | 2952.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2997.40 | 2989.25 | 2972.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 3023.90 | 2993.43 | 2982.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 2944.40 | 2982.60 | 2983.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 2944.40 | 2982.60 | 2983.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 12:15:00 | 2924.00 | 2955.68 | 2969.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 2965.10 | 2951.44 | 2962.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 2965.10 | 2951.44 | 2962.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 2965.10 | 2951.44 | 2962.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:00:00 | 2920.30 | 2948.97 | 2954.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 2774.28 | 2827.05 | 2874.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 2766.70 | 2764.96 | 2798.96 | SL hit (close>ema200) qty=0.50 sl=2764.96 alert=retest2 |

### Cycle 19 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 2833.90 | 2814.42 | 2813.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 2873.90 | 2831.69 | 2821.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 2868.10 | 2869.09 | 2850.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 2870.10 | 2869.09 | 2850.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 2889.00 | 2882.25 | 2869.94 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 2822.30 | 2865.22 | 2866.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 2809.00 | 2840.25 | 2853.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 2838.10 | 2832.53 | 2845.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 2838.10 | 2832.53 | 2845.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2845.00 | 2835.03 | 2845.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 2833.50 | 2835.03 | 2845.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2839.10 | 2835.84 | 2845.04 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 2898.00 | 2858.64 | 2853.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 2907.60 | 2868.43 | 2858.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 2875.70 | 2877.68 | 2867.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 13:15:00 | 2875.70 | 2877.68 | 2867.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 2875.70 | 2877.68 | 2867.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:45:00 | 2869.50 | 2877.68 | 2867.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 2869.00 | 2875.95 | 2867.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 2869.00 | 2875.95 | 2867.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 2874.10 | 2875.58 | 2868.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 2839.50 | 2875.58 | 2868.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 2829.30 | 2866.32 | 2864.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:15:00 | 2826.00 | 2866.32 | 2864.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 2817.20 | 2856.50 | 2860.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 2815.50 | 2842.43 | 2852.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 2816.30 | 2809.31 | 2824.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 2816.30 | 2809.31 | 2824.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2830.40 | 2813.53 | 2824.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 2829.20 | 2813.53 | 2824.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 2833.40 | 2817.50 | 2825.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 2839.00 | 2817.50 | 2825.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 2860.30 | 2826.06 | 2828.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 2860.30 | 2826.06 | 2828.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 2859.00 | 2832.65 | 2831.37 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 2822.30 | 2839.39 | 2839.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 2805.70 | 2821.72 | 2827.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 2830.00 | 2823.38 | 2827.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 2830.00 | 2823.38 | 2827.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 2830.00 | 2823.38 | 2827.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 2809.90 | 2823.38 | 2827.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:30:00 | 2813.90 | 2823.07 | 2826.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:00:00 | 2818.70 | 2814.04 | 2819.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:00:00 | 2820.40 | 2815.31 | 2819.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 2818.00 | 2815.85 | 2819.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 2818.00 | 2815.85 | 2819.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 2811.50 | 2814.98 | 2818.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:30:00 | 2816.90 | 2814.98 | 2818.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 2852.00 | 2821.57 | 2820.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 2852.00 | 2821.57 | 2820.80 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 2808.80 | 2819.02 | 2819.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 2799.90 | 2815.19 | 2817.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 15:15:00 | 2780.30 | 2771.80 | 2785.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 2759.60 | 2769.36 | 2783.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2759.60 | 2769.36 | 2783.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:15:00 | 2744.40 | 2762.36 | 2776.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:45:00 | 2744.90 | 2751.40 | 2764.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:15:00 | 2747.40 | 2749.50 | 2760.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 2724.30 | 2740.38 | 2746.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 2607.18 | 2638.92 | 2672.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 2607.65 | 2638.92 | 2672.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 2610.03 | 2638.92 | 2672.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 2588.09 | 2638.92 | 2672.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 2641.00 | 2635.10 | 2661.65 | SL hit (close>ema200) qty=0.50 sl=2635.10 alert=retest2 |

### Cycle 27 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 2729.10 | 2676.86 | 2673.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 2747.60 | 2723.89 | 2704.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 2754.00 | 2757.74 | 2739.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:15:00 | 2771.00 | 2757.74 | 2739.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 2751.50 | 2756.49 | 2740.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 2788.00 | 2759.95 | 2749.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 2783.40 | 2766.94 | 2755.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 2780.00 | 2786.43 | 2785.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 2776.20 | 2784.28 | 2784.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 2776.20 | 2784.28 | 2784.78 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 2818.00 | 2790.29 | 2787.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 2830.00 | 2798.23 | 2791.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 2782.10 | 2814.94 | 2805.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 2782.10 | 2814.94 | 2805.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2782.10 | 2814.94 | 2805.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 2782.10 | 2814.94 | 2805.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 2765.70 | 2805.09 | 2802.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 2763.00 | 2805.09 | 2802.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 2772.70 | 2798.62 | 2799.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 2749.40 | 2779.58 | 2789.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 2671.20 | 2667.59 | 2698.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 2671.20 | 2667.59 | 2698.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 2697.90 | 2682.95 | 2695.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 2697.90 | 2682.95 | 2695.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 2691.60 | 2684.68 | 2694.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 2704.30 | 2684.68 | 2694.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 2700.00 | 2687.74 | 2695.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 2712.00 | 2687.74 | 2695.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 2701.80 | 2690.56 | 2695.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:30:00 | 2709.30 | 2690.56 | 2695.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2691.50 | 2683.17 | 2688.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 2695.00 | 2683.17 | 2688.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 2690.50 | 2684.63 | 2689.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:45:00 | 2692.50 | 2684.63 | 2689.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 2684.50 | 2684.61 | 2688.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:15:00 | 2678.50 | 2684.61 | 2688.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 2678.60 | 2683.41 | 2687.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 2680.20 | 2680.16 | 2684.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 2674.20 | 2678.73 | 2683.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 2678.40 | 2675.85 | 2680.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:00:00 | 2678.40 | 2675.85 | 2680.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 2674.50 | 2675.58 | 2680.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 2656.90 | 2670.95 | 2677.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:45:00 | 2663.30 | 2668.94 | 2675.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 2544.57 | 2591.34 | 2611.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 2544.67 | 2591.34 | 2611.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 2546.19 | 2591.34 | 2611.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 2540.49 | 2591.34 | 2611.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 2530.14 | 2591.34 | 2611.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:15:00 | 2524.05 | 2577.17 | 2603.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-09 09:15:00 | 2410.65 | 2511.25 | 2556.86 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 31 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 2406.80 | 2387.48 | 2386.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2442.00 | 2398.38 | 2391.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 2513.80 | 2519.40 | 2487.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:45:00 | 2516.80 | 2519.40 | 2487.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 2543.70 | 2560.17 | 2539.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 2543.00 | 2560.17 | 2539.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 2549.80 | 2558.10 | 2540.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 2593.00 | 2558.10 | 2540.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:15:00 | 2551.30 | 2569.11 | 2553.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 2481.60 | 2540.52 | 2543.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 2481.60 | 2540.52 | 2543.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 2460.50 | 2524.52 | 2536.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 2487.40 | 2480.15 | 2503.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 2487.40 | 2480.15 | 2503.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2487.40 | 2480.15 | 2503.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 2488.00 | 2480.15 | 2503.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2502.20 | 2484.56 | 2503.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 2501.00 | 2484.56 | 2503.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 2495.50 | 2486.75 | 2502.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 2475.50 | 2483.89 | 2495.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 2480.00 | 2480.63 | 2492.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:15:00 | 2477.20 | 2487.44 | 2488.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 2495.00 | 2489.70 | 2489.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 2495.00 | 2489.70 | 2489.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 2532.90 | 2498.34 | 2493.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 2502.70 | 2509.04 | 2500.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 2502.70 | 2509.04 | 2500.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 2502.70 | 2509.04 | 2500.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 2502.70 | 2509.04 | 2500.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2512.70 | 2509.77 | 2502.05 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 2481.20 | 2499.64 | 2500.49 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 2509.80 | 2499.81 | 2498.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-08 09:15:00 | 2527.00 | 2505.25 | 2501.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 2502.50 | 2504.98 | 2501.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 11:15:00 | 2502.50 | 2504.98 | 2501.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2502.50 | 2504.98 | 2501.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 2502.50 | 2504.98 | 2501.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 2496.90 | 2503.36 | 2501.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 2492.00 | 2503.36 | 2501.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 2490.10 | 2500.71 | 2500.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 2491.10 | 2500.71 | 2500.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 2485.00 | 2497.57 | 2498.94 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 2559.00 | 2507.14 | 2502.88 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 2474.00 | 2506.26 | 2507.31 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 2525.00 | 2507.25 | 2505.72 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 10:15:00 | 2492.80 | 2502.59 | 2503.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 11:15:00 | 2472.00 | 2496.47 | 2500.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 2486.40 | 2482.61 | 2490.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 10:15:00 | 2486.40 | 2482.61 | 2490.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2486.40 | 2482.61 | 2490.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 2492.00 | 2482.61 | 2490.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 2486.20 | 2483.33 | 2489.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 2494.80 | 2483.33 | 2489.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 2481.10 | 2482.88 | 2489.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:15:00 | 2470.80 | 2482.88 | 2489.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 2475.90 | 2479.22 | 2486.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 2347.26 | 2388.51 | 2419.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 2352.11 | 2388.51 | 2419.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 2338.00 | 2335.53 | 2366.32 | SL hit (close>ema200) qty=0.50 sl=2335.53 alert=retest2 |

### Cycle 41 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 2393.50 | 2339.21 | 2339.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 2424.00 | 2356.17 | 2346.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 2472.00 | 2554.96 | 2520.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 2472.00 | 2554.96 | 2520.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 2472.00 | 2554.96 | 2520.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 2472.00 | 2554.96 | 2520.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2470.00 | 2537.97 | 2515.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2446.70 | 2537.97 | 2515.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 2389.50 | 2490.06 | 2496.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 2380.00 | 2468.05 | 2486.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 2438.70 | 2409.86 | 2441.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 2438.70 | 2409.86 | 2441.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2447.60 | 2421.11 | 2441.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 2417.70 | 2449.79 | 2449.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 2470.30 | 2419.48 | 2412.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 2470.30 | 2419.48 | 2412.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 2477.50 | 2431.09 | 2418.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 2452.00 | 2456.75 | 2438.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:00:00 | 2452.00 | 2456.75 | 2438.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 2445.90 | 2454.58 | 2439.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:45:00 | 2439.80 | 2454.58 | 2439.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 2454.50 | 2454.56 | 2440.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:30:00 | 2442.30 | 2454.56 | 2440.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 2437.70 | 2450.62 | 2441.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 2437.70 | 2450.62 | 2441.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 2439.80 | 2448.46 | 2441.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:15:00 | 2434.00 | 2448.46 | 2441.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 2434.00 | 2445.56 | 2440.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 2417.70 | 2445.56 | 2440.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2403.10 | 2437.07 | 2437.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 2403.10 | 2437.07 | 2437.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 2408.40 | 2431.34 | 2434.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 2365.10 | 2405.93 | 2416.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 2358.90 | 2353.35 | 2372.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 2358.90 | 2353.35 | 2372.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 2366.60 | 2357.07 | 2370.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 2349.30 | 2360.44 | 2366.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 2395.00 | 2370.09 | 2368.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 2395.00 | 2370.09 | 2368.93 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 2344.70 | 2369.36 | 2371.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 2334.80 | 2362.45 | 2367.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 2381.20 | 2366.20 | 2369.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 2381.20 | 2366.20 | 2369.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 2381.20 | 2366.20 | 2369.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 2405.90 | 2366.20 | 2369.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 2372.90 | 2367.54 | 2369.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:30:00 | 2390.90 | 2367.54 | 2369.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 2386.90 | 2371.41 | 2371.05 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 2367.00 | 2370.53 | 2370.68 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 2375.30 | 2371.48 | 2371.10 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 2363.20 | 2369.83 | 2370.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 2356.00 | 2367.06 | 2369.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 2267.10 | 2265.70 | 2292.31 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 12:30:00 | 2252.40 | 2260.56 | 2283.31 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 2258.00 | 2251.20 | 2269.07 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 2262.00 | 2253.36 | 2268.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 2260.00 | 2253.36 | 2268.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2240.00 | 2233.39 | 2244.72 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2139.78 | 2233.39 | 2244.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2240.00 | 2233.39 | 2244.72 | SL hit (close>ema400) qty=0.50 sl=2233.39 alert=retest1 |

### Cycle 51 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 2281.90 | 2227.57 | 2220.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 2393.40 | 2260.73 | 2236.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2378.00 | 2433.61 | 2373.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2378.00 | 2433.61 | 2373.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2378.00 | 2433.61 | 2373.90 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 2361.00 | 2420.77 | 2423.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 2351.40 | 2391.63 | 2408.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 2331.60 | 2320.65 | 2352.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 2344.40 | 2320.65 | 2352.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2340.00 | 2325.30 | 2348.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 2313.90 | 2325.29 | 2344.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 2359.30 | 2333.24 | 2345.17 | SL hit (close>static) qty=1.00 sl=2352.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 2389.50 | 2354.76 | 2352.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 2422.50 | 2368.31 | 2359.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2407.70 | 2417.60 | 2392.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 10:15:00 | 2413.90 | 2417.60 | 2392.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 2384.50 | 2407.84 | 2393.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 2384.50 | 2407.84 | 2393.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 2368.00 | 2399.87 | 2391.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 2368.00 | 2399.87 | 2391.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 2365.20 | 2386.22 | 2386.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 2343.90 | 2374.24 | 2380.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 2243.90 | 2234.88 | 2274.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 2253.90 | 2234.88 | 2274.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 2251.10 | 2245.53 | 2270.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 2245.60 | 2268.11 | 2273.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 2133.32 | 2185.91 | 2221.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 2290.10 | 2144.63 | 2172.96 | SL hit (close>ema200) qty=0.50 sl=2144.63 alert=retest2 |

### Cycle 55 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 2307.50 | 2197.98 | 2193.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 13:15:00 | 2328.00 | 2309.33 | 2286.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 2421.40 | 2455.66 | 2430.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 2421.40 | 2455.66 | 2430.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2421.40 | 2455.66 | 2430.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 2472.10 | 2460.08 | 2437.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 13:15:00 | 2719.31 | 2689.84 | 2671.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 09:15:00 | 2624.00 | 2709.24 | 2715.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 10:15:00 | 2612.00 | 2689.79 | 2706.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 11:15:00 | 2637.10 | 2635.76 | 2661.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:00:00 | 2637.10 | 2635.76 | 2661.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 2651.90 | 2638.81 | 2653.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:30:00 | 2652.50 | 2638.81 | 2653.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 2655.50 | 2642.15 | 2653.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 2655.50 | 2642.15 | 2653.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 2649.70 | 2643.66 | 2652.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 2635.90 | 2642.71 | 2651.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:30:00 | 2643.70 | 2642.55 | 2650.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 15:00:00 | 2644.50 | 2642.94 | 2650.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:45:00 | 2641.00 | 2642.91 | 2648.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 2648.10 | 2643.95 | 2648.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-07 13:15:00 | 2670.00 | 2651.53 | 2651.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 13:15:00 | 2670.00 | 2651.53 | 2651.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 15:15:00 | 2688.00 | 2663.01 | 2656.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 2651.00 | 2663.76 | 2659.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 11:15:00 | 2651.00 | 2663.76 | 2659.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 2651.00 | 2663.76 | 2659.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 2651.00 | 2663.76 | 2659.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 2658.90 | 2662.79 | 2659.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:30:00 | 2656.50 | 2662.79 | 2659.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 2657.00 | 2661.63 | 2658.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 2656.10 | 2661.63 | 2658.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 2653.80 | 2660.06 | 2658.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 2653.80 | 2660.06 | 2658.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 2656.80 | 2659.41 | 2658.23 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 13:15:00 | 3431.00 | 2025-05-29 09:15:00 | 3736.37 | TARGET_HIT | 1.00 | 8.90% |
| BUY | retest2 | 2025-05-21 10:15:00 | 3410.90 | 2025-05-29 09:15:00 | 3734.28 | TARGET_HIT | 1.00 | 9.48% |
| BUY | retest2 | 2025-05-21 12:30:00 | 3396.70 | 2025-05-29 15:15:00 | 3774.10 | TARGET_HIT | 1.00 | 11.11% |
| BUY | retest2 | 2025-05-21 13:00:00 | 3394.80 | 2025-05-29 15:15:00 | 3751.99 | TARGET_HIT | 1.00 | 10.52% |
| SELL | retest2 | 2025-06-09 12:00:00 | 3375.90 | 2025-06-12 13:15:00 | 3207.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 09:15:00 | 3378.20 | 2025-06-12 13:15:00 | 3209.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 12:00:00 | 3375.90 | 2025-06-16 14:15:00 | 3170.90 | STOP_HIT | 0.50 | 6.07% |
| SELL | retest2 | 2025-06-11 09:15:00 | 3378.20 | 2025-06-16 14:15:00 | 3170.90 | STOP_HIT | 0.50 | 6.14% |
| BUY | retest2 | 2025-06-24 11:30:00 | 3284.80 | 2025-06-25 09:15:00 | 3233.40 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-02 14:15:00 | 3265.40 | 2025-07-07 15:15:00 | 3267.00 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-07-17 11:15:00 | 3107.90 | 2025-07-21 09:15:00 | 2952.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 11:15:00 | 3107.90 | 2025-07-22 09:15:00 | 2984.00 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-08-08 09:15:00 | 2750.00 | 2025-08-13 11:15:00 | 2733.00 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-08-18 09:15:00 | 2733.20 | 2025-08-22 11:15:00 | 2730.50 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-08-18 12:45:00 | 2726.80 | 2025-08-22 11:15:00 | 2730.50 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-08-18 13:30:00 | 2725.80 | 2025-08-22 11:15:00 | 2730.50 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-08-18 14:15:00 | 2752.20 | 2025-08-22 11:15:00 | 2730.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-20 15:00:00 | 2772.10 | 2025-08-22 11:15:00 | 2730.50 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-08-25 11:15:00 | 2742.20 | 2025-08-29 09:15:00 | 2605.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 11:15:00 | 2742.20 | 2025-08-29 11:15:00 | 2653.20 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2025-08-26 09:15:00 | 2694.10 | 2025-09-02 10:15:00 | 2698.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-09-04 09:45:00 | 2723.70 | 2025-09-04 10:15:00 | 2700.10 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-08 15:00:00 | 2664.80 | 2025-09-09 10:15:00 | 2686.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-22 10:15:00 | 3023.90 | 2025-09-23 09:15:00 | 2944.40 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-09-25 15:00:00 | 2920.30 | 2025-09-29 11:15:00 | 2774.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 15:00:00 | 2920.30 | 2025-09-30 15:15:00 | 2766.70 | STOP_HIT | 0.50 | 5.26% |
| SELL | retest2 | 2025-10-24 10:15:00 | 2809.90 | 2025-10-28 09:15:00 | 2852.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-10-24 12:30:00 | 2813.90 | 2025-10-28 09:15:00 | 2852.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-10-27 12:00:00 | 2818.70 | 2025-10-28 09:15:00 | 2852.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-27 13:00:00 | 2820.40 | 2025-10-28 09:15:00 | 2852.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-30 13:15:00 | 2744.40 | 2025-11-07 09:15:00 | 2607.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:45:00 | 2744.90 | 2025-11-07 09:15:00 | 2607.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 14:15:00 | 2747.40 | 2025-11-07 09:15:00 | 2610.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 2724.30 | 2025-11-07 09:15:00 | 2588.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 13:15:00 | 2744.40 | 2025-11-07 12:15:00 | 2641.00 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-10-31 10:45:00 | 2744.90 | 2025-11-07 12:15:00 | 2641.00 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2025-10-31 14:15:00 | 2747.40 | 2025-11-07 12:15:00 | 2641.00 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2025-11-04 09:15:00 | 2724.30 | 2025-11-07 12:15:00 | 2641.00 | STOP_HIT | 0.50 | 3.06% |
| BUY | retest2 | 2025-11-14 12:00:00 | 2788.00 | 2025-11-19 13:15:00 | 2776.20 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-11-14 15:00:00 | 2783.40 | 2025-11-19 13:15:00 | 2776.20 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-11-19 12:15:00 | 2780.00 | 2025-11-19 13:15:00 | 2776.20 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-11-28 12:15:00 | 2678.50 | 2025-12-08 10:15:00 | 2544.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 13:00:00 | 2678.60 | 2025-12-08 10:15:00 | 2544.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:30:00 | 2680.20 | 2025-12-08 10:15:00 | 2546.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 2674.20 | 2025-12-08 10:15:00 | 2540.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 09:30:00 | 2656.90 | 2025-12-08 10:15:00 | 2530.14 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2025-12-02 10:45:00 | 2663.30 | 2025-12-08 11:15:00 | 2524.05 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2025-11-28 12:15:00 | 2678.50 | 2025-12-09 09:15:00 | 2410.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-28 13:00:00 | 2678.60 | 2025-12-09 09:15:00 | 2410.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-01 09:30:00 | 2680.20 | 2025-12-09 09:15:00 | 2412.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 2674.20 | 2025-12-09 09:15:00 | 2406.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-02 09:30:00 | 2656.90 | 2025-12-11 09:15:00 | 2463.00 | STOP_HIT | 0.50 | 7.30% |
| SELL | retest2 | 2025-12-02 10:45:00 | 2663.30 | 2025-12-11 09:15:00 | 2463.00 | STOP_HIT | 0.50 | 7.52% |
| BUY | retest2 | 2025-12-29 09:15:00 | 2593.00 | 2025-12-30 09:15:00 | 2481.60 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2025-12-29 13:15:00 | 2551.30 | 2025-12-30 09:15:00 | 2481.60 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2026-01-01 09:30:00 | 2475.50 | 2026-01-02 15:15:00 | 2495.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-01-01 11:30:00 | 2480.00 | 2026-01-02 15:15:00 | 2495.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-01-02 14:15:00 | 2477.20 | 2026-01-02 15:15:00 | 2495.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2470.80 | 2026-01-20 13:15:00 | 2347.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 2475.90 | 2026-01-20 13:15:00 | 2352.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2470.80 | 2026-01-21 15:15:00 | 2338.00 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2026-01-14 15:00:00 | 2475.90 | 2026-01-21 15:15:00 | 2338.00 | STOP_HIT | 0.50 | 5.57% |
| SELL | retest2 | 2026-02-04 09:15:00 | 2417.70 | 2026-02-09 10:15:00 | 2470.30 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-02-18 10:15:00 | 2349.30 | 2026-02-18 13:15:00 | 2395.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest1 | 2026-02-25 12:30:00 | 2252.40 | 2026-03-02 09:15:00 | 2139.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-25 12:30:00 | 2252.40 | 2026-03-02 09:15:00 | 2240.00 | STOP_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2026-02-26 11:00:00 | 2258.00 | 2026-03-02 09:15:00 | 2145.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-26 11:00:00 | 2258.00 | 2026-03-02 09:15:00 | 2240.00 | STOP_HIT | 0.50 | 0.80% |
| SELL | retest2 | 2026-03-17 12:15:00 | 2313.90 | 2026-03-17 13:15:00 | 2359.30 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-03-27 09:15:00 | 2245.60 | 2026-03-30 09:15:00 | 2133.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:15:00 | 2245.60 | 2026-04-01 09:15:00 | 2290.10 | STOP_HIT | 0.50 | -1.98% |
| BUY | retest2 | 2026-04-13 11:30:00 | 2472.10 | 2026-04-23 13:15:00 | 2719.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-06 12:30:00 | 2635.90 | 2026-05-07 13:15:00 | 2670.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-05-06 13:30:00 | 2643.70 | 2026-05-07 13:15:00 | 2670.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-05-06 15:00:00 | 2644.50 | 2026-05-07 13:15:00 | 2670.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-05-07 09:45:00 | 2641.00 | 2026-05-07 13:15:00 | 2670.00 | STOP_HIT | 1.00 | -1.10% |
