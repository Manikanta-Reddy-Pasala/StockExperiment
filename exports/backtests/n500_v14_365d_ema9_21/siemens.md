# Siemens Ltd. (SIEMENS)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 3838.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 72 |
| ALERT1 | 46 |
| ALERT2 | 45 |
| ALERT2_SKIP | 24 |
| ALERT3 | 145 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 61 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 45
- **Target hits / Stop hits / Partials:** 0 / 62 / 4
- **Avg / median % per leg:** 0.09% / -0.45%
- **Sum % (uncompounded):** 6.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 8 | 27.6% | 0 | 28 | 1 | 0.15% | 4.3% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.43% | 8.9% |
| BUY @ 3rd Alert (retest2) | 27 | 6 | 22.2% | 0 | 27 | 0 | -0.17% | -4.6% |
| SELL (all) | 37 | 13 | 35.1% | 0 | 34 | 3 | 0.05% | 1.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 13 | 35.1% | 0 | 34 | 3 | 0.05% | 1.8% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.43% | 8.9% |
| retest2 (combined) | 64 | 19 | 29.7% | 0 | 61 | 3 | -0.04% | -2.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 2904.40 | 2891.60 | 2891.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 2931.40 | 2901.63 | 2895.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 2912.00 | 2915.38 | 2906.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 15:15:00 | 2936.60 | 2915.38 | 2906.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 09:15:00 | 3083.43 | 3014.87 | 2986.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 3049.90 | 3055.82 | 3026.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 3049.90 | 3055.82 | 3026.36 | SL hit (close<ema200) qty=0.50 sl=3055.82 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 3036.00 | 3055.82 | 3026.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 3094.80 | 3092.91 | 3063.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:45:00 | 3148.00 | 3113.73 | 3075.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 13:00:00 | 3147.00 | 3132.75 | 3091.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 3145.60 | 3138.54 | 3107.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 3163.50 | 3138.54 | 3107.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 3256.80 | 3269.25 | 3255.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 3256.80 | 3269.25 | 3255.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 3266.80 | 3268.76 | 3256.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 3259.90 | 3268.76 | 3256.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 3253.70 | 3267.48 | 3257.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 3253.70 | 3267.48 | 3257.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 3242.00 | 3262.39 | 3256.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 3281.70 | 3262.39 | 3256.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 3267.00 | 3312.57 | 3318.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 3267.00 | 3312.57 | 3318.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 3267.00 | 3312.57 | 3318.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 3267.00 | 3312.57 | 3318.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 3267.00 | 3312.57 | 3318.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 3267.00 | 3312.57 | 3318.44 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 12:15:00 | 3318.80 | 3309.65 | 3308.70 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 3286.50 | 3305.40 | 3307.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 10:15:00 | 3277.00 | 3299.72 | 3304.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 3300.80 | 3299.60 | 3303.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 14:00:00 | 3300.80 | 3299.60 | 3303.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 3309.90 | 3301.66 | 3303.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 3309.90 | 3301.66 | 3303.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 3309.00 | 3303.13 | 3304.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 3320.40 | 3303.13 | 3304.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 3310.40 | 3304.58 | 3304.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:15:00 | 3304.80 | 3304.58 | 3304.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 3306.60 | 3302.11 | 3302.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 3309.70 | 3303.62 | 3302.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 3309.70 | 3303.62 | 3302.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 3309.70 | 3303.62 | 3302.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 3344.00 | 3312.08 | 3307.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 3346.30 | 3366.77 | 3356.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 10:15:00 | 3346.30 | 3366.77 | 3356.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3346.30 | 3366.77 | 3356.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 3346.30 | 3366.77 | 3356.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3350.90 | 3363.60 | 3356.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:30:00 | 3366.00 | 3365.26 | 3357.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 3307.90 | 3348.60 | 3351.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 3307.90 | 3348.60 | 3351.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 3291.40 | 3326.75 | 3340.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 3292.50 | 3269.65 | 3287.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 3292.50 | 3269.65 | 3287.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 3292.50 | 3269.65 | 3287.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 3292.50 | 3269.65 | 3287.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 3277.60 | 3271.24 | 3286.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 3281.00 | 3271.24 | 3286.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 3278.60 | 3273.15 | 3284.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 3281.80 | 3273.15 | 3284.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 3330.90 | 3284.28 | 3286.75 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 3318.10 | 3291.04 | 3289.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 11:15:00 | 3358.20 | 3304.48 | 3295.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 3313.90 | 3317.47 | 3307.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 3313.90 | 3317.47 | 3307.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 3313.90 | 3317.47 | 3307.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 3313.40 | 3317.47 | 3307.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 3307.40 | 3315.46 | 3307.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 3315.00 | 3315.46 | 3307.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 3316.60 | 3315.68 | 3308.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 13:15:00 | 3328.70 | 3315.68 | 3308.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 3284.60 | 3324.10 | 3317.47 | SL hit (close<static) qty=1.00 sl=3303.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 3272.20 | 3310.14 | 3312.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 3208.60 | 3278.82 | 3295.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 3166.50 | 3136.33 | 3182.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 10:00:00 | 3166.50 | 3136.33 | 3182.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 3200.20 | 3149.11 | 3184.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 3196.90 | 3149.11 | 3184.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 3226.00 | 3164.49 | 3188.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:45:00 | 3226.00 | 3164.49 | 3188.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 3115.90 | 3156.16 | 3176.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:15:00 | 3113.60 | 3156.16 | 3176.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:00:00 | 3105.60 | 3140.40 | 3165.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 3254.00 | 3173.43 | 3164.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 3254.00 | 3173.43 | 3164.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 3254.00 | 3173.43 | 3164.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 3276.70 | 3232.37 | 3205.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 11:15:00 | 3308.90 | 3315.90 | 3286.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 11:30:00 | 3312.80 | 3315.90 | 3286.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 3286.30 | 3309.98 | 3286.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 3280.50 | 3309.98 | 3286.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 3294.00 | 3306.78 | 3287.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 3270.30 | 3306.78 | 3287.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 3289.20 | 3303.26 | 3287.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:45:00 | 3283.10 | 3303.26 | 3287.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 3286.20 | 3299.85 | 3287.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 3304.70 | 3299.85 | 3287.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 3263.20 | 3295.84 | 3294.35 | SL hit (close<static) qty=1.00 sl=3284.10 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 3255.30 | 3287.73 | 3290.80 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 3318.80 | 3286.22 | 3285.60 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 3264.00 | 3285.91 | 3287.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 11:15:00 | 3258.00 | 3280.33 | 3285.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 3144.80 | 3138.92 | 3173.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:30:00 | 3139.00 | 3138.92 | 3173.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 3153.80 | 3125.97 | 3152.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 3153.80 | 3125.97 | 3152.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 3150.00 | 3130.78 | 3151.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 3139.90 | 3130.78 | 3151.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3127.10 | 3130.04 | 3149.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:30:00 | 3092.20 | 3131.53 | 3142.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:00:00 | 3107.40 | 3126.70 | 3139.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 13:45:00 | 3104.20 | 3094.42 | 3108.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 15:15:00 | 3105.70 | 3098.32 | 3109.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 3105.70 | 3099.79 | 3108.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 3126.00 | 3099.79 | 3108.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 3119.90 | 3103.82 | 3109.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 3137.30 | 3103.82 | 3109.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 3134.90 | 3110.03 | 3112.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:45:00 | 3127.60 | 3110.03 | 3112.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 3136.30 | 3115.29 | 3114.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 3136.30 | 3115.29 | 3114.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 3136.30 | 3115.29 | 3114.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 3136.30 | 3115.29 | 3114.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 11:15:00 | 3136.30 | 3115.29 | 3114.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 3152.00 | 3130.58 | 3123.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 3127.80 | 3140.22 | 3132.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 3127.80 | 3140.22 | 3132.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3127.80 | 3140.22 | 3132.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 3127.80 | 3140.22 | 3132.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 3144.00 | 3140.98 | 3133.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 3146.60 | 3141.98 | 3134.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:30:00 | 3147.60 | 3142.61 | 3136.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 3146.00 | 3142.61 | 3136.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 14:30:00 | 3149.30 | 3142.57 | 3136.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 3134.40 | 3140.94 | 3136.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 3133.60 | 3140.94 | 3136.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 3123.60 | 3137.47 | 3135.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 3123.60 | 3137.47 | 3135.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 3119.20 | 3133.82 | 3133.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 3119.20 | 3133.82 | 3133.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 3119.20 | 3133.82 | 3133.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 3119.20 | 3133.82 | 3133.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 3119.20 | 3133.82 | 3133.91 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 3141.50 | 3134.70 | 3133.81 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 3107.40 | 3129.24 | 3131.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 3086.30 | 3115.71 | 3122.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 3063.00 | 3061.14 | 3078.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 3063.00 | 3061.14 | 3078.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 3058.10 | 3060.35 | 3075.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 3067.30 | 3060.35 | 3075.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 3066.50 | 3058.04 | 3069.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 3066.80 | 3058.04 | 3069.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 3071.70 | 3060.78 | 3069.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 3069.80 | 3060.78 | 3069.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 3061.30 | 3060.88 | 3068.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 3059.30 | 3060.88 | 3068.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 3044.30 | 3057.56 | 3066.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 3039.90 | 3057.56 | 3066.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 3073.60 | 3062.74 | 3065.64 | SL hit (close>static) qty=1.00 sl=3073.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 3038.50 | 3063.82 | 3065.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 3036.40 | 3048.31 | 3056.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:30:00 | 3028.20 | 3000.11 | 3010.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 3040.40 | 3012.89 | 3014.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 3040.40 | 3012.89 | 3014.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 3040.40 | 3018.39 | 3016.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 3040.40 | 3018.39 | 3016.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 3040.40 | 3018.39 | 3016.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 3040.40 | 3018.39 | 3016.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 3088.00 | 3035.91 | 3025.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 3071.90 | 3077.28 | 3059.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 3071.90 | 3077.28 | 3059.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 3071.90 | 3077.28 | 3059.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:15:00 | 3055.80 | 3077.28 | 3059.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 3073.30 | 3076.48 | 3060.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:45:00 | 3070.00 | 3076.48 | 3060.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 3079.30 | 3075.51 | 3064.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:15:00 | 3107.20 | 3075.51 | 3064.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 3032.40 | 3080.94 | 3077.89 | SL hit (close<static) qty=1.00 sl=3064.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 3030.00 | 3070.75 | 3073.53 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 3124.00 | 3077.92 | 3076.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3132.00 | 3107.36 | 3093.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 3120.00 | 3129.87 | 3116.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 3120.00 | 3129.87 | 3116.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 3120.00 | 3129.87 | 3116.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 3120.00 | 3129.87 | 3116.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 3136.20 | 3131.14 | 3118.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:45:00 | 3113.50 | 3131.14 | 3118.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 3135.70 | 3139.93 | 3127.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 3129.90 | 3139.93 | 3127.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 3158.70 | 3159.17 | 3145.18 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 14:15:00 | 3113.50 | 3139.93 | 3140.28 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 12:15:00 | 3152.70 | 3138.93 | 3137.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 3220.60 | 3161.34 | 3149.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 12:15:00 | 3200.30 | 3205.86 | 3186.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:00:00 | 3200.30 | 3205.86 | 3186.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 3173.30 | 3198.41 | 3186.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 3173.30 | 3198.41 | 3186.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 3162.60 | 3191.25 | 3184.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 3160.60 | 3191.25 | 3184.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 3197.00 | 3190.57 | 3185.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 3199.80 | 3190.57 | 3185.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 3166.00 | 3185.09 | 3184.51 | SL hit (close<static) qty=1.00 sl=3176.10 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 3162.00 | 3180.47 | 3182.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 3091.70 | 3162.72 | 3174.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 3082.00 | 3060.08 | 3085.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 3082.00 | 3060.08 | 3085.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 3081.00 | 3064.26 | 3085.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 3069.90 | 3064.26 | 3085.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:30:00 | 3074.40 | 3068.05 | 3082.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 3099.90 | 3076.68 | 3082.65 | SL hit (close>static) qty=1.00 sl=3090.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 3099.90 | 3076.68 | 3082.65 | SL hit (close>static) qty=1.00 sl=3090.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 3132.00 | 3094.05 | 3089.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 3139.80 | 3108.86 | 3097.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 3200.00 | 3202.57 | 3179.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:45:00 | 3203.10 | 3202.57 | 3179.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 3165.50 | 3191.72 | 3180.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 3165.50 | 3191.72 | 3180.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 3152.00 | 3183.78 | 3177.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 3152.00 | 3183.78 | 3177.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 3144.30 | 3168.49 | 3171.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 3106.20 | 3156.03 | 3165.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 3123.30 | 3121.75 | 3142.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 10:00:00 | 3123.30 | 3121.75 | 3142.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 3132.30 | 3123.23 | 3137.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:45:00 | 3138.30 | 3123.23 | 3137.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 3122.00 | 3121.65 | 3134.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 3132.00 | 3121.65 | 3134.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3108.10 | 3117.97 | 3130.38 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 3175.80 | 3138.00 | 3136.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 3192.40 | 3148.88 | 3141.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 3148.70 | 3170.17 | 3157.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 12:15:00 | 3148.70 | 3170.17 | 3157.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 3148.70 | 3170.17 | 3157.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 3148.00 | 3170.17 | 3157.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 3144.30 | 3165.00 | 3156.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 3144.30 | 3165.00 | 3156.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 3249.40 | 3207.74 | 3191.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 3255.70 | 3207.74 | 3191.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 3282.40 | 3312.46 | 3314.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 3282.40 | 3312.46 | 3314.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 3267.00 | 3295.92 | 3305.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 3131.30 | 3122.47 | 3156.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 3131.30 | 3122.47 | 3156.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 3129.60 | 3117.98 | 3128.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 3129.60 | 3117.98 | 3128.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 3121.00 | 3118.58 | 3127.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 3142.90 | 3118.58 | 3127.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 3137.60 | 3122.39 | 3128.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 3154.40 | 3122.39 | 3128.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 3109.40 | 3119.79 | 3126.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:30:00 | 3105.80 | 3116.59 | 3124.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 3138.00 | 3127.10 | 3125.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 3138.00 | 3127.10 | 3125.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 3162.10 | 3134.55 | 3129.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 3228.00 | 3236.47 | 3212.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 3228.00 | 3236.47 | 3212.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3228.00 | 3236.47 | 3212.46 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 3182.70 | 3211.18 | 3211.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 11:15:00 | 3177.00 | 3204.34 | 3208.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 3097.90 | 3090.74 | 3122.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:30:00 | 3106.00 | 3090.74 | 3122.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 3107.30 | 3091.62 | 3110.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 3105.20 | 3091.62 | 3110.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 3100.00 | 3093.29 | 3109.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 3104.00 | 3093.29 | 3109.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 3112.00 | 3097.04 | 3109.48 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 3133.40 | 3116.73 | 3115.75 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 3109.80 | 3117.32 | 3117.34 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 3142.20 | 3122.30 | 3119.60 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 3109.50 | 3118.77 | 3119.02 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3125.70 | 3120.16 | 3119.63 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 3111.40 | 3119.28 | 3119.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 14:15:00 | 3098.50 | 3113.60 | 3116.65 | Break + close below crossover candle low |

### Cycle 35 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 3151.50 | 3116.69 | 3116.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 10:15:00 | 3164.50 | 3149.44 | 3137.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 12:15:00 | 3145.20 | 3151.00 | 3140.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 3145.20 | 3151.00 | 3140.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 3145.20 | 3151.00 | 3140.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 3145.20 | 3151.00 | 3140.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 3140.40 | 3148.88 | 3140.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 3137.70 | 3148.88 | 3140.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3154.90 | 3150.08 | 3141.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 3137.30 | 3150.08 | 3141.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3149.10 | 3151.47 | 3143.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 3149.10 | 3151.47 | 3143.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3146.40 | 3154.99 | 3150.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 3146.40 | 3154.99 | 3150.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 3148.60 | 3153.71 | 3150.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 3149.30 | 3153.71 | 3150.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 3117.10 | 3146.39 | 3147.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 3104.80 | 3138.07 | 3143.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 3145.80 | 3126.83 | 3134.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 3145.80 | 3126.83 | 3134.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 3145.80 | 3126.83 | 3134.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 3157.40 | 3126.83 | 3134.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 3147.50 | 3130.97 | 3135.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:30:00 | 3142.00 | 3138.02 | 3138.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 15:15:00 | 3165.00 | 3143.74 | 3140.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 3165.00 | 3143.74 | 3140.86 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 3121.80 | 3139.20 | 3140.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 3112.00 | 3128.44 | 3134.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 3116.80 | 3114.20 | 3123.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 3116.80 | 3114.20 | 3123.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3116.80 | 3114.20 | 3123.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 3122.80 | 3114.20 | 3123.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 3135.00 | 3118.36 | 3124.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 3135.00 | 3118.36 | 3124.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 3118.50 | 3118.39 | 3124.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 3108.00 | 3120.85 | 3123.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 3106.70 | 3117.44 | 3120.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 3107.90 | 3053.36 | 3046.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 3107.90 | 3053.36 | 3046.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 3107.90 | 3053.36 | 3046.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 3179.60 | 3100.63 | 3081.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 3214.80 | 3226.03 | 3188.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:00:00 | 3214.80 | 3226.03 | 3188.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 3203.00 | 3221.43 | 3189.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 3203.00 | 3221.43 | 3189.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 3192.70 | 3214.13 | 3208.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 3192.70 | 3214.13 | 3208.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 3161.90 | 3203.68 | 3203.99 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 3276.00 | 3194.57 | 3185.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 3305.00 | 3216.66 | 3196.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 3283.20 | 3283.51 | 3248.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:30:00 | 3282.00 | 3283.51 | 3248.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 3288.00 | 3292.93 | 3282.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 3287.00 | 3292.93 | 3282.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 3287.90 | 3291.93 | 3282.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 3276.00 | 3291.93 | 3282.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 3324.30 | 3341.09 | 3323.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 3324.30 | 3341.09 | 3323.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 3341.30 | 3341.13 | 3324.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:45:00 | 3352.10 | 3339.76 | 3330.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 3349.00 | 3339.57 | 3332.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 3358.00 | 3339.57 | 3332.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:30:00 | 3350.00 | 3348.57 | 3339.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 3301.40 | 3339.13 | 3336.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 3301.40 | 3339.13 | 3336.41 | SL hit (close<static) qty=1.00 sl=3308.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 3301.40 | 3339.13 | 3336.41 | SL hit (close<static) qty=1.00 sl=3308.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 3301.40 | 3339.13 | 3336.41 | SL hit (close<static) qty=1.00 sl=3308.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 3301.40 | 3339.13 | 3336.41 | SL hit (close<static) qty=1.00 sl=3308.80 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 3301.40 | 3339.13 | 3336.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 12:15:00 | 3303.90 | 3332.09 | 3333.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 3267.30 | 3303.87 | 3317.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 11:15:00 | 3163.00 | 3156.94 | 3198.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 11:45:00 | 3162.90 | 3156.94 | 3198.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 3192.90 | 3176.14 | 3185.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 3192.90 | 3176.14 | 3185.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 3197.00 | 3180.31 | 3186.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 3193.30 | 3180.31 | 3186.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 3158.90 | 3177.31 | 3184.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 3170.60 | 3177.31 | 3184.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 3172.80 | 3152.83 | 3165.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 3172.80 | 3152.83 | 3165.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 3169.10 | 3156.08 | 3165.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 12:30:00 | 3153.00 | 3153.87 | 3163.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 3149.50 | 3154.00 | 3161.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 3151.80 | 3154.56 | 3160.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 3144.90 | 3151.55 | 3157.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 3161.70 | 3153.62 | 3157.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 3161.70 | 3153.62 | 3157.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 3165.70 | 3156.04 | 3158.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 3157.90 | 3156.04 | 3158.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3155.70 | 3155.97 | 3158.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 3145.00 | 3155.97 | 3158.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 3147.60 | 3149.83 | 3154.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 3131.30 | 3114.70 | 3113.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 3131.30 | 3114.70 | 3113.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 3131.30 | 3114.70 | 3113.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 3131.30 | 3114.70 | 3113.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 3131.30 | 3114.70 | 3113.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 3131.30 | 3114.70 | 3113.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 3131.30 | 3114.70 | 3113.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 3144.30 | 3124.11 | 3118.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 3125.00 | 3126.17 | 3120.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 3125.00 | 3126.17 | 3120.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 3125.00 | 3126.17 | 3120.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 3142.60 | 3130.86 | 3126.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 14:15:00 | 3095.00 | 3123.51 | 3124.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 3095.00 | 3123.51 | 3124.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 3090.00 | 3116.81 | 3121.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 3049.00 | 3034.75 | 3055.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:30:00 | 3046.40 | 3034.75 | 3055.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 3060.50 | 3039.90 | 3056.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 3059.00 | 3039.90 | 3056.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 3076.50 | 3047.22 | 3057.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 3076.60 | 3047.22 | 3057.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 3070.40 | 3059.88 | 3061.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 3077.40 | 3059.88 | 3061.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 3060.00 | 3059.91 | 3061.27 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 3073.20 | 3064.18 | 3063.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 3094.00 | 3072.40 | 3067.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 3071.50 | 3078.96 | 3072.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 3071.50 | 3078.96 | 3072.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 3071.50 | 3078.96 | 3072.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:45:00 | 3068.50 | 3078.96 | 3072.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 3092.00 | 3081.57 | 3074.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 15:00:00 | 3096.90 | 3085.61 | 3077.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 3096.10 | 3092.73 | 3088.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 3096.90 | 3088.98 | 3087.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 11:30:00 | 3095.00 | 3109.82 | 3105.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 3093.40 | 3102.85 | 3103.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 3093.40 | 3102.85 | 3103.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 3093.40 | 3102.85 | 3103.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 3093.40 | 3102.85 | 3103.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 3093.40 | 3102.85 | 3103.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 3010.00 | 3084.28 | 3094.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 3069.00 | 3067.74 | 3084.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 3069.00 | 3067.74 | 3084.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3069.00 | 3067.74 | 3084.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 3069.00 | 3067.74 | 3084.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 3065.20 | 3067.23 | 3082.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:30:00 | 3053.60 | 3071.53 | 3080.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 3059.90 | 3071.53 | 3080.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 2906.90 | 2967.17 | 2979.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2940.10 | 2936.88 | 2955.28 | SL hit (close>ema200) qty=0.50 sl=2936.88 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 2900.92 | 2924.34 | 2940.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 2886.10 | 2875.93 | 2902.07 | SL hit (close>ema200) qty=0.50 sl=2875.93 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 2929.00 | 2899.45 | 2899.42 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 2889.50 | 2902.60 | 2903.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 2858.40 | 2893.76 | 2899.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 2895.00 | 2888.46 | 2895.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 2895.00 | 2888.46 | 2895.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 2895.00 | 2888.46 | 2895.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 2895.00 | 2888.46 | 2895.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 2893.10 | 2889.39 | 2895.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 2921.00 | 2889.39 | 2895.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 2891.70 | 2889.85 | 2894.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 2912.60 | 2889.85 | 2894.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 2938.80 | 2899.64 | 2898.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 2980.00 | 2915.71 | 2906.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 3046.90 | 3069.17 | 3036.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 3046.90 | 3069.17 | 3036.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 3046.90 | 3069.17 | 3036.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 3040.90 | 3069.17 | 3036.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 3028.30 | 3061.13 | 3038.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 3028.30 | 3061.13 | 3038.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 3037.50 | 3056.40 | 3038.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 13:30:00 | 3064.90 | 3051.30 | 3037.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 2985.00 | 3026.71 | 3028.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 2985.00 | 3026.71 | 3028.02 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 3068.10 | 3034.41 | 3030.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 3092.50 | 3046.03 | 3036.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 3257.30 | 3266.63 | 3212.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 3256.50 | 3266.63 | 3212.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3197.40 | 3273.43 | 3258.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:45:00 | 3197.40 | 3273.43 | 3258.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 3183.80 | 3255.51 | 3251.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:15:00 | 3139.00 | 3255.51 | 3251.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 3139.00 | 3232.21 | 3241.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 09:15:00 | 3111.80 | 3208.12 | 3229.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 3134.60 | 3119.97 | 3141.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 3134.60 | 3119.97 | 3141.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 3134.60 | 3119.97 | 3141.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 3137.20 | 3119.97 | 3141.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 3133.20 | 3122.61 | 3141.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 3138.80 | 3122.61 | 3141.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 3146.10 | 3129.45 | 3141.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 3120.80 | 3137.27 | 3142.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 3133.40 | 3139.44 | 3142.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 3157.60 | 3142.86 | 3143.66 | SL hit (close>static) qty=1.00 sl=3154.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 3157.60 | 3142.86 | 3143.66 | SL hit (close>static) qty=1.00 sl=3154.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 3102.80 | 3143.57 | 3143.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 3175.80 | 3140.70 | 3141.48 | SL hit (close>static) qty=1.00 sl=3154.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 13:45:00 | 3129.30 | 3139.35 | 3140.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 3159.50 | 3131.94 | 3135.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 3159.50 | 3131.94 | 3135.04 | SL hit (close>static) qty=1.00 sl=3154.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 3159.50 | 3131.94 | 3135.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 3172.80 | 3140.11 | 3138.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 3200.00 | 3152.09 | 3144.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 11:15:00 | 3166.10 | 3178.22 | 3163.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 11:15:00 | 3166.10 | 3178.22 | 3163.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 3166.10 | 3178.22 | 3163.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 3166.10 | 3178.22 | 3163.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 3178.00 | 3181.91 | 3171.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 3172.70 | 3181.91 | 3171.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 3186.00 | 3189.50 | 3180.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 3194.10 | 3189.50 | 3180.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 3190.20 | 3189.64 | 3181.36 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 3147.20 | 3173.05 | 3175.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 3127.20 | 3163.88 | 3170.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 3174.90 | 3149.88 | 3161.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 3174.90 | 3149.88 | 3161.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 3174.90 | 3149.88 | 3161.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:15:00 | 3191.70 | 3149.88 | 3161.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 3226.20 | 3165.14 | 3167.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 3218.90 | 3165.14 | 3167.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 3269.10 | 3185.93 | 3176.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 13:15:00 | 3304.80 | 3221.73 | 3195.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 14:15:00 | 3206.70 | 3218.73 | 3196.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 15:00:00 | 3206.70 | 3218.73 | 3196.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 3190.20 | 3213.02 | 3195.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 3247.30 | 3213.02 | 3195.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 3321.90 | 3344.52 | 3344.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 15:15:00 | 3321.90 | 3344.52 | 3344.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 3251.00 | 3325.81 | 3336.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 11:15:00 | 3221.50 | 3210.53 | 3252.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:00:00 | 3221.50 | 3210.53 | 3252.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 3285.10 | 3225.30 | 3242.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 3284.70 | 3225.30 | 3242.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 3269.20 | 3234.08 | 3245.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:15:00 | 3253.10 | 3234.08 | 3245.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 3296.20 | 3246.50 | 3249.99 | SL hit (close>static) qty=1.00 sl=3292.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 3310.10 | 3259.22 | 3255.46 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3195.60 | 3254.48 | 3255.80 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 3304.80 | 3248.90 | 3244.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 3335.70 | 3293.22 | 3280.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 3288.00 | 3307.13 | 3293.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 3288.00 | 3307.13 | 3293.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3288.00 | 3307.13 | 3293.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:45:00 | 3292.80 | 3307.13 | 3293.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 3257.50 | 3297.20 | 3290.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 3257.50 | 3297.20 | 3290.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 3278.10 | 3293.38 | 3289.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:15:00 | 3254.60 | 3293.38 | 3289.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3231.50 | 3281.01 | 3284.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 3210.00 | 3260.55 | 3273.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 11:15:00 | 3160.30 | 3158.31 | 3191.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 11:45:00 | 3173.00 | 3158.31 | 3191.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 3193.00 | 3169.79 | 3184.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 3188.80 | 3169.79 | 3184.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 3206.10 | 3177.06 | 3186.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:30:00 | 3215.00 | 3177.06 | 3186.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 3222.60 | 3194.80 | 3192.91 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 3149.80 | 3191.67 | 3192.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 3117.90 | 3170.14 | 3182.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 3172.90 | 3130.12 | 3153.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 3172.90 | 3130.12 | 3153.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 3172.90 | 3130.12 | 3153.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 3172.90 | 3130.12 | 3153.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 3173.00 | 3138.70 | 3155.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 3171.00 | 3138.70 | 3155.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 3155.00 | 3143.00 | 3154.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 3155.00 | 3143.00 | 3154.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 3141.50 | 3142.70 | 3153.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 3137.10 | 3142.70 | 3153.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 2980.24 | 3079.32 | 3119.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 3015.40 | 3009.63 | 3051.31 | SL hit (close>ema200) qty=0.50 sl=3009.63 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 3112.50 | 3064.56 | 3058.62 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 3045.00 | 3059.98 | 3061.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2980.50 | 3044.09 | 3054.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3051.10 | 2988.82 | 3013.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3051.10 | 2988.82 | 3013.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3051.10 | 2988.82 | 3013.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 3051.10 | 2988.82 | 3013.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 2999.20 | 2990.89 | 3011.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 2920.70 | 3013.49 | 3017.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 2986.00 | 2991.64 | 2999.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 3011.70 | 3004.02 | 3003.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 3011.70 | 3004.02 | 3003.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 3011.70 | 3004.02 | 3003.86 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 2969.20 | 2997.05 | 3000.71 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 3035.10 | 3007.13 | 3004.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 3046.70 | 3015.04 | 3008.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 3340.00 | 3347.42 | 3299.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 3340.00 | 3347.42 | 3299.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 3846.50 | 3852.66 | 3817.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 3825.00 | 3852.66 | 3817.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 3774.50 | 3837.03 | 3813.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 3774.50 | 3837.03 | 3813.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 3800.20 | 3829.67 | 3812.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 3776.30 | 3829.67 | 3812.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 3789.50 | 3814.65 | 3807.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:30:00 | 3810.00 | 3813.24 | 3807.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 3808.30 | 3824.38 | 3825.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 3808.30 | 3824.38 | 3825.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 3792.40 | 3817.04 | 3821.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 3819.00 | 3801.47 | 3810.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 11:15:00 | 3819.00 | 3801.47 | 3810.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 3819.00 | 3801.47 | 3810.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 3819.00 | 3801.47 | 3810.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 3805.00 | 3802.18 | 3809.96 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 3905.80 | 3827.28 | 3819.29 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 3807.50 | 3830.61 | 3832.91 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 3848.90 | 3833.42 | 3832.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 3864.00 | 3839.53 | 3835.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 3821.60 | 3847.06 | 3842.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 3821.60 | 3847.06 | 3842.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 3821.60 | 3847.06 | 3842.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 3818.00 | 3847.06 | 3842.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 3813.70 | 3840.39 | 3839.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 3813.70 | 3840.39 | 3839.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 3804.10 | 3833.13 | 3836.50 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 15:15:00 | 2936.60 | 2025-05-16 09:15:00 | 3083.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-13 15:15:00 | 2936.60 | 2025-05-19 09:15:00 | 3049.90 | STOP_HIT | 0.50 | 3.86% |
| BUY | retest2 | 2025-05-20 10:45:00 | 3148.00 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | 3.78% |
| BUY | retest2 | 2025-05-20 13:00:00 | 3147.00 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | 3.81% |
| BUY | retest2 | 2025-05-21 09:30:00 | 3145.60 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | 3.86% |
| BUY | retest2 | 2025-05-21 10:00:00 | 3163.50 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | 3.27% |
| BUY | retest2 | 2025-05-27 09:15:00 | 3281.70 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-06-05 10:15:00 | 3304.80 | 2025-06-06 11:15:00 | 3309.70 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-06-06 10:30:00 | 3306.60 | 2025-06-06 11:15:00 | 3309.70 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-06-11 12:30:00 | 3366.00 | 2025-06-12 09:15:00 | 3307.90 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-18 13:15:00 | 3328.70 | 2025-06-19 10:15:00 | 3284.60 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-25 10:15:00 | 3113.60 | 2025-06-27 09:15:00 | 3254.00 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2025-06-25 12:00:00 | 3105.60 | 2025-06-27 09:15:00 | 3254.00 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2025-07-03 09:15:00 | 3304.70 | 2025-07-04 10:15:00 | 3263.20 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-16 09:30:00 | 3092.20 | 2025-07-18 11:15:00 | 3136.30 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-07-16 11:00:00 | 3107.40 | 2025-07-18 11:15:00 | 3136.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-17 13:45:00 | 3104.20 | 2025-07-18 11:15:00 | 3136.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-17 15:15:00 | 3105.70 | 2025-07-18 11:15:00 | 3136.30 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-22 11:30:00 | 3146.60 | 2025-07-23 10:15:00 | 3119.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-22 13:30:00 | 3147.60 | 2025-07-23 10:15:00 | 3119.20 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-22 14:00:00 | 3146.00 | 2025-07-23 10:15:00 | 3119.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-22 14:30:00 | 3149.30 | 2025-07-23 10:15:00 | 3119.20 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-30 10:15:00 | 3039.90 | 2025-07-30 14:15:00 | 3073.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-31 09:15:00 | 3038.50 | 2025-08-05 12:15:00 | 3040.40 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-07-31 13:15:00 | 3036.40 | 2025-08-05 12:15:00 | 3040.40 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-08-05 09:30:00 | 3028.20 | 2025-08-05 12:15:00 | 3040.40 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-08-07 14:15:00 | 3107.20 | 2025-08-08 14:15:00 | 3032.40 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-08-25 11:15:00 | 3199.80 | 2025-08-25 14:15:00 | 3166.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-08-29 12:15:00 | 3069.90 | 2025-09-01 10:15:00 | 3099.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-29 14:30:00 | 3074.40 | 2025-09-01 10:15:00 | 3099.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-15 10:15:00 | 3255.70 | 2025-09-22 10:15:00 | 3282.40 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-10-01 11:30:00 | 3105.80 | 2025-10-03 12:15:00 | 3138.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-29 13:30:00 | 3142.00 | 2025-10-29 15:15:00 | 3165.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-04 11:15:00 | 3108.00 | 2025-11-13 09:15:00 | 3107.90 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-11-06 10:00:00 | 3106.70 | 2025-11-13 09:15:00 | 3107.90 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-12-04 09:45:00 | 3352.10 | 2025-12-05 11:15:00 | 3301.40 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-04 13:45:00 | 3349.00 | 2025-12-05 11:15:00 | 3301.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-04 14:15:00 | 3358.00 | 2025-12-05 11:15:00 | 3301.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-05 10:30:00 | 3350.00 | 2025-12-05 11:15:00 | 3301.40 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-12-15 12:30:00 | 3153.00 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-12-16 09:15:00 | 3149.50 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-12-16 10:15:00 | 3151.80 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2025-12-16 13:15:00 | 3144.90 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-12-17 10:15:00 | 3145.00 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-12-17 14:00:00 | 3147.60 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-12-24 12:15:00 | 3142.60 | 2025-12-24 14:15:00 | 3095.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-02 15:00:00 | 3096.90 | 2026-01-08 13:15:00 | 3093.40 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-01-06 10:00:00 | 3096.10 | 2026-01-08 13:15:00 | 3093.40 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2026-01-06 14:00:00 | 3096.90 | 2026-01-08 13:15:00 | 3093.40 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-01-08 11:30:00 | 3095.00 | 2026-01-08 13:15:00 | 3093.40 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-01-09 14:30:00 | 3053.60 | 2026-01-16 09:15:00 | 2906.90 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2026-01-09 14:30:00 | 3053.60 | 2026-01-19 09:15:00 | 2940.10 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2026-01-09 15:00:00 | 3059.90 | 2026-01-20 11:15:00 | 2900.92 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2026-01-09 15:00:00 | 3059.90 | 2026-01-21 11:15:00 | 2886.10 | STOP_HIT | 0.50 | 5.68% |
| BUY | retest2 | 2026-02-01 13:30:00 | 3064.90 | 2026-02-01 15:15:00 | 2985.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-02-12 09:15:00 | 3120.80 | 2026-02-12 13:15:00 | 3157.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-02-12 12:15:00 | 3133.40 | 2026-02-12 13:15:00 | 3157.60 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3102.80 | 2026-02-13 11:15:00 | 3175.80 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-02-13 13:45:00 | 3129.30 | 2026-02-16 11:15:00 | 3159.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-02-23 09:15:00 | 3247.30 | 2026-03-02 15:15:00 | 3321.90 | STOP_HIT | 1.00 | 2.30% |
| SELL | retest2 | 2026-03-06 11:15:00 | 3253.10 | 2026-03-06 11:15:00 | 3296.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-03-20 14:15:00 | 3137.10 | 2026-03-23 10:15:00 | 2980.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:15:00 | 3137.10 | 2026-03-24 11:15:00 | 3015.40 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2026-04-02 09:15:00 | 2920.70 | 2026-04-06 09:15:00 | 3011.70 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2026-04-02 14:30:00 | 2986.00 | 2026-04-06 09:15:00 | 3011.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-04-24 14:30:00 | 3810.00 | 2026-04-29 11:15:00 | 3808.30 | STOP_HIT | 1.00 | -0.04% |
