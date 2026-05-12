# AIA Engineering Ltd. (AIAENG)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 3955.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 87 |
| ALERT1 | 55 |
| ALERT2 | 53 |
| ALERT2_SKIP | 24 |
| ALERT3 | 144 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 78 |
| PARTIAL | 9 |
| TARGET_HIT | 0 |
| STOP_HIT | 81 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 89 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 56
- **Target hits / Stop hits / Partials:** 0 / 80 / 9
- **Avg / median % per leg:** 0.19% / -0.60%
- **Sum % (uncompounded):** 16.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 9 | 18.8% | 0 | 48 | 0 | -0.78% | -37.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.91% | -0.9% |
| BUY @ 3rd Alert (retest2) | 47 | 9 | 19.1% | 0 | 47 | 0 | -0.78% | -36.5% |
| SELL (all) | 41 | 24 | 58.5% | 0 | 32 | 9 | 1.32% | 54.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.04% | -1.0% |
| SELL @ 3rd Alert (retest2) | 40 | 24 | 60.0% | 0 | 31 | 9 | 1.38% | 55.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.97% | -1.9% |
| retest2 (combined) | 87 | 33 | 37.9% | 0 | 78 | 9 | 0.21% | 18.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3180.30 | 3134.56 | 3133.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 3215.20 | 3150.69 | 3140.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 3227.00 | 3227.48 | 3196.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:30:00 | 3220.00 | 3227.48 | 3196.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 3256.80 | 3256.68 | 3236.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 3303.60 | 3267.38 | 3243.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:30:00 | 3303.10 | 3279.46 | 3253.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:45:00 | 3310.50 | 3313.30 | 3294.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 15:15:00 | 3287.50 | 3303.17 | 3304.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 3287.50 | 3303.17 | 3304.14 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 3318.40 | 3306.21 | 3305.44 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 3294.30 | 3304.95 | 3305.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 3282.20 | 3298.79 | 3302.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 13:15:00 | 3245.00 | 3244.43 | 3264.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 14:00:00 | 3245.00 | 3244.43 | 3264.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 3317.70 | 3259.09 | 3269.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:30:00 | 3309.80 | 3259.09 | 3269.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 3314.90 | 3270.25 | 3273.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 3331.70 | 3270.25 | 3273.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 3397.60 | 3295.72 | 3284.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 3450.10 | 3397.90 | 3375.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 12:15:00 | 3502.20 | 3506.09 | 3459.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 13:00:00 | 3502.20 | 3506.09 | 3459.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 3482.10 | 3496.60 | 3466.12 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 3429.70 | 3462.19 | 3464.87 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 3511.00 | 3468.34 | 3465.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 3541.90 | 3495.94 | 3480.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 3502.30 | 3506.91 | 3494.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 13:15:00 | 3502.30 | 3506.91 | 3494.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 3502.30 | 3506.91 | 3494.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 3498.60 | 3506.91 | 3494.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 3500.00 | 3505.46 | 3497.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 3499.60 | 3505.46 | 3497.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 3497.20 | 3503.81 | 3497.18 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 3493.50 | 3503.04 | 3503.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 3453.30 | 3492.45 | 3498.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 3469.00 | 3457.59 | 3471.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 3406.40 | 3457.59 | 3471.23 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 3441.70 | 3418.68 | 3426.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 3441.70 | 3418.68 | 3426.22 | SL hit (close>ema400) qty=1.00 sl=3426.22 alert=retest1 |

### Cycle 9 — BUY (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 10:15:00 | 3341.70 | 3312.95 | 3310.03 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 15:15:00 | 3303.90 | 3321.92 | 3323.93 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 11:15:00 | 3328.30 | 3325.67 | 3325.36 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 3297.30 | 3319.99 | 3322.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 3290.00 | 3307.52 | 3315.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3325.00 | 3278.73 | 3290.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 3325.00 | 3278.73 | 3290.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 3325.00 | 3278.73 | 3290.53 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 3319.60 | 3299.51 | 3298.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 3332.70 | 3317.26 | 3309.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 3455.90 | 3462.53 | 3426.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 15:00:00 | 3455.90 | 3462.53 | 3426.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 3429.10 | 3454.48 | 3428.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 3430.70 | 3454.48 | 3428.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 3423.80 | 3448.35 | 3428.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 3421.20 | 3448.35 | 3428.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 3415.70 | 3441.82 | 3427.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:45:00 | 3413.70 | 3441.82 | 3427.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 3419.40 | 3437.33 | 3426.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 3412.20 | 3437.33 | 3426.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 3425.90 | 3435.05 | 3426.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:45:00 | 3438.70 | 3434.16 | 3426.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 3433.90 | 3433.33 | 3427.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 3442.00 | 3431.46 | 3426.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 3390.70 | 3423.31 | 3423.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 3390.70 | 3423.31 | 3423.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 3374.50 | 3413.55 | 3419.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 15:15:00 | 3425.00 | 3370.75 | 3383.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 15:15:00 | 3425.00 | 3370.75 | 3383.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 3425.00 | 3370.75 | 3383.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 3430.10 | 3370.75 | 3383.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3401.00 | 3376.80 | 3384.68 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 3393.10 | 3389.60 | 3389.42 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 3383.00 | 3388.81 | 3389.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 3372.90 | 3385.63 | 3387.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 13:15:00 | 3384.80 | 3384.24 | 3386.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 13:15:00 | 3384.80 | 3384.24 | 3386.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 3384.80 | 3384.24 | 3386.44 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 3413.80 | 3388.29 | 3387.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 12:15:00 | 3434.40 | 3404.00 | 3395.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 3413.60 | 3422.75 | 3408.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 3413.60 | 3422.75 | 3408.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 3406.10 | 3419.42 | 3408.56 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 3371.30 | 3401.21 | 3402.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 3369.30 | 3394.83 | 3399.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 3352.70 | 3343.30 | 3354.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 3352.70 | 3343.30 | 3354.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 3352.70 | 3343.30 | 3354.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 3357.80 | 3343.30 | 3354.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 3358.30 | 3346.30 | 3354.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:45:00 | 3354.10 | 3346.30 | 3354.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 3353.80 | 3347.80 | 3354.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 3336.10 | 3350.04 | 3353.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 3169.29 | 3221.73 | 3256.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 3200.10 | 3191.54 | 3217.93 | SL hit (close>ema200) qty=0.50 sl=3191.54 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 3150.00 | 3118.18 | 3114.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 11:15:00 | 3182.40 | 3140.35 | 3126.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 3161.80 | 3168.44 | 3148.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 3161.80 | 3168.44 | 3148.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 3161.80 | 3168.44 | 3148.28 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 3103.10 | 3137.24 | 3140.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 3083.10 | 3120.55 | 3132.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 3107.00 | 3096.04 | 3112.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 11:45:00 | 3105.60 | 3096.04 | 3112.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 3114.00 | 3099.64 | 3112.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:30:00 | 3110.70 | 3099.64 | 3112.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 3133.20 | 3106.35 | 3114.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:15:00 | 3144.20 | 3106.35 | 3114.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 3110.90 | 3107.26 | 3114.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 3134.60 | 3107.26 | 3114.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 3120.30 | 3109.87 | 3114.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 3131.20 | 3109.87 | 3114.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 3107.50 | 3109.39 | 3113.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 3093.10 | 3109.39 | 3113.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 3124.60 | 3110.41 | 3110.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 3124.60 | 3110.41 | 3110.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 3131.10 | 3117.88 | 3114.02 | Break + close above crossover candle high |

### Cycle 22 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 3059.70 | 3111.49 | 3112.49 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 15:15:00 | 3120.00 | 3110.85 | 3110.10 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 3100.30 | 3108.74 | 3109.21 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 3114.60 | 3110.00 | 3109.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 14:15:00 | 3120.00 | 3112.87 | 3111.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 3105.90 | 3112.93 | 3111.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 3105.90 | 3112.93 | 3111.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 3105.90 | 3112.93 | 3111.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 3092.40 | 3112.93 | 3111.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 3090.00 | 3108.35 | 3109.55 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 3124.60 | 3112.59 | 3110.96 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 3092.40 | 3109.84 | 3111.24 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 3119.90 | 3111.92 | 3111.80 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 3099.00 | 3111.15 | 3112.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 3081.00 | 3104.78 | 3108.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 3118.10 | 3107.44 | 3109.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 3118.10 | 3107.44 | 3109.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 3118.10 | 3107.44 | 3109.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:15:00 | 3091.60 | 3110.60 | 3110.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 3094.20 | 3109.48 | 3110.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:00:00 | 3091.20 | 3105.83 | 3108.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:30:00 | 3089.40 | 3098.95 | 3104.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 3035.90 | 3031.77 | 3048.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 3053.90 | 3031.77 | 3048.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 3046.40 | 3035.36 | 3047.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 3047.20 | 3035.36 | 3047.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 3051.40 | 3038.57 | 3047.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 3051.40 | 3038.57 | 3047.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 3053.00 | 3041.45 | 3048.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 3069.90 | 3041.45 | 3048.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 3085.70 | 3054.69 | 3053.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 3085.70 | 3054.69 | 3053.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 3092.90 | 3062.34 | 3057.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 3102.10 | 3104.98 | 3086.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 3102.10 | 3104.98 | 3086.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 3093.90 | 3104.99 | 3091.63 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 13:15:00 | 3070.00 | 3084.58 | 3084.94 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 3091.10 | 3085.52 | 3085.17 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 3070.00 | 3083.49 | 3084.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 3055.50 | 3076.54 | 3081.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 3047.50 | 3037.48 | 3050.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 3047.50 | 3037.48 | 3050.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3047.50 | 3037.48 | 3050.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 3047.50 | 3037.48 | 3050.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 3042.00 | 3039.14 | 3047.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 3040.50 | 3039.14 | 3047.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3065.70 | 3042.17 | 3046.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 3061.10 | 3042.17 | 3046.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 3064.10 | 3046.56 | 3047.94 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 3059.90 | 3049.23 | 3049.03 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 3039.00 | 3050.53 | 3050.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 15:15:00 | 3029.10 | 3042.73 | 3046.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 10:15:00 | 3035.60 | 3035.43 | 3042.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 3035.60 | 3035.43 | 3042.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3035.60 | 3035.43 | 3042.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 3037.00 | 3035.43 | 3042.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 3051.90 | 3038.72 | 3043.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 3051.90 | 3038.72 | 3043.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 3057.90 | 3042.56 | 3044.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:30:00 | 3055.70 | 3042.56 | 3044.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 13:15:00 | 3077.10 | 3049.47 | 3047.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 14:15:00 | 3090.00 | 3057.57 | 3051.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 3100.10 | 3101.52 | 3088.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 3100.10 | 3101.52 | 3088.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 3110.00 | 3117.67 | 3106.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:30:00 | 3131.00 | 3122.56 | 3109.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 3128.50 | 3131.35 | 3123.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 3069.40 | 3110.52 | 3115.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 3069.40 | 3110.52 | 3115.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 3048.60 | 3077.99 | 3094.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 3086.30 | 3061.30 | 3072.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 3086.30 | 3061.30 | 3072.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 3086.30 | 3061.30 | 3072.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 3086.30 | 3061.30 | 3072.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 3074.50 | 3063.94 | 3072.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 3059.20 | 3063.94 | 3072.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 10:15:00 | 3086.00 | 3075.58 | 3074.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 3086.00 | 3075.58 | 3074.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 14:15:00 | 3098.50 | 3083.89 | 3079.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 3071.00 | 3082.07 | 3079.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 3071.00 | 3082.07 | 3079.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 3071.00 | 3082.07 | 3079.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 3077.70 | 3082.07 | 3079.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 3073.10 | 3080.27 | 3078.71 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 3065.10 | 3077.24 | 3077.47 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 3112.40 | 3081.74 | 3079.15 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 3061.80 | 3074.67 | 3076.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 14:15:00 | 3052.60 | 3066.35 | 3070.46 | Break + close below crossover candle low |

### Cycle 43 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 3117.00 | 3076.27 | 3074.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 10:15:00 | 3148.70 | 3090.75 | 3081.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 3190.20 | 3201.92 | 3170.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:45:00 | 3204.00 | 3201.92 | 3170.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 3173.00 | 3196.14 | 3170.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 3173.60 | 3196.14 | 3170.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 3170.00 | 3190.91 | 3170.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 3171.30 | 3190.91 | 3170.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 3163.70 | 3185.47 | 3170.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 3163.70 | 3185.47 | 3170.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 3165.00 | 3181.38 | 3169.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 3157.60 | 3181.38 | 3169.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3158.00 | 3174.69 | 3169.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 11:30:00 | 3194.30 | 3178.23 | 3172.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 3195.10 | 3182.58 | 3174.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 3189.00 | 3180.72 | 3176.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:30:00 | 3198.00 | 3187.05 | 3179.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3256.00 | 3264.00 | 3247.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-13 13:15:00 | 3215.50 | 3239.23 | 3239.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 3215.50 | 3239.23 | 3239.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 3201.10 | 3231.59 | 3236.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 13:15:00 | 3228.30 | 3223.29 | 3229.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:00:00 | 3228.30 | 3223.29 | 3229.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 3232.00 | 3225.03 | 3230.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 3229.40 | 3225.03 | 3230.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 3225.00 | 3225.03 | 3229.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 3227.60 | 3225.03 | 3229.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 3225.40 | 3225.10 | 3229.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 3214.70 | 3227.10 | 3229.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 15:15:00 | 3270.00 | 3230.10 | 3229.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 3270.00 | 3230.10 | 3229.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 3333.40 | 3250.76 | 3238.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 3296.00 | 3302.70 | 3278.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:30:00 | 3323.10 | 3308.41 | 3293.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 3292.90 | 3305.30 | 3293.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 3292.90 | 3305.30 | 3293.29 | SL hit (close<ema400) qty=1.00 sl=3293.29 alert=retest1 |

### Cycle 46 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 3330.00 | 3349.16 | 3350.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 3317.90 | 3338.76 | 3345.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 3355.50 | 3341.84 | 3345.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 11:15:00 | 3355.50 | 3341.84 | 3345.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 3355.50 | 3341.84 | 3345.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:15:00 | 3350.00 | 3341.84 | 3345.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 3335.00 | 3340.47 | 3344.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:45:00 | 3328.90 | 3339.18 | 3343.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 3316.40 | 3338.35 | 3342.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 3315.00 | 3290.03 | 3288.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 3315.00 | 3290.03 | 3288.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 12:15:00 | 3327.20 | 3297.47 | 3292.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 3308.10 | 3311.57 | 3301.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 3308.10 | 3311.57 | 3301.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 3301.40 | 3309.53 | 3301.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:00:00 | 3314.90 | 3305.70 | 3301.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 15:15:00 | 3285.00 | 3304.05 | 3301.62 | SL hit (close<static) qty=1.00 sl=3295.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 3228.50 | 3288.94 | 3294.98 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 3380.90 | 3287.48 | 3287.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 3431.50 | 3328.13 | 3306.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 3672.20 | 3678.30 | 3627.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 3672.20 | 3678.30 | 3627.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 3679.60 | 3726.23 | 3693.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 3679.60 | 3726.23 | 3693.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 3703.60 | 3721.70 | 3694.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:15:00 | 3731.70 | 3717.96 | 3695.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 3716.00 | 3712.04 | 3701.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 3724.60 | 3709.54 | 3703.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 10:00:00 | 3725.00 | 3720.70 | 3710.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 3730.20 | 3770.11 | 3744.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 3743.20 | 3770.11 | 3744.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 3719.40 | 3759.97 | 3742.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 3719.40 | 3759.97 | 3742.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 3705.50 | 3740.25 | 3735.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 3710.00 | 3740.25 | 3735.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 3703.90 | 3727.36 | 3730.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 3703.90 | 3727.36 | 3730.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 3685.00 | 3718.88 | 3726.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 10:15:00 | 3722.20 | 3717.92 | 3724.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 10:15:00 | 3722.20 | 3717.92 | 3724.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 3722.20 | 3717.92 | 3724.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 3722.20 | 3717.92 | 3724.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 3717.00 | 3717.73 | 3723.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:45:00 | 3700.00 | 3719.46 | 3722.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 3701.00 | 3719.46 | 3722.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 3732.20 | 3719.05 | 3721.94 | SL hit (close>static) qty=1.00 sl=3723.70 alert=retest2 |

### Cycle 51 — BUY (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 10:15:00 | 3749.40 | 3725.12 | 3724.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 3769.90 | 3734.08 | 3728.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 10:15:00 | 3795.00 | 3797.35 | 3769.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:30:00 | 3773.50 | 3797.35 | 3769.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 3772.30 | 3792.47 | 3776.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 3772.30 | 3792.47 | 3776.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 3740.50 | 3782.08 | 3772.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 3779.50 | 3782.08 | 3772.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 15:15:00 | 3810.50 | 3854.97 | 3857.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 3810.50 | 3854.97 | 3857.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 3803.80 | 3844.74 | 3853.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 3738.00 | 3708.76 | 3744.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 3738.00 | 3708.76 | 3744.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 3749.50 | 3716.91 | 3744.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 3749.50 | 3716.91 | 3744.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 3768.50 | 3727.23 | 3747.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 3769.60 | 3727.23 | 3747.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 3793.30 | 3740.44 | 3751.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 3793.30 | 3740.44 | 3751.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 3785.80 | 3761.44 | 3759.68 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 3742.60 | 3757.67 | 3758.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 3731.20 | 3752.38 | 3755.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 3744.30 | 3674.10 | 3695.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 3744.30 | 3674.10 | 3695.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 3744.30 | 3674.10 | 3695.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 3744.30 | 3674.10 | 3695.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 3734.20 | 3686.12 | 3698.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 3733.20 | 3686.12 | 3698.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 3734.00 | 3709.53 | 3707.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 3748.80 | 3717.38 | 3711.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 3826.00 | 3841.28 | 3795.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 3826.00 | 3841.28 | 3795.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 3826.00 | 3841.28 | 3795.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 3795.00 | 3841.28 | 3795.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 3792.40 | 3825.25 | 3795.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 3790.10 | 3825.25 | 3795.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 3781.60 | 3816.52 | 3794.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 3781.60 | 3816.52 | 3794.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 3728.50 | 3777.06 | 3780.64 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 3803.10 | 3783.68 | 3783.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 14:15:00 | 3826.90 | 3799.17 | 3790.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 3747.50 | 3789.77 | 3788.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 3747.50 | 3789.77 | 3788.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 3747.50 | 3789.77 | 3788.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 3742.00 | 3789.77 | 3788.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 3765.80 | 3784.98 | 3786.12 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 3845.10 | 3796.42 | 3790.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 3862.00 | 3821.26 | 3807.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 3883.00 | 3890.37 | 3857.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 3883.00 | 3890.37 | 3857.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 3883.00 | 3890.37 | 3857.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:00:00 | 3959.90 | 3915.85 | 3887.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:30:00 | 3949.90 | 3932.39 | 3905.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 3949.00 | 3930.05 | 3909.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 3875.40 | 3899.19 | 3902.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 3875.40 | 3899.19 | 3902.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 3863.90 | 3892.13 | 3898.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 3854.90 | 3851.06 | 3872.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 3854.90 | 3851.06 | 3872.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 3780.50 | 3836.95 | 3864.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 3776.80 | 3836.95 | 3864.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 3865.40 | 3842.64 | 3864.32 | SL hit (close>static) qty=1.00 sl=3865.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 12:15:00 | 3905.00 | 3877.89 | 3877.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 3928.90 | 3893.28 | 3884.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 3974.80 | 3990.92 | 3954.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 3974.80 | 3990.92 | 3954.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 4066.50 | 4099.61 | 4065.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 4066.50 | 4099.61 | 4065.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 4060.00 | 4091.69 | 4065.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 4042.90 | 4091.69 | 4065.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 4055.30 | 4084.41 | 4064.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 4050.00 | 4084.41 | 4064.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 4056.90 | 4078.91 | 4063.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 4055.00 | 4078.91 | 4063.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 4066.40 | 4076.41 | 4064.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 4075.00 | 4076.41 | 4064.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 15:15:00 | 4030.10 | 4067.86 | 4064.21 | SL hit (close<static) qty=1.00 sl=4048.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 4028.30 | 4059.95 | 4060.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 4001.40 | 4048.24 | 4055.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 4111.20 | 4031.69 | 4039.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 4111.20 | 4031.69 | 4039.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 4111.20 | 4031.69 | 4039.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 4130.00 | 4031.69 | 4039.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 10:15:00 | 4097.90 | 4044.93 | 4044.50 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 4024.10 | 4048.84 | 4051.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 14:15:00 | 3952.80 | 4019.70 | 4036.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 3941.30 | 3937.07 | 3976.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 3941.30 | 3937.07 | 3976.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 3990.00 | 3947.66 | 3978.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 3930.30 | 3937.37 | 3970.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 3929.50 | 3932.11 | 3962.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 3926.10 | 3912.22 | 3937.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:00:00 | 3931.00 | 3918.74 | 3934.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 3941.40 | 3923.27 | 3935.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 3935.00 | 3923.27 | 3935.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 3950.10 | 3928.64 | 3936.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 3911.90 | 3928.64 | 3936.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 14:15:00 | 3959.50 | 3918.71 | 3926.21 | SL hit (close>static) qty=1.00 sl=3959.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 12:15:00 | 3798.80 | 3769.61 | 3767.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 3925.40 | 3809.01 | 3786.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 09:15:00 | 3806.50 | 3808.51 | 3788.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 10:00:00 | 3806.50 | 3808.51 | 3788.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 3875.00 | 3869.98 | 3846.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:45:00 | 3852.00 | 3869.98 | 3846.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 4001.00 | 4001.77 | 3943.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 3917.60 | 4001.77 | 3943.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 3915.90 | 3984.59 | 3941.38 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 3839.60 | 3913.65 | 3917.95 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 14:15:00 | 3989.00 | 3928.72 | 3924.41 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 3875.00 | 3918.07 | 3921.19 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 4024.00 | 3937.13 | 3927.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 4050.20 | 3971.08 | 3945.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 3929.20 | 4006.80 | 3984.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 3929.20 | 4006.80 | 3984.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 3929.20 | 4006.80 | 3984.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:30:00 | 3932.80 | 4006.80 | 3984.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 3941.50 | 3993.74 | 3980.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:45:00 | 3941.50 | 3993.74 | 3980.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 4022.20 | 4037.51 | 4015.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:00:00 | 4022.20 | 4037.51 | 4015.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 3997.10 | 4029.43 | 4013.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:30:00 | 3999.10 | 4029.43 | 4013.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 3987.10 | 4020.96 | 4011.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 3987.10 | 4020.96 | 4011.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 3922.70 | 3992.84 | 3999.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 3875.00 | 3953.15 | 3979.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 3916.30 | 3908.69 | 3943.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:00:00 | 3916.30 | 3908.69 | 3943.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 3925.80 | 3912.11 | 3941.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 3925.80 | 3912.11 | 3941.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 3937.70 | 3917.23 | 3941.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:30:00 | 3938.50 | 3917.23 | 3941.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 3962.50 | 3926.28 | 3943.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 3966.20 | 3926.28 | 3943.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 4019.00 | 3944.83 | 3950.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 4025.00 | 3944.83 | 3950.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 4029.90 | 3961.84 | 3957.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 4120.60 | 4003.70 | 3977.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 4025.20 | 4064.80 | 4030.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 4025.20 | 4064.80 | 4030.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 4025.20 | 4064.80 | 4030.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 4025.20 | 4064.80 | 4030.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 4031.20 | 4058.08 | 4030.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:15:00 | 4025.30 | 4058.08 | 4030.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 4025.60 | 4051.58 | 4030.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 4022.50 | 4051.58 | 4030.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 4017.30 | 4044.73 | 4029.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 4012.00 | 4044.73 | 4029.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 4017.20 | 4039.22 | 4028.00 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 3978.60 | 4020.47 | 4021.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 3939.70 | 3994.27 | 4008.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 3965.00 | 3960.63 | 3982.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:45:00 | 3964.30 | 3960.63 | 3982.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 3868.00 | 3871.88 | 3905.06 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 3945.50 | 3916.75 | 3915.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 3963.90 | 3926.18 | 3920.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 3955.30 | 3966.70 | 3950.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 3955.30 | 3966.70 | 3950.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 3955.30 | 3966.70 | 3950.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 3955.30 | 3966.70 | 3950.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 3951.70 | 3963.70 | 3950.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 3955.90 | 3963.70 | 3950.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 3965.80 | 3964.12 | 3951.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 3976.10 | 3959.34 | 3952.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 3977.90 | 3963.46 | 3955.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 12:15:00 | 3938.50 | 3958.44 | 3954.49 | SL hit (close<static) qty=1.00 sl=3941.10 alert=retest2 |

### Cycle 74 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 3902.50 | 3943.35 | 3948.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 3895.00 | 3928.00 | 3939.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 13:15:00 | 3873.50 | 3867.49 | 3892.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 14:00:00 | 3873.50 | 3867.49 | 3892.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 3911.40 | 3857.61 | 3870.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 3911.40 | 3857.61 | 3870.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 3909.30 | 3867.95 | 3873.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 3865.00 | 3867.95 | 3873.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 3926.20 | 3879.60 | 3878.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 3926.20 | 3879.60 | 3878.44 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 3845.60 | 3883.35 | 3887.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 3817.00 | 3859.63 | 3873.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 3649.60 | 3633.05 | 3676.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 10:00:00 | 3649.60 | 3633.05 | 3676.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 3666.00 | 3642.91 | 3673.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:30:00 | 3675.60 | 3642.91 | 3673.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 3672.10 | 3648.75 | 3673.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 3672.10 | 3648.75 | 3673.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 3681.80 | 3655.36 | 3674.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 3681.80 | 3655.36 | 3674.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 3675.00 | 3659.29 | 3674.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:30:00 | 3677.40 | 3659.29 | 3674.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3707.50 | 3625.78 | 3637.90 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 3713.60 | 3656.42 | 3650.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 3784.30 | 3699.25 | 3673.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 3711.50 | 3717.79 | 3689.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 10:45:00 | 3720.00 | 3717.79 | 3689.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 3710.00 | 3715.06 | 3693.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 3710.00 | 3715.06 | 3693.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 3669.70 | 3714.04 | 3701.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:30:00 | 3708.30 | 3715.35 | 3703.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:00:00 | 3707.80 | 3713.84 | 3703.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:15:00 | 3718.40 | 3711.83 | 3703.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 3712.90 | 3711.75 | 3704.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 3678.00 | 3705.00 | 3702.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 3696.20 | 3705.00 | 3702.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 3661.20 | 3696.24 | 3698.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 3661.20 | 3696.24 | 3698.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 3648.90 | 3686.77 | 3694.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 3690.10 | 3687.44 | 3693.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 3690.10 | 3687.44 | 3693.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 3690.10 | 3687.44 | 3693.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 3695.50 | 3687.44 | 3693.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 3672.50 | 3684.45 | 3691.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:45:00 | 3689.60 | 3684.45 | 3691.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 3668.60 | 3680.57 | 3688.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:30:00 | 3687.40 | 3680.57 | 3688.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 3665.50 | 3675.06 | 3684.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 3686.90 | 3675.06 | 3684.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 3662.60 | 3654.67 | 3669.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 3676.80 | 3654.67 | 3669.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 3698.40 | 3663.41 | 3672.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 3825.20 | 3663.41 | 3672.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 3900.00 | 3710.73 | 3693.04 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 3644.20 | 3713.70 | 3720.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 3585.10 | 3653.56 | 3686.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 3335.70 | 3331.28 | 3407.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 3335.70 | 3331.28 | 3407.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 3369.00 | 3350.54 | 3403.66 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 3505.80 | 3430.65 | 3423.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 12:15:00 | 3601.80 | 3517.61 | 3484.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 15:15:00 | 3635.00 | 3638.33 | 3584.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 09:15:00 | 3584.10 | 3638.33 | 3584.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3590.90 | 3628.85 | 3585.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 3633.20 | 3627.78 | 3592.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 3662.20 | 3650.18 | 3612.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 3659.80 | 3646.09 | 3620.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 3633.50 | 3667.06 | 3668.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 3633.50 | 3667.06 | 3668.59 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 3780.90 | 3685.50 | 3676.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 3824.10 | 3766.10 | 3729.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 3925.00 | 3929.92 | 3885.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 14:30:00 | 3922.00 | 3929.92 | 3885.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 3990.90 | 3941.26 | 3897.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 4047.70 | 3978.14 | 3938.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:00:00 | 4025.00 | 3993.71 | 3952.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:00:00 | 4008.70 | 3996.71 | 3957.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 4066.40 | 3998.97 | 3968.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 4002.60 | 4011.30 | 3993.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:30:00 | 4033.30 | 4015.57 | 4000.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 4030.50 | 4015.57 | 4000.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:30:00 | 4024.60 | 4038.44 | 4018.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 12:00:00 | 4028.90 | 4047.48 | 4036.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 4020.00 | 4041.99 | 4034.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 4000.70 | 4025.73 | 4028.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 4000.70 | 4025.73 | 4028.52 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 4085.50 | 4037.69 | 4033.70 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 13:15:00 | 4011.10 | 4028.40 | 4030.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 3983.60 | 4007.04 | 4017.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 4017.50 | 4003.56 | 4012.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 4017.50 | 4003.56 | 4012.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 4017.50 | 4003.56 | 4012.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 4006.00 | 4003.56 | 4012.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 4015.90 | 4006.03 | 4012.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:30:00 | 4017.00 | 4006.03 | 4012.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 3989.10 | 4002.64 | 4010.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:15:00 | 3964.50 | 3999.79 | 4008.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:00:00 | 3981.60 | 3949.69 | 3963.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:30:00 | 3976.10 | 3951.98 | 3962.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 3954.10 | 3951.98 | 3962.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 3940.40 | 3949.66 | 3960.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 3924.80 | 3949.66 | 3960.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 3974.40 | 3955.44 | 3960.43 | SL hit (close>static) qty=1.00 sl=3963.60 alert=retest2 |

### Cycle 87 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 3985.10 | 3926.89 | 3923.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 4001.50 | 3941.81 | 3930.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 3987.90 | 3996.48 | 3974.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 3987.90 | 3996.48 | 3974.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 3972.50 | 3991.68 | 3974.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 3972.50 | 3991.68 | 3974.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 3955.00 | 3984.35 | 3972.89 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 10:30:00 | 3303.60 | 2025-05-20 15:15:00 | 3287.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-05-15 12:30:00 | 3303.10 | 2025-05-20 15:15:00 | 3287.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-05-19 09:45:00 | 3310.50 | 2025-05-20 15:15:00 | 3287.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2025-06-16 09:15:00 | 3406.40 | 2025-06-18 09:15:00 | 3441.70 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-06-18 11:15:00 | 3400.80 | 2025-06-20 14:15:00 | 3230.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 12:15:00 | 3404.40 | 2025-06-20 14:15:00 | 3234.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 09:30:00 | 3401.20 | 2025-06-20 14:15:00 | 3231.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 11:15:00 | 3400.80 | 2025-06-25 09:15:00 | 3297.10 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2025-06-18 12:15:00 | 3404.40 | 2025-06-25 09:15:00 | 3297.10 | STOP_HIT | 0.50 | 3.15% |
| SELL | retest2 | 2025-06-19 09:30:00 | 3401.20 | 2025-06-25 09:15:00 | 3297.10 | STOP_HIT | 0.50 | 3.06% |
| BUY | retest2 | 2025-07-10 14:45:00 | 3438.70 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-11 09:15:00 | 3433.90 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-07-11 09:45:00 | 3442.00 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-07-24 10:30:00 | 3336.10 | 2025-07-29 10:15:00 | 3169.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:30:00 | 3336.10 | 2025-07-30 11:15:00 | 3200.10 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-08-12 10:15:00 | 3093.10 | 2025-08-13 10:15:00 | 3124.60 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-08-25 13:15:00 | 3091.60 | 2025-09-01 10:15:00 | 3085.70 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-08-25 14:15:00 | 3094.20 | 2025-09-01 10:15:00 | 3085.70 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-08-25 15:00:00 | 3091.20 | 2025-09-01 10:15:00 | 3085.70 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-08-26 09:30:00 | 3089.40 | 2025-09-01 10:15:00 | 3085.70 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-09-18 09:30:00 | 3131.00 | 2025-09-19 14:15:00 | 3069.40 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-09-19 11:15:00 | 3128.50 | 2025-09-19 14:15:00 | 3069.40 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-09-24 11:15:00 | 3059.20 | 2025-09-25 10:15:00 | 3086.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-07 11:30:00 | 3194.30 | 2025-10-13 13:15:00 | 3215.50 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-10-07 12:45:00 | 3195.10 | 2025-10-13 13:15:00 | 3215.50 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-10-08 10:15:00 | 3189.00 | 2025-10-13 13:15:00 | 3215.50 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2025-10-08 11:30:00 | 3198.00 | 2025-10-13 13:15:00 | 3215.50 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-10-15 12:15:00 | 3214.70 | 2025-10-15 15:15:00 | 3270.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2025-10-20 09:30:00 | 3323.10 | 2025-10-20 10:15:00 | 3292.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-20 13:15:00 | 3320.40 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-10-20 14:00:00 | 3320.20 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-10-21 13:45:00 | 3328.10 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-10-23 10:15:00 | 3347.20 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-27 09:15:00 | 3371.80 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-10-27 11:15:00 | 3380.70 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-27 12:30:00 | 3367.60 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-27 14:00:00 | 3371.00 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-29 14:45:00 | 3328.90 | 2025-11-04 11:15:00 | 3315.00 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3316.40 | 2025-11-04 11:15:00 | 3315.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-11-06 14:00:00 | 3314.90 | 2025-11-06 15:15:00 | 3285.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-18 12:15:00 | 3731.70 | 2025-11-21 14:15:00 | 3703.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-11-19 10:15:00 | 3716.00 | 2025-11-21 14:15:00 | 3703.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-11-19 15:00:00 | 3724.60 | 2025-11-21 14:15:00 | 3703.90 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-11-20 10:00:00 | 3725.00 | 2025-11-21 14:15:00 | 3703.90 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-11-24 14:45:00 | 3700.00 | 2025-11-25 09:15:00 | 3732.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-24 15:15:00 | 3701.00 | 2025-11-25 09:15:00 | 3732.20 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-27 09:15:00 | 3779.50 | 2025-12-04 15:15:00 | 3810.50 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-12-24 10:00:00 | 3959.90 | 2025-12-26 15:15:00 | 3875.40 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-12-24 13:30:00 | 3949.90 | 2025-12-26 15:15:00 | 3875.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-26 09:15:00 | 3949.00 | 2025-12-26 15:15:00 | 3875.40 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-12-30 09:15:00 | 3776.80 | 2025-12-30 09:15:00 | 3865.40 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-01-06 12:15:00 | 4075.00 | 2026-01-06 15:15:00 | 4030.10 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-13 09:30:00 | 3930.30 | 2026-01-16 14:15:00 | 3959.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-13 11:30:00 | 3929.50 | 2026-01-20 14:15:00 | 3733.78 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2026-01-14 11:00:00 | 3926.10 | 2026-01-20 14:15:00 | 3733.02 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2026-01-14 14:00:00 | 3931.00 | 2026-01-20 14:15:00 | 3729.79 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2026-01-16 09:15:00 | 3911.90 | 2026-01-20 14:15:00 | 3734.45 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2026-01-19 09:15:00 | 3902.20 | 2026-01-21 10:15:00 | 3707.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:30:00 | 3929.50 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-01-14 11:00:00 | 3926.10 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-01-14 14:00:00 | 3931.00 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2026-01-16 09:15:00 | 3911.90 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2026-01-19 09:15:00 | 3902.20 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.24% |
| BUY | retest2 | 2026-02-20 09:30:00 | 3976.10 | 2026-02-20 12:15:00 | 3938.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-20 10:30:00 | 3977.90 | 2026-02-20 12:15:00 | 3938.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-02-26 09:15:00 | 3865.00 | 2026-02-26 09:15:00 | 3926.20 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-03-12 10:30:00 | 3708.30 | 2026-03-13 09:15:00 | 3661.20 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-03-12 12:00:00 | 3707.80 | 2026-03-13 09:15:00 | 3661.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-03-12 13:15:00 | 3718.40 | 2026-03-13 09:15:00 | 3661.20 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-12 15:00:00 | 3712.90 | 2026-03-13 09:15:00 | 3661.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-04-02 11:30:00 | 3633.20 | 2026-04-09 14:15:00 | 3633.50 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-04-02 14:45:00 | 3662.20 | 2026-04-09 14:15:00 | 3633.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-04-06 11:15:00 | 3659.80 | 2026-04-09 14:15:00 | 3633.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-04-20 09:45:00 | 4047.70 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-04-20 12:00:00 | 4025.00 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-04-20 13:00:00 | 4008.70 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-04-21 09:15:00 | 4066.40 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-04-22 13:30:00 | 4033.30 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-04-22 14:00:00 | 4030.50 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-04-23 10:30:00 | 4024.60 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-04-24 12:00:00 | 4028.90 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-04-29 13:15:00 | 3964.50 | 2026-05-04 15:15:00 | 3974.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-05-04 10:00:00 | 3981.60 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-05-04 11:30:00 | 3976.10 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2026-05-04 12:00:00 | 3954.10 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-05-04 13:15:00 | 3924.80 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-05-05 10:00:00 | 3923.10 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-05-05 10:30:00 | 3930.10 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-05-06 14:00:00 | 3919.80 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -1.67% |
