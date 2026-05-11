# Navin Fluorine International Ltd. (NAVINFLUOR)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 7039.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 165 |
| ALERT1 | 106 |
| ALERT2 | 104 |
| ALERT2_SKIP | 51 |
| ALERT3 | 280 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 129 |
| PARTIAL | 5 |
| TARGET_HIT | 8 |
| STOP_HIT | 123 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 38 / 98
- **Target hits / Stop hits / Partials:** 8 / 123 / 5
- **Avg / median % per leg:** -0.28% / -1.00%
- **Sum % (uncompounded):** -38.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 22 | 29.3% | 5 | 70 | 0 | -0.01% | -0.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 75 | 22 | 29.3% | 5 | 70 | 0 | -0.01% | -0.4% |
| SELL (all) | 61 | 16 | 26.2% | 3 | 53 | 5 | -0.62% | -37.7% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.07% | -3.2% |
| SELL @ 3rd Alert (retest2) | 58 | 16 | 27.6% | 3 | 50 | 5 | -0.59% | -34.4% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.07% | -3.2% |
| retest2 (combined) | 133 | 38 | 28.6% | 8 | 120 | 5 | -0.26% | -34.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 3307.85 | 3271.74 | 3270.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 3334.95 | 3291.94 | 3280.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 3321.00 | 3323.05 | 3305.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 14:15:00 | 3321.00 | 3323.05 | 3305.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 3321.00 | 3323.05 | 3305.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:45:00 | 3327.00 | 3323.05 | 3305.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 3292.95 | 3322.44 | 3314.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 3292.95 | 3322.44 | 3314.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 3317.95 | 3321.54 | 3314.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 3332.95 | 3319.37 | 3314.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 12:15:00 | 3361.70 | 3380.54 | 3381.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 3361.70 | 3380.54 | 3381.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 13:15:00 | 3352.80 | 3374.99 | 3378.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 10:15:00 | 3350.30 | 3333.96 | 3346.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 10:15:00 | 3350.30 | 3333.96 | 3346.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 3350.30 | 3333.96 | 3346.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:45:00 | 3356.00 | 3333.96 | 3346.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 3364.00 | 3339.97 | 3347.80 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 13:15:00 | 3378.55 | 3354.01 | 3353.21 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 3327.45 | 3353.82 | 3357.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 3318.05 | 3342.02 | 3350.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 3337.30 | 3331.02 | 3339.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 13:15:00 | 3337.30 | 3331.02 | 3339.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 3337.30 | 3331.02 | 3339.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:30:00 | 3340.90 | 3331.02 | 3339.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 3317.95 | 3328.41 | 3337.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:15:00 | 3305.00 | 3328.41 | 3337.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 14:15:00 | 3308.60 | 3263.99 | 3261.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 14:15:00 | 3308.60 | 3263.99 | 3261.04 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 3065.80 | 3222.21 | 3243.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 2973.10 | 3172.39 | 3218.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 3235.90 | 3131.89 | 3174.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 3235.90 | 3131.89 | 3174.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 3235.90 | 3131.89 | 3174.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 3235.90 | 3131.89 | 3174.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 3262.10 | 3157.94 | 3182.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 3262.10 | 3157.94 | 3182.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 3277.65 | 3199.98 | 3198.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 3294.85 | 3218.96 | 3207.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 12:15:00 | 3311.00 | 3313.73 | 3287.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 13:00:00 | 3311.00 | 3313.73 | 3287.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 3539.15 | 3559.26 | 3538.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 3568.40 | 3559.26 | 3538.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:00:00 | 3557.05 | 3557.80 | 3541.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 12:00:00 | 3552.00 | 3556.64 | 3542.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 13:15:00 | 3678.20 | 3699.90 | 3700.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 13:15:00 | 3678.20 | 3699.90 | 3700.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 3659.75 | 3691.87 | 3696.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 3625.00 | 3600.24 | 3633.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 3625.00 | 3600.24 | 3633.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 3625.00 | 3600.24 | 3633.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 3625.00 | 3600.24 | 3633.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 3606.55 | 3599.93 | 3612.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:45:00 | 3595.10 | 3599.93 | 3612.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 3660.00 | 3611.94 | 3616.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 3646.45 | 3611.94 | 3616.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 3632.95 | 3616.14 | 3618.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:15:00 | 3617.65 | 3616.14 | 3618.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:00:00 | 3600.75 | 3613.06 | 3616.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:00:00 | 3619.80 | 3592.97 | 3601.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 3629.95 | 3608.69 | 3608.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 3629.95 | 3608.69 | 3608.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 3662.85 | 3627.51 | 3618.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 3626.80 | 3640.25 | 3627.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 3626.80 | 3640.25 | 3627.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 3626.80 | 3640.25 | 3627.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 3626.80 | 3640.25 | 3627.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 3614.15 | 3635.03 | 3626.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:45:00 | 3616.05 | 3635.03 | 3626.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 3600.05 | 3628.03 | 3623.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 15:00:00 | 3600.05 | 3628.03 | 3623.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 3625.00 | 3627.43 | 3624.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 3648.05 | 3627.43 | 3624.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 3633.00 | 3626.25 | 3623.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 13:15:00 | 3606.55 | 3620.73 | 3621.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 13:15:00 | 3606.55 | 3620.73 | 3621.77 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 3650.00 | 3627.42 | 3624.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 3694.90 | 3649.30 | 3635.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 3647.15 | 3655.69 | 3644.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 3647.15 | 3655.69 | 3644.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 3647.15 | 3655.69 | 3644.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 3632.25 | 3655.69 | 3644.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 3676.00 | 3659.75 | 3647.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:15:00 | 3692.00 | 3662.41 | 3650.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:15:00 | 3714.00 | 3669.11 | 3658.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 3630.00 | 3661.01 | 3657.00 | SL hit (close<static) qty=1.00 sl=3632.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 3645.50 | 3653.75 | 3654.14 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 14:15:00 | 3666.00 | 3656.20 | 3655.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 3704.50 | 3668.07 | 3660.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 3650.00 | 3674.57 | 3669.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 3650.00 | 3674.57 | 3669.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 3650.00 | 3674.57 | 3669.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 3650.00 | 3674.57 | 3669.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 3644.45 | 3668.54 | 3667.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 3632.80 | 3668.54 | 3667.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 3680.00 | 3670.77 | 3668.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:45:00 | 3679.95 | 3670.77 | 3668.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 3664.95 | 3671.50 | 3669.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 3665.65 | 3668.63 | 3668.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 10:15:00 | 3643.15 | 3663.54 | 3665.88 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 3695.45 | 3669.62 | 3666.16 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 3645.05 | 3665.16 | 3665.49 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 3685.00 | 3669.07 | 3667.17 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 13:15:00 | 3652.65 | 3665.34 | 3665.78 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 3726.60 | 3676.37 | 3670.53 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 3653.00 | 3675.76 | 3675.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 3598.45 | 3653.34 | 3663.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 12:15:00 | 3598.50 | 3580.54 | 3606.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 12:45:00 | 3594.75 | 3580.54 | 3606.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 3590.10 | 3582.45 | 3605.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 3575.00 | 3579.45 | 3598.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 3563.50 | 3583.57 | 3598.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 13:15:00 | 3554.70 | 3541.64 | 3541.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 13:15:00 | 3554.70 | 3541.64 | 3541.03 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 14:15:00 | 3534.25 | 3540.16 | 3540.41 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 10:15:00 | 3558.15 | 3543.55 | 3541.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 11:15:00 | 3594.20 | 3553.68 | 3546.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 3758.65 | 3768.86 | 3717.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 09:30:00 | 3753.90 | 3768.86 | 3717.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 3715.20 | 3758.13 | 3717.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 3715.20 | 3758.13 | 3717.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 3678.00 | 3742.10 | 3713.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:45:00 | 3700.00 | 3742.10 | 3713.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 3684.25 | 3730.53 | 3710.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:45:00 | 3680.00 | 3730.53 | 3710.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 3675.15 | 3698.05 | 3700.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 3663.00 | 3691.04 | 3696.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3560.90 | 3524.08 | 3580.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 3560.90 | 3524.08 | 3580.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 3564.60 | 3537.92 | 3577.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:30:00 | 3576.15 | 3537.92 | 3577.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 3532.00 | 3511.26 | 3547.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:15:00 | 3553.85 | 3511.26 | 3547.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 3540.05 | 3517.02 | 3546.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 3543.35 | 3517.02 | 3546.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 3543.55 | 3522.32 | 3546.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 3543.55 | 3522.32 | 3546.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 3543.20 | 3526.50 | 3546.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:00:00 | 3543.20 | 3526.50 | 3546.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 3547.00 | 3530.60 | 3546.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 3547.00 | 3530.60 | 3546.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 3601.95 | 3544.87 | 3551.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:30:00 | 3609.45 | 3544.87 | 3551.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 3620.00 | 3559.90 | 3557.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 3637.70 | 3584.55 | 3569.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 3575.70 | 3586.45 | 3574.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 13:15:00 | 3575.70 | 3586.45 | 3574.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 3575.70 | 3586.45 | 3574.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:30:00 | 3591.25 | 3586.45 | 3574.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 3546.00 | 3578.36 | 3572.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 3546.00 | 3578.36 | 3572.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 3560.00 | 3574.69 | 3570.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 3577.90 | 3574.69 | 3570.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 12:15:00 | 3540.00 | 3569.15 | 3570.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 12:15:00 | 3540.00 | 3569.15 | 3570.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 3501.75 | 3547.89 | 3559.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 3268.70 | 3268.03 | 3331.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 3321.45 | 3290.89 | 3319.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 3321.45 | 3290.89 | 3319.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 3321.45 | 3290.89 | 3319.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 3330.00 | 3298.71 | 3320.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 3339.40 | 3298.71 | 3320.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 3307.10 | 3300.39 | 3319.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:30:00 | 3337.00 | 3300.39 | 3319.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 3305.65 | 3303.59 | 3317.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 11:30:00 | 3320.55 | 3303.59 | 3317.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 3290.40 | 3301.53 | 3314.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 14:45:00 | 3282.80 | 3296.62 | 3310.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 10:15:00 | 3289.50 | 3295.54 | 3307.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 14:30:00 | 3290.05 | 3288.71 | 3298.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 15:00:00 | 3287.35 | 3288.71 | 3298.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 3333.75 | 3298.23 | 3301.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 3333.75 | 3298.23 | 3301.51 | SL hit (close>static) qty=1.00 sl=3315.75 alert=retest2 |

### Cycle 27 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 11:15:00 | 3314.25 | 3303.85 | 3303.65 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 14:15:00 | 3300.25 | 3303.51 | 3303.63 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 15:15:00 | 3305.10 | 3303.83 | 3303.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 3351.10 | 3313.28 | 3308.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 3294.95 | 3321.70 | 3316.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 3294.95 | 3321.70 | 3316.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 3294.95 | 3321.70 | 3316.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 3294.95 | 3321.70 | 3316.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 3288.45 | 3315.05 | 3314.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:45:00 | 3289.90 | 3315.05 | 3314.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 3304.00 | 3312.84 | 3313.42 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 3336.10 | 3308.23 | 3307.18 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 14:15:00 | 3295.20 | 3305.18 | 3306.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 15:15:00 | 3280.00 | 3300.15 | 3303.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 10:15:00 | 3297.00 | 3294.52 | 3300.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 10:15:00 | 3297.00 | 3294.52 | 3300.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 3297.00 | 3294.52 | 3300.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 3297.00 | 3294.52 | 3300.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 3299.65 | 3292.42 | 3297.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:00:00 | 3299.65 | 3292.42 | 3297.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 3289.90 | 3291.91 | 3297.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 15:15:00 | 3281.00 | 3291.91 | 3297.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 3281.90 | 3289.21 | 3294.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:00:00 | 3284.40 | 3279.24 | 3285.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 14:15:00 | 3300.00 | 3289.40 | 3288.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 3300.00 | 3289.40 | 3288.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 3320.00 | 3296.90 | 3292.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 13:15:00 | 3296.70 | 3299.76 | 3295.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 13:15:00 | 3296.70 | 3299.76 | 3295.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 3296.70 | 3299.76 | 3295.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:30:00 | 3290.00 | 3299.76 | 3295.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 3295.05 | 3298.82 | 3295.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 3295.05 | 3298.82 | 3295.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 3296.00 | 3298.25 | 3295.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 3314.75 | 3298.25 | 3295.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 3312.80 | 3301.16 | 3296.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:30:00 | 3330.05 | 3311.21 | 3302.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 15:15:00 | 3331.05 | 3320.76 | 3309.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:30:00 | 3330.35 | 3345.58 | 3340.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 3297.80 | 3333.73 | 3336.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 3297.80 | 3333.73 | 3336.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 3252.35 | 3311.26 | 3325.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 13:15:00 | 3315.20 | 3302.33 | 3315.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 13:15:00 | 3315.20 | 3302.33 | 3315.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 3315.20 | 3302.33 | 3315.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:00:00 | 3315.20 | 3302.33 | 3315.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 3347.60 | 3311.38 | 3318.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 3347.60 | 3311.38 | 3318.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 3348.30 | 3318.77 | 3321.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 3373.65 | 3318.77 | 3321.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 3398.00 | 3334.61 | 3328.02 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 3301.70 | 3338.37 | 3341.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 09:15:00 | 3276.05 | 3320.56 | 3332.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 15:15:00 | 3290.00 | 3289.30 | 3308.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-13 09:15:00 | 3290.00 | 3289.30 | 3308.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 3285.00 | 3288.44 | 3306.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 12:45:00 | 3277.95 | 3286.17 | 3300.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 13:15:00 | 3277.10 | 3286.17 | 3300.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 14:15:00 | 3274.00 | 3284.63 | 3298.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 09:45:00 | 3265.90 | 3278.69 | 3292.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 3287.00 | 3277.63 | 3288.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:00:00 | 3287.00 | 3277.63 | 3288.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 3283.00 | 3278.70 | 3287.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:30:00 | 3290.00 | 3278.70 | 3287.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 3253.00 | 3272.44 | 3282.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:45:00 | 3245.40 | 3258.61 | 3268.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 11:15:00 | 3323.00 | 3256.35 | 3247.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 3323.00 | 3256.35 | 3247.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 3342.00 | 3304.96 | 3279.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 3371.55 | 3383.03 | 3358.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:00:00 | 3371.55 | 3383.03 | 3358.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 3362.50 | 3378.92 | 3359.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 3362.50 | 3378.92 | 3359.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 3365.15 | 3376.17 | 3359.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 3365.15 | 3376.17 | 3359.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 3366.30 | 3375.25 | 3363.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 3373.80 | 3375.25 | 3363.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 3366.60 | 3373.52 | 3363.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 3366.60 | 3373.52 | 3363.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 3364.80 | 3371.78 | 3363.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 3364.80 | 3371.78 | 3363.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 3353.25 | 3368.07 | 3362.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 3353.25 | 3368.07 | 3362.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 3343.20 | 3363.10 | 3361.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:30:00 | 3343.45 | 3363.10 | 3361.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 3438.05 | 3423.78 | 3401.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 3401.15 | 3423.78 | 3401.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 3487.25 | 3535.86 | 3498.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 3487.25 | 3535.86 | 3498.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 3441.25 | 3516.94 | 3493.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:30:00 | 3433.60 | 3516.94 | 3493.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 3455.00 | 3476.53 | 3479.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 3378.95 | 3432.09 | 3453.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 3370.00 | 3355.55 | 3392.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 3370.00 | 3355.55 | 3392.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 3370.60 | 3355.09 | 3375.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 3370.60 | 3355.09 | 3375.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 3393.00 | 3362.67 | 3376.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 3392.20 | 3362.67 | 3376.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 3377.50 | 3365.64 | 3376.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:15:00 | 3366.85 | 3367.14 | 3376.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:30:00 | 3369.90 | 3368.25 | 3375.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 15:00:00 | 3361.90 | 3368.25 | 3375.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 3418.00 | 3376.83 | 3378.10 | SL hit (close>static) qty=1.00 sl=3396.95 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 11:15:00 | 3389.55 | 3380.89 | 3379.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 12:15:00 | 3403.30 | 3385.37 | 3381.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 12:15:00 | 3397.65 | 3398.29 | 3390.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 13:00:00 | 3397.65 | 3398.29 | 3390.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 3388.90 | 3396.41 | 3390.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:30:00 | 3388.90 | 3396.41 | 3390.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 3383.70 | 3393.87 | 3390.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 3383.70 | 3393.87 | 3390.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 3397.35 | 3394.56 | 3390.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 3367.00 | 3394.56 | 3390.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 3323.45 | 3380.34 | 3384.66 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 11:15:00 | 3386.30 | 3365.57 | 3363.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 13:15:00 | 3430.55 | 3382.55 | 3371.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 3406.20 | 3412.46 | 3392.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 10:15:00 | 3406.20 | 3412.46 | 3392.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 3406.20 | 3412.46 | 3392.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:30:00 | 3400.25 | 3412.46 | 3392.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 3345.30 | 3399.02 | 3388.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 3345.30 | 3399.02 | 3388.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 3323.35 | 3383.89 | 3382.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:45:00 | 3329.35 | 3383.89 | 3382.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 3326.65 | 3372.44 | 3377.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 3300.75 | 3346.15 | 3363.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 3370.50 | 3346.20 | 3359.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 3370.50 | 3346.20 | 3359.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 3370.50 | 3346.20 | 3359.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 3370.50 | 3346.20 | 3359.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 3363.40 | 3349.64 | 3360.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:15:00 | 3375.90 | 3349.64 | 3360.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 3375.05 | 3354.72 | 3361.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 3375.05 | 3354.72 | 3361.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 3350.55 | 3359.65 | 3362.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:45:00 | 3332.70 | 3358.02 | 3361.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 3328.70 | 3352.16 | 3358.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:30:00 | 3332.85 | 3301.03 | 3302.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 15:00:00 | 3311.50 | 3301.03 | 3302.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 3517.00 | 3343.69 | 3321.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 09:15:00 | 3517.00 | 3343.69 | 3321.81 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 3292.70 | 3352.85 | 3354.43 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 3383.15 | 3351.31 | 3350.11 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 3303.40 | 3347.59 | 3350.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 3284.70 | 3315.72 | 3328.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 3333.85 | 3319.06 | 3327.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 14:15:00 | 3333.85 | 3319.06 | 3327.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 3333.85 | 3319.06 | 3327.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:00:00 | 3333.85 | 3319.06 | 3327.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 3330.30 | 3321.30 | 3327.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 3357.75 | 3328.59 | 3330.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 3350.00 | 3332.88 | 3332.42 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 3276.45 | 3321.59 | 3327.33 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 3376.35 | 3327.68 | 3321.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 3455.00 | 3358.63 | 3336.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 13:15:00 | 3530.00 | 3533.15 | 3495.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:45:00 | 3531.85 | 3533.15 | 3495.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 3473.85 | 3522.81 | 3500.52 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 3460.00 | 3491.77 | 3494.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 10:15:00 | 3442.70 | 3476.31 | 3486.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 3337.00 | 3306.83 | 3355.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 3337.00 | 3306.83 | 3355.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 3330.50 | 3288.53 | 3306.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 3330.50 | 3288.53 | 3306.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 3346.95 | 3300.21 | 3310.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 3346.95 | 3300.21 | 3310.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 3379.20 | 3323.58 | 3319.85 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 3285.00 | 3314.79 | 3318.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 13:15:00 | 3267.00 | 3301.34 | 3311.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 3315.95 | 3290.09 | 3300.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 11:15:00 | 3315.95 | 3290.09 | 3300.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 3315.95 | 3290.09 | 3300.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 3315.95 | 3290.09 | 3300.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 3312.50 | 3294.57 | 3301.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 3312.50 | 3294.57 | 3301.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 3330.20 | 3301.70 | 3303.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 3330.20 | 3301.70 | 3303.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 3326.30 | 3306.62 | 3305.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 3408.15 | 3330.58 | 3317.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 3467.45 | 3480.87 | 3452.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 11:00:00 | 3467.45 | 3480.87 | 3452.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 3498.80 | 3484.46 | 3457.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 3500.05 | 3479.98 | 3463.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 3508.95 | 3509.09 | 3488.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 15:15:00 | 3559.50 | 3582.49 | 3584.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 15:15:00 | 3559.50 | 3582.49 | 3584.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 3545.30 | 3575.05 | 3581.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 3587.05 | 3556.78 | 3566.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 3587.05 | 3556.78 | 3566.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 3587.05 | 3556.78 | 3566.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 3587.05 | 3556.78 | 3566.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 3598.05 | 3565.03 | 3569.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 3587.60 | 3565.03 | 3569.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 3620.05 | 3576.04 | 3573.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 12:15:00 | 3632.65 | 3587.36 | 3579.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 11:15:00 | 3609.45 | 3618.10 | 3602.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 12:00:00 | 3609.45 | 3618.10 | 3602.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 3593.80 | 3613.24 | 3601.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:45:00 | 3597.00 | 3613.24 | 3601.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 3609.30 | 3612.45 | 3602.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 3616.35 | 3608.64 | 3601.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 3586.35 | 3604.18 | 3600.48 | SL hit (close<static) qty=1.00 sl=3591.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 3571.80 | 3596.39 | 3597.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 12:15:00 | 3563.00 | 3589.72 | 3594.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 3552.80 | 3544.09 | 3562.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 3552.80 | 3544.09 | 3562.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 3552.80 | 3544.09 | 3562.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 3552.80 | 3544.09 | 3562.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 3524.00 | 3542.28 | 3558.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 3511.55 | 3542.28 | 3558.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:15:00 | 3512.05 | 3538.31 | 3555.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 3505.25 | 3517.73 | 3536.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 3335.97 | 3374.39 | 3413.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 3336.45 | 3374.39 | 3413.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 3329.99 | 3374.39 | 3413.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 3409.00 | 3339.39 | 3360.47 | SL hit (close>ema200) qty=0.50 sl=3339.39 alert=retest2 |

### Cycle 57 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 3385.30 | 3372.02 | 3371.15 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 15:15:00 | 3357.50 | 3369.12 | 3369.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 10:15:00 | 3343.55 | 3362.22 | 3366.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 10:15:00 | 3347.00 | 3336.28 | 3347.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 10:15:00 | 3347.00 | 3336.28 | 3347.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 3347.00 | 3336.28 | 3347.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 3336.15 | 3336.28 | 3347.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 3322.70 | 3333.56 | 3344.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 15:15:00 | 3301.00 | 3327.42 | 3339.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 12:15:00 | 3277.00 | 3262.09 | 3262.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 3277.00 | 3262.09 | 3262.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 3290.45 | 3267.76 | 3264.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 3341.35 | 3341.72 | 3314.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 3341.35 | 3341.72 | 3314.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 3334.10 | 3341.34 | 3321.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 3337.30 | 3341.34 | 3321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 3463.25 | 3369.58 | 3339.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:30:00 | 3476.45 | 3386.66 | 3350.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 11:30:00 | 3468.50 | 3400.72 | 3359.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 13:30:00 | 3478.85 | 3426.99 | 3379.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 13:45:00 | 3473.90 | 3462.61 | 3425.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-09 09:15:00 | 3824.10 | 3559.63 | 3479.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 3602.60 | 3631.32 | 3634.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 3581.40 | 3621.33 | 3629.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 14:15:00 | 3632.05 | 3621.41 | 3627.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 14:15:00 | 3632.05 | 3621.41 | 3627.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 3632.05 | 3621.41 | 3627.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 3632.05 | 3621.41 | 3627.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 3650.00 | 3627.13 | 3629.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 3619.55 | 3627.13 | 3629.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 3593.00 | 3620.30 | 3626.44 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 3695.65 | 3641.87 | 3635.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 3710.90 | 3670.22 | 3651.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 3745.75 | 3754.78 | 3720.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 3745.75 | 3754.78 | 3720.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 3735.30 | 3750.88 | 3721.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 3745.65 | 3750.88 | 3721.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 3716.00 | 3742.00 | 3728.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 3711.05 | 3742.00 | 3728.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 3742.15 | 3742.03 | 3729.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:15:00 | 3771.00 | 3745.52 | 3734.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 3670.15 | 3770.48 | 3764.11 | SL hit (close<static) qty=1.00 sl=3716.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 3675.40 | 3751.46 | 3756.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 3656.25 | 3732.42 | 3746.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 3745.55 | 3707.58 | 3725.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 3745.55 | 3707.58 | 3725.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 3745.55 | 3707.58 | 3725.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 3745.55 | 3707.58 | 3725.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 3724.75 | 3711.01 | 3725.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 3748.85 | 3711.01 | 3725.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 3720.75 | 3712.96 | 3725.01 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 3778.40 | 3733.11 | 3731.81 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 3717.30 | 3730.23 | 3731.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 3704.25 | 3725.03 | 3728.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 3599.95 | 3599.03 | 3639.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 3599.95 | 3599.03 | 3639.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 3622.35 | 3589.35 | 3620.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 3622.35 | 3589.35 | 3620.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 3629.25 | 3597.33 | 3621.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 3629.25 | 3597.33 | 3621.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 3648.65 | 3607.59 | 3624.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:15:00 | 3685.00 | 3607.59 | 3624.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 3688.10 | 3623.69 | 3630.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:00:00 | 3688.10 | 3623.69 | 3630.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 3755.00 | 3649.95 | 3641.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 3808.35 | 3707.57 | 3672.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 4093.55 | 4109.07 | 4002.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:45:00 | 4069.85 | 4109.07 | 4002.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 4237.70 | 4147.94 | 4061.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 4246.00 | 4166.27 | 4078.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:30:00 | 4243.10 | 4175.62 | 4090.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 10:30:00 | 4243.15 | 4196.61 | 4157.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:00:00 | 4245.00 | 4206.29 | 4165.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 4196.30 | 4234.04 | 4216.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 4173.55 | 4204.04 | 4208.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 4173.55 | 4204.04 | 4208.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 4142.65 | 4191.77 | 4202.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 4084.70 | 4083.29 | 4133.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 4084.70 | 4083.29 | 4133.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 4120.95 | 4097.72 | 4122.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 4050.40 | 4097.72 | 4122.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 4131.50 | 4104.47 | 4123.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:00:00 | 4011.15 | 4083.88 | 4105.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:00:00 | 4020.60 | 4023.38 | 4062.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:00:00 | 4006.55 | 4018.66 | 4053.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 4113.60 | 4064.32 | 4064.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 4113.60 | 4064.32 | 4064.28 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 10:15:00 | 4048.80 | 4072.71 | 4073.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 15:15:00 | 4025.35 | 4047.23 | 4055.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 13:15:00 | 4012.45 | 4005.48 | 4027.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 13:45:00 | 4004.45 | 4005.48 | 4027.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 4038.90 | 4012.16 | 4028.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:00:00 | 4038.90 | 4012.16 | 4028.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 4037.00 | 4017.13 | 4029.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 4005.95 | 4017.13 | 4029.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 14:15:00 | 3805.65 | 3874.20 | 3928.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-27 12:15:00 | 3605.36 | 3742.26 | 3837.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 3834.45 | 3762.85 | 3762.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 3875.90 | 3796.05 | 3778.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 4104.35 | 4123.83 | 4058.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:00:00 | 4104.35 | 4123.83 | 4058.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 4071.65 | 4102.80 | 4064.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 4243.75 | 4102.80 | 4064.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:45:00 | 4089.90 | 4097.09 | 4079.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 4020.35 | 4077.09 | 4072.91 | SL hit (close<static) qty=1.00 sl=4063.55 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 4000.00 | 4061.67 | 4066.28 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 4117.50 | 4071.78 | 4068.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 15:15:00 | 4170.00 | 4091.43 | 4078.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 14:15:00 | 4105.00 | 4113.43 | 4097.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-12 15:00:00 | 4105.00 | 4113.43 | 4097.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 4109.00 | 4112.54 | 4098.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 4055.00 | 4112.54 | 4098.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 4120.40 | 4114.11 | 4100.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 4177.00 | 4103.76 | 4100.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:45:00 | 4135.90 | 4138.91 | 4122.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 4130.50 | 4137.23 | 4123.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 10:15:00 | 4082.80 | 4133.55 | 4135.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 10:15:00 | 4082.80 | 4133.55 | 4135.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-19 13:15:00 | 4049.00 | 4101.98 | 4119.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 4090.10 | 4080.56 | 4101.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 11:00:00 | 4090.10 | 4080.56 | 4101.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 4144.00 | 4093.24 | 4105.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:00:00 | 4144.00 | 4093.24 | 4105.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 4143.50 | 4103.30 | 4109.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:30:00 | 4148.30 | 4103.30 | 4109.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 13:15:00 | 4160.95 | 4114.83 | 4113.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 4209.10 | 4151.17 | 4132.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 11:15:00 | 4219.20 | 4236.89 | 4201.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 11:30:00 | 4214.85 | 4236.89 | 4201.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 4199.05 | 4229.32 | 4201.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:00:00 | 4199.05 | 4229.32 | 4201.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 4201.05 | 4223.66 | 4201.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 13:45:00 | 4199.35 | 4223.66 | 4201.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 4194.00 | 4217.73 | 4200.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 15:00:00 | 4194.00 | 4217.73 | 4200.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 4200.00 | 4214.19 | 4200.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 4186.45 | 4214.19 | 4200.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 4179.05 | 4207.16 | 4198.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 4189.60 | 4207.16 | 4198.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 4157.95 | 4197.32 | 4195.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 4157.95 | 4197.32 | 4195.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 4139.95 | 4185.84 | 4190.14 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 13:15:00 | 4213.80 | 4193.52 | 4193.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 4245.00 | 4204.76 | 4198.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 10:15:00 | 4203.00 | 4204.41 | 4198.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 10:15:00 | 4203.00 | 4204.41 | 4198.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 4203.00 | 4204.41 | 4198.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:45:00 | 4214.75 | 4204.41 | 4198.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 4198.75 | 4203.27 | 4198.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 4226.50 | 4201.41 | 4199.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:45:00 | 4260.85 | 4211.13 | 4203.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 11:00:00 | 4213.95 | 4211.69 | 4204.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 15:15:00 | 4228.00 | 4219.48 | 4211.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 4228.00 | 4221.18 | 4213.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 4322.35 | 4221.18 | 4213.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 4195.00 | 4220.80 | 4214.56 | SL hit (close<static) qty=1.00 sl=4200.20 alert=retest2 |

### Cycle 76 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 4206.30 | 4210.79 | 4211.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 4183.95 | 4197.95 | 4203.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 4199.70 | 4197.15 | 4201.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:00:00 | 4199.70 | 4197.15 | 4201.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 4200.55 | 4197.83 | 4201.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 4203.50 | 4197.83 | 4201.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 4214.85 | 4201.23 | 4202.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 4214.85 | 4201.23 | 4202.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 4244.10 | 4209.81 | 4206.73 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 3997.70 | 4173.58 | 4191.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 11:15:00 | 3975.25 | 4106.94 | 4156.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 13:15:00 | 4094.65 | 4085.99 | 4136.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 13:45:00 | 4093.50 | 4085.99 | 4136.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 4199.85 | 4108.76 | 4142.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:30:00 | 4177.20 | 4108.76 | 4142.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 4189.20 | 4124.85 | 4146.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 4145.00 | 4124.85 | 4146.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:45:00 | 4143.80 | 4128.84 | 4146.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 3730.50 | 4024.04 | 4083.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 15:15:00 | 3980.00 | 3928.78 | 3925.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 4101.05 | 3963.23 | 3941.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 4253.00 | 4261.69 | 4218.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 4253.00 | 4261.69 | 4218.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 4218.00 | 4248.46 | 4219.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:30:00 | 4223.00 | 4248.46 | 4219.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 4216.90 | 4242.15 | 4219.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:30:00 | 4268.90 | 4231.37 | 4221.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-05 11:15:00 | 4695.79 | 4585.75 | 4534.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 4555.00 | 4612.57 | 4614.91 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 10:15:00 | 4639.90 | 4619.94 | 4617.98 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 4602.00 | 4616.35 | 4616.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 12:15:00 | 4581.30 | 4609.34 | 4613.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 4610.90 | 4609.66 | 4613.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 13:15:00 | 4610.90 | 4609.66 | 4613.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 4610.90 | 4609.66 | 4613.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:00:00 | 4610.90 | 4609.66 | 4613.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 4587.60 | 4605.24 | 4610.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 12:15:00 | 4547.00 | 4587.92 | 4600.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 12:15:00 | 4319.65 | 4410.23 | 4486.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 4344.00 | 4308.23 | 4363.98 | SL hit (close>ema200) qty=0.50 sl=4308.23 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 4372.30 | 4303.99 | 4297.18 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 4253.70 | 4295.26 | 4299.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 4244.10 | 4270.85 | 4284.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 4339.70 | 4273.65 | 4278.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 4339.70 | 4273.65 | 4278.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 4339.70 | 4273.65 | 4278.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 4339.70 | 4273.65 | 4278.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 4320.30 | 4282.98 | 4282.59 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 4269.00 | 4286.70 | 4286.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 10:15:00 | 4266.00 | 4279.93 | 4283.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 4273.40 | 4271.57 | 4277.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 4273.40 | 4271.57 | 4277.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 4273.40 | 4271.57 | 4277.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 4273.40 | 4271.57 | 4277.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 4268.80 | 4271.01 | 4277.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 4276.80 | 4271.01 | 4277.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 4267.60 | 4270.33 | 4276.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:00:00 | 4246.10 | 4263.60 | 4269.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 4291.00 | 4273.61 | 4273.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 15:15:00 | 4291.00 | 4273.61 | 4273.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 4314.60 | 4291.64 | 4282.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 13:15:00 | 4293.80 | 4333.56 | 4312.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 13:15:00 | 4293.80 | 4333.56 | 4312.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 4293.80 | 4333.56 | 4312.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:00:00 | 4293.80 | 4333.56 | 4312.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 4255.20 | 4317.89 | 4307.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:30:00 | 4254.50 | 4317.89 | 4307.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 4267.70 | 4298.43 | 4299.80 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 4337.50 | 4302.33 | 4300.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 4390.00 | 4327.51 | 4312.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 4374.70 | 4385.32 | 4351.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 4374.70 | 4385.32 | 4351.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 4433.50 | 4458.04 | 4437.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 4433.50 | 4458.04 | 4437.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 4452.10 | 4456.85 | 4438.43 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 4381.50 | 4433.25 | 4433.44 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 4456.50 | 4429.32 | 4427.37 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 4373.00 | 4421.03 | 4424.69 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 4466.00 | 4413.30 | 4413.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 10:15:00 | 4560.00 | 4442.64 | 4426.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 4510.40 | 4514.76 | 4484.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:00:00 | 4510.40 | 4514.76 | 4484.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 4485.00 | 4516.20 | 4495.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 4485.00 | 4516.20 | 4495.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 4480.00 | 4508.96 | 4494.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:15:00 | 4470.40 | 4508.96 | 4494.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 4491.10 | 4494.98 | 4490.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 4523.70 | 4494.98 | 4490.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 4685.80 | 4497.97 | 4492.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 15:15:00 | 4641.00 | 4664.04 | 4667.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 4641.00 | 4664.04 | 4667.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 4532.50 | 4627.81 | 4648.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 15:15:00 | 4639.90 | 4628.91 | 4645.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 15:15:00 | 4639.90 | 4628.91 | 4645.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 4639.90 | 4628.91 | 4645.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:15:00 | 4623.80 | 4628.91 | 4645.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 4659.20 | 4634.97 | 4646.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 4659.20 | 4634.97 | 4646.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 4730.40 | 4654.06 | 4653.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 4759.00 | 4675.05 | 4663.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 4699.30 | 4699.59 | 4679.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:00:00 | 4699.30 | 4699.59 | 4679.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 4699.80 | 4699.69 | 4683.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 4699.80 | 4699.69 | 4683.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 4708.10 | 4700.34 | 4686.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:15:00 | 4687.40 | 4700.34 | 4686.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 4725.50 | 4705.37 | 4689.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 4766.70 | 4709.15 | 4695.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:30:00 | 4750.00 | 4746.00 | 4715.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 4971.50 | 5005.89 | 5007.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 4971.50 | 5005.89 | 5007.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 4918.30 | 4981.59 | 4996.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 5077.30 | 4990.24 | 4996.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 14:15:00 | 5077.30 | 4990.24 | 4996.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 5077.30 | 4990.24 | 4996.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 5077.30 | 4990.24 | 4996.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 15:15:00 | 5080.00 | 5008.19 | 5004.46 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 14:15:00 | 4974.90 | 5004.76 | 5006.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 15:15:00 | 4959.00 | 4995.61 | 5002.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 5013.70 | 4999.22 | 5003.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 5013.70 | 4999.22 | 5003.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 5013.70 | 4999.22 | 5003.61 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 5031.50 | 5010.18 | 5007.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 5084.90 | 5029.36 | 5017.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 4964.20 | 5025.55 | 5018.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 4964.20 | 5025.55 | 5018.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 4964.20 | 5025.55 | 5018.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 4964.20 | 5025.55 | 5018.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 4964.80 | 5013.40 | 5013.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:15:00 | 4956.30 | 5013.40 | 5013.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 11:15:00 | 4970.60 | 5004.84 | 5009.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 4922.10 | 4966.04 | 4980.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 4909.10 | 4900.52 | 4933.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 4909.10 | 4900.52 | 4933.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 4929.40 | 4904.23 | 4926.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 4925.10 | 4904.23 | 4926.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 4929.10 | 4909.20 | 4926.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:15:00 | 4930.10 | 4909.20 | 4926.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 4959.10 | 4919.18 | 4929.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 4959.10 | 4919.18 | 4929.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 4994.10 | 4934.16 | 4935.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 4989.40 | 4934.16 | 4935.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 4992.40 | 4945.81 | 4940.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 5056.90 | 5002.98 | 4979.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 5074.90 | 5076.29 | 5037.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 15:00:00 | 5074.90 | 5076.29 | 5037.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 5052.60 | 5083.27 | 5053.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 5052.60 | 5083.27 | 5053.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 5054.60 | 5077.54 | 5053.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:15:00 | 5016.20 | 5077.54 | 5053.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 5005.30 | 5063.09 | 5049.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 5021.60 | 5063.09 | 5049.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 5003.50 | 5051.17 | 5045.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 5009.70 | 5051.17 | 5045.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 5069.80 | 5084.55 | 5069.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 5127.00 | 5093.60 | 5077.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:30:00 | 5127.00 | 5106.23 | 5086.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:00:00 | 5133.70 | 5118.83 | 5099.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 5051.00 | 5211.52 | 5185.85 | SL hit (close<static) qty=1.00 sl=5066.20 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 5060.00 | 5155.69 | 5163.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 4967.50 | 5118.06 | 5145.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 5103.50 | 5052.72 | 5093.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 5103.50 | 5052.72 | 5093.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 5103.50 | 5052.72 | 5093.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 5103.50 | 5052.72 | 5093.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 5090.00 | 5060.18 | 5093.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 5111.50 | 5060.18 | 5093.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 5105.50 | 5069.24 | 5094.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 5105.50 | 5069.24 | 5094.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 5096.00 | 5074.59 | 5094.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 5099.00 | 5074.59 | 5094.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 5104.00 | 5080.47 | 5095.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 5107.00 | 5080.47 | 5095.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 5102.50 | 5084.88 | 5095.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 5102.50 | 5084.88 | 5095.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 5152.00 | 5100.24 | 5101.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 5152.00 | 5100.24 | 5101.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 5147.00 | 5109.59 | 5105.29 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 5039.50 | 5110.04 | 5114.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 4995.00 | 5068.28 | 5092.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 4833.50 | 4830.48 | 4878.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 09:30:00 | 4810.50 | 4830.48 | 4878.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 4870.50 | 4838.49 | 4877.46 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 4957.00 | 4888.85 | 4887.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 5035.00 | 4952.20 | 4925.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 5057.50 | 5062.34 | 5013.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 5057.50 | 5062.34 | 5013.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 5020.00 | 5053.87 | 5014.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 5129.50 | 5053.87 | 5014.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:15:00 | 5112.00 | 5061.60 | 5021.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 5097.50 | 5077.56 | 5036.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 5016.00 | 5032.49 | 5032.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 5016.00 | 5032.49 | 5032.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 15:15:00 | 4981.00 | 5022.19 | 5027.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 5015.00 | 5014.00 | 5022.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:00:00 | 5015.00 | 5014.00 | 5022.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 4977.00 | 4993.28 | 5008.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 4997.50 | 4993.28 | 5008.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 4965.00 | 4987.62 | 5004.26 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 15:15:00 | 5030.00 | 5013.64 | 5011.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 5140.50 | 5039.01 | 5023.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 12:15:00 | 4994.00 | 5044.21 | 5030.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 4994.00 | 5044.21 | 5030.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 4994.00 | 5044.21 | 5030.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 4997.50 | 5044.21 | 5030.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 5017.50 | 5038.87 | 5029.63 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 5002.00 | 5023.92 | 5023.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 4896.00 | 4998.33 | 5012.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 4700.00 | 4696.30 | 4758.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 09:15:00 | 4633.40 | 4696.30 | 4758.59 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 15:00:00 | 4669.90 | 4649.42 | 4701.08 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 09:15:00 | 4670.00 | 4655.73 | 4699.26 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 4675.20 | 4659.63 | 4697.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 4707.70 | 4669.24 | 4698.04 | SL hit (close>ema400) qty=1.00 sl=4698.04 alert=retest1 |

### Cycle 109 — BUY (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 15:15:00 | 4757.00 | 4714.27 | 4711.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 4829.00 | 4737.21 | 4721.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 4822.90 | 4828.35 | 4790.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:45:00 | 4827.80 | 4828.35 | 4790.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 4780.30 | 4814.20 | 4790.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 4780.30 | 4814.20 | 4790.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 4769.80 | 4805.32 | 4788.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:15:00 | 4759.90 | 4805.32 | 4788.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 4759.90 | 4796.24 | 4786.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 4761.00 | 4796.24 | 4786.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 4754.00 | 4780.31 | 4780.23 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 4730.70 | 4770.39 | 4775.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 4688.70 | 4754.05 | 4767.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 4690.20 | 4678.08 | 4704.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 4690.20 | 4678.08 | 4704.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 4690.20 | 4678.08 | 4704.20 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 4770.60 | 4710.17 | 4710.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 4782.80 | 4733.95 | 4721.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 15:15:00 | 4740.00 | 4775.99 | 4752.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 15:15:00 | 4740.00 | 4775.99 | 4752.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 4740.00 | 4775.99 | 4752.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 4817.00 | 4775.72 | 4764.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 4813.90 | 4787.87 | 4772.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 4817.00 | 4789.30 | 4776.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 4823.10 | 4788.02 | 4782.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 4838.90 | 4798.20 | 4787.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:00:00 | 4847.10 | 4807.98 | 4792.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 4840.80 | 4815.22 | 4797.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:45:00 | 4868.00 | 4831.90 | 4810.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:00:00 | 4843.10 | 4886.34 | 4883.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 4851.50 | 4879.37 | 4880.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 4851.50 | 4879.37 | 4880.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 4799.80 | 4854.95 | 4868.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 4738.80 | 4721.20 | 4766.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 4738.80 | 4721.20 | 4766.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 4710.00 | 4661.25 | 4688.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:45:00 | 4719.90 | 4661.25 | 4688.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 4734.10 | 4675.82 | 4692.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 4734.10 | 4675.82 | 4692.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 4586.60 | 4564.71 | 4587.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:45:00 | 4590.80 | 4564.71 | 4587.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 4624.00 | 4576.57 | 4590.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 4624.00 | 4576.57 | 4590.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 4622.00 | 4585.65 | 4593.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 4641.10 | 4585.65 | 4593.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 4625.70 | 4599.63 | 4598.39 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 4588.40 | 4606.29 | 4606.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 12:15:00 | 4570.20 | 4599.07 | 4603.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 13:15:00 | 4572.80 | 4564.76 | 4579.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 4572.80 | 4564.76 | 4579.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 4572.10 | 4568.34 | 4578.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 4586.40 | 4568.34 | 4578.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 4635.90 | 4581.85 | 4583.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 4635.90 | 4581.85 | 4583.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 4693.40 | 4604.16 | 4593.53 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 4577.80 | 4600.84 | 4603.91 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 4695.00 | 4615.65 | 4609.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 4726.00 | 4637.72 | 4619.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 4880.00 | 4898.76 | 4845.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 14:00:00 | 4880.00 | 4898.76 | 4845.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 5090.30 | 5126.69 | 5090.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 5102.80 | 5126.69 | 5090.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 5081.60 | 5117.67 | 5090.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:00:00 | 5081.60 | 5117.67 | 5090.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 5142.90 | 5122.72 | 5094.86 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 5012.20 | 5078.00 | 5080.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 5002.50 | 5062.90 | 5073.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 15:15:00 | 5080.00 | 5060.77 | 5068.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 15:15:00 | 5080.00 | 5060.77 | 5068.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 5080.00 | 5060.77 | 5068.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 5175.20 | 5060.77 | 5068.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 5125.10 | 5073.63 | 5073.43 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 5055.20 | 5073.58 | 5074.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 5023.30 | 5063.53 | 5069.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 5064.40 | 5063.12 | 5068.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 5064.40 | 5063.12 | 5068.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 5064.40 | 5063.12 | 5068.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 5092.90 | 5063.12 | 5068.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 5054.20 | 5061.34 | 5067.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 5020.90 | 5051.07 | 5061.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 5722.80 | 5120.62 | 5054.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 5722.80 | 5120.62 | 5054.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 5972.50 | 5621.24 | 5387.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 6038.50 | 6058.57 | 5953.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:15:00 | 6028.00 | 6058.57 | 5953.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 5969.50 | 6031.30 | 5959.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 5969.50 | 6031.30 | 5959.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 5974.00 | 6010.83 | 5961.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 5959.00 | 6010.83 | 5961.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 5969.00 | 6002.47 | 5962.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 5967.00 | 6002.47 | 5962.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 5965.00 | 5994.97 | 5962.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 6025.50 | 5994.97 | 5962.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 5997.50 | 5995.48 | 5965.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 6060.50 | 5986.07 | 5972.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 10:00:00 | 6046.00 | 5998.05 | 5979.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 10:30:00 | 6048.50 | 6001.44 | 5982.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:30:00 | 6046.50 | 6012.49 | 5996.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 6010.00 | 6015.75 | 6000.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 6005.00 | 6015.75 | 6000.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 6000.00 | 6012.60 | 6000.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 6000.00 | 6012.60 | 6000.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 6025.50 | 6015.18 | 6003.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:30:00 | 6006.50 | 6015.18 | 6003.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 6015.50 | 6015.94 | 6005.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 6084.50 | 6015.94 | 6005.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:15:00 | 6043.00 | 6020.75 | 6008.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 6044.00 | 6023.90 | 6011.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:45:00 | 6043.50 | 6030.42 | 6015.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 6018.50 | 6031.83 | 6022.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 6018.50 | 6031.83 | 6022.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 6040.00 | 6033.46 | 6024.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 6075.00 | 6026.77 | 6024.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 6017.00 | 6038.18 | 6032.50 | SL hit (close<static) qty=1.00 sl=6018.50 alert=retest2 |

### Cycle 122 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 5991.00 | 6022.08 | 6026.08 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 6038.50 | 6021.43 | 6021.19 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 6004.50 | 6018.04 | 6019.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 10:15:00 | 5989.50 | 6012.05 | 6016.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 5968.00 | 5943.84 | 5970.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 13:15:00 | 5968.00 | 5943.84 | 5970.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 5968.00 | 5943.84 | 5970.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 5968.00 | 5943.84 | 5970.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 5982.00 | 5951.47 | 5971.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:45:00 | 5943.50 | 5951.47 | 5971.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 5982.00 | 5957.58 | 5972.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 6008.00 | 5957.58 | 5972.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 5989.50 | 5972.11 | 5977.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 6002.50 | 5972.11 | 5977.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 5913.50 | 5960.39 | 5971.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 5898.00 | 5941.61 | 5961.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 5818.00 | 5784.09 | 5779.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 5818.00 | 5784.09 | 5779.75 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 5775.00 | 5777.00 | 5777.17 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 5815.00 | 5784.60 | 5780.61 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 5733.00 | 5770.75 | 5774.79 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 5822.50 | 5784.18 | 5780.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 13:15:00 | 5832.50 | 5793.84 | 5785.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 5777.50 | 5800.03 | 5790.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 5777.50 | 5800.03 | 5790.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 5777.50 | 5800.03 | 5790.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 5777.00 | 5800.03 | 5790.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 5788.50 | 5797.72 | 5790.76 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 5731.00 | 5782.50 | 5784.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 5642.50 | 5742.06 | 5760.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 5639.00 | 5612.65 | 5666.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:45:00 | 5633.00 | 5612.65 | 5666.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 5616.00 | 5613.32 | 5661.76 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5807.00 | 5683.66 | 5678.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 5872.00 | 5721.33 | 5696.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 6099.50 | 6124.54 | 6055.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 12:00:00 | 6099.50 | 6124.54 | 6055.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 6084.00 | 6122.74 | 6081.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 6068.00 | 6122.74 | 6081.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 6114.50 | 6121.09 | 6084.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 6114.50 | 6121.09 | 6084.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 6051.00 | 6104.90 | 6083.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:30:00 | 6049.50 | 6104.90 | 6083.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 6045.00 | 6092.92 | 6079.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 6045.00 | 6092.92 | 6079.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 6026.00 | 6071.55 | 6071.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 5959.50 | 6040.25 | 6057.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 5857.50 | 5843.37 | 5904.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:30:00 | 5835.50 | 5843.37 | 5904.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 5913.00 | 5857.30 | 5905.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:45:00 | 5840.00 | 5855.14 | 5900.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 5832.50 | 5857.65 | 5883.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 5914.50 | 5876.82 | 5875.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 11:15:00 | 5914.50 | 5876.82 | 5875.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 12:15:00 | 5975.50 | 5896.56 | 5884.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 5945.00 | 5969.33 | 5939.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 12:15:00 | 5945.00 | 5969.33 | 5939.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 5945.00 | 5969.33 | 5939.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 5943.50 | 5969.33 | 5939.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 5933.50 | 5962.16 | 5938.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:45:00 | 5925.50 | 5962.16 | 5938.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 5867.50 | 5943.23 | 5932.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 5867.50 | 5943.23 | 5932.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 5870.00 | 5928.58 | 5926.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 5870.00 | 5928.58 | 5926.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 5902.50 | 5923.37 | 5924.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 5796.50 | 5879.22 | 5902.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 5811.00 | 5808.63 | 5849.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:45:00 | 5806.50 | 5808.63 | 5849.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 5915.00 | 5822.92 | 5845.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 5915.00 | 5822.92 | 5845.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 5832.00 | 5824.74 | 5844.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 5779.00 | 5820.65 | 5839.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 5940.50 | 5842.79 | 5844.70 | SL hit (close>static) qty=1.00 sl=5924.50 alert=retest2 |

### Cycle 135 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 6133.50 | 5900.93 | 5870.96 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 14:15:00 | 5899.00 | 5923.58 | 5926.13 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 5982.50 | 5936.68 | 5930.98 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 5853.00 | 5916.45 | 5922.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 5812.50 | 5895.66 | 5912.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 12:15:00 | 5791.50 | 5789.21 | 5829.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:00:00 | 5791.50 | 5789.21 | 5829.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 5845.50 | 5803.31 | 5829.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 5845.50 | 5803.31 | 5829.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 5822.00 | 5807.05 | 5828.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 5779.50 | 5807.05 | 5828.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 5862.00 | 5801.68 | 5804.61 | SL hit (close>static) qty=1.00 sl=5845.50 alert=retest2 |

### Cycle 139 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 5803.50 | 5787.83 | 5787.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 5852.00 | 5800.67 | 5793.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 13:15:00 | 6131.50 | 6134.14 | 6073.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:00:00 | 6131.50 | 6134.14 | 6073.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 6110.00 | 6121.21 | 6077.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 6016.50 | 6121.21 | 6077.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 6005.50 | 6098.07 | 6070.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 6043.00 | 6098.07 | 6070.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 6020.50 | 6082.55 | 6066.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:15:00 | 5984.00 | 6082.55 | 6066.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 6006.50 | 6050.22 | 6054.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 5930.00 | 6026.18 | 6042.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 12:15:00 | 5826.50 | 5786.02 | 5855.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 13:00:00 | 5826.50 | 5786.02 | 5855.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 5869.50 | 5802.72 | 5856.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 5892.50 | 5802.72 | 5856.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 5914.00 | 5824.97 | 5861.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 5924.00 | 5824.97 | 5861.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 5907.00 | 5841.38 | 5865.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 5915.00 | 5856.30 | 5870.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 5973.50 | 5879.74 | 5879.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 5973.50 | 5879.74 | 5879.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 11:15:00 | 5965.00 | 5896.79 | 5887.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 13:15:00 | 5990.00 | 5927.95 | 5904.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 5898.00 | 5923.09 | 5906.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 15:15:00 | 5898.00 | 5923.09 | 5906.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 5898.00 | 5923.09 | 5906.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 5811.00 | 5923.09 | 5906.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 5806.00 | 5899.67 | 5896.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 5804.00 | 5899.67 | 5896.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 5755.00 | 5870.74 | 5884.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 5718.50 | 5829.45 | 5862.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 5878.00 | 5829.33 | 5855.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 5878.00 | 5829.33 | 5855.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 5878.00 | 5829.33 | 5855.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 5878.00 | 5829.33 | 5855.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 5870.00 | 5837.46 | 5857.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 5895.00 | 5837.46 | 5857.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 5930.00 | 5858.62 | 5863.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 5930.00 | 5858.62 | 5863.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 5990.00 | 5884.89 | 5875.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 6078.50 | 5967.77 | 5922.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 5920.00 | 6017.42 | 5979.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 5920.00 | 6017.42 | 5979.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 5920.00 | 6017.42 | 5979.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 6072.00 | 6039.73 | 5993.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 6125.00 | 6077.09 | 6034.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:45:00 | 6050.00 | 6085.84 | 6053.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 5948.00 | 6034.21 | 6036.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 5948.00 | 6034.21 | 6036.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 12:15:00 | 5899.00 | 5973.44 | 6003.56 | Break + close below crossover candle low |

### Cycle 145 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 6355.50 | 6031.73 | 6017.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 11:15:00 | 6444.50 | 6356.92 | 6306.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 6550.00 | 6550.37 | 6474.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 6550.00 | 6550.37 | 6474.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 6432.50 | 6515.22 | 6470.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:45:00 | 6416.50 | 6515.22 | 6470.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 6417.00 | 6495.58 | 6465.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:45:00 | 6396.50 | 6495.58 | 6465.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 6353.00 | 6467.06 | 6455.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 6353.00 | 6467.06 | 6455.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 15:15:00 | 6325.00 | 6438.65 | 6443.81 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 09:15:00 | 6536.00 | 6458.12 | 6452.19 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 6434.50 | 6451.02 | 6451.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 6409.00 | 6442.62 | 6448.04 | Break + close below crossover candle low |

### Cycle 149 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 6561.00 | 6466.29 | 6458.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 10:15:00 | 6648.00 | 6502.63 | 6475.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 6430.00 | 6530.65 | 6507.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 6430.00 | 6530.65 | 6507.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 6430.00 | 6530.65 | 6507.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 6430.00 | 6530.65 | 6507.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 6432.00 | 6510.92 | 6500.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:30:00 | 6463.00 | 6495.63 | 6494.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 6410.00 | 6478.51 | 6487.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 6410.00 | 6478.51 | 6487.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 6371.50 | 6457.11 | 6476.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 6351.50 | 6275.05 | 6307.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 6351.50 | 6275.05 | 6307.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 6351.50 | 6275.05 | 6307.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 6351.50 | 6275.05 | 6307.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 6346.50 | 6289.34 | 6311.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 6355.50 | 6289.34 | 6311.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 6455.00 | 6335.38 | 6329.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 6484.50 | 6399.16 | 6376.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 6545.50 | 6581.20 | 6520.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 6545.50 | 6581.20 | 6520.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 6545.50 | 6581.20 | 6520.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 6523.00 | 6581.20 | 6520.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 6491.00 | 6563.16 | 6517.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 6491.00 | 6563.16 | 6517.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 6413.50 | 6533.23 | 6508.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 6413.50 | 6533.23 | 6508.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 6305.00 | 6460.59 | 6477.75 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 6471.00 | 6454.54 | 6452.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 13:15:00 | 6490.50 | 6461.73 | 6456.34 | Break + close above crossover candle high |

### Cycle 154 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 6329.50 | 6442.18 | 6449.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 6242.50 | 6363.49 | 6408.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 6274.50 | 6248.88 | 6304.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 6274.50 | 6248.88 | 6304.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 6298.50 | 6258.80 | 6303.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 6085.50 | 6258.80 | 6303.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 6377.50 | 6164.15 | 6207.25 | SL hit (close>static) qty=1.00 sl=6315.00 alert=retest2 |

### Cycle 155 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 6375.00 | 6232.85 | 6232.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 6409.00 | 6268.08 | 6248.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 6414.50 | 6521.77 | 6445.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 6414.50 | 6521.77 | 6445.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 6414.50 | 6521.77 | 6445.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 6495.00 | 6414.88 | 6413.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 10:45:00 | 6490.50 | 6497.46 | 6473.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 11:15:00 | 6495.50 | 6497.46 | 6473.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 13:15:00 | 6358.00 | 6443.47 | 6452.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 6358.00 | 6443.47 | 6452.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 6311.00 | 6416.97 | 6439.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 6159.00 | 6153.40 | 6231.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 14:30:00 | 6176.50 | 6153.40 | 6231.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 6214.00 | 6168.17 | 6224.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:30:00 | 6213.00 | 6168.17 | 6224.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 6180.50 | 6176.53 | 6218.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 6159.00 | 6176.53 | 6218.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 6285.00 | 6193.08 | 6209.28 | SL hit (close>static) qty=1.00 sl=6233.50 alert=retest2 |

### Cycle 157 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 6266.50 | 6219.75 | 6219.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 6305.00 | 6256.50 | 6238.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 6331.00 | 6336.51 | 6302.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 6331.00 | 6336.51 | 6302.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 6331.00 | 6336.51 | 6302.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:15:00 | 6347.00 | 6336.51 | 6302.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 6339.50 | 6334.80 | 6305.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 6251.50 | 6295.54 | 6296.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 6251.50 | 6295.54 | 6296.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 6186.50 | 6265.17 | 6281.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 6190.00 | 6186.16 | 6232.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 6190.00 | 6186.16 | 6232.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 6190.00 | 6186.16 | 6232.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 6035.00 | 6186.16 | 6232.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 6275.00 | 6118.16 | 6101.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 6275.00 | 6118.16 | 6101.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 6313.00 | 6185.50 | 6136.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 13:15:00 | 6193.00 | 6200.12 | 6152.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:00:00 | 6193.00 | 6200.12 | 6152.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 6159.00 | 6195.45 | 6162.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 6115.50 | 6195.45 | 6162.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 6071.00 | 6170.56 | 6154.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 6071.00 | 6170.56 | 6154.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 6022.00 | 6140.85 | 6142.04 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 14:15:00 | 6178.00 | 6127.88 | 6123.33 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 10:15:00 | 6052.50 | 6119.22 | 6121.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 14:15:00 | 6045.00 | 6089.89 | 6105.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 5912.50 | 5895.65 | 5977.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 5912.50 | 5895.65 | 5977.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 5849.50 | 5882.64 | 5957.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 5839.50 | 5882.64 | 5957.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 5815.50 | 5869.21 | 5944.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 12:45:00 | 5830.00 | 5859.29 | 5926.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 10:00:00 | 5826.50 | 5846.70 | 5898.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 6110.00 | 5883.48 | 5885.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 6110.00 | 5883.48 | 5885.84 | SL hit (close>static) qty=1.00 sl=6069.00 alert=retest2 |

### Cycle 163 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 6025.00 | 5911.78 | 5898.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 6143.00 | 6045.78 | 5983.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 6139.00 | 6152.86 | 6106.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 6139.00 | 6152.86 | 6106.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 6139.00 | 6152.86 | 6106.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 6187.00 | 6163.09 | 6115.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 6267.00 | 6362.25 | 6364.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 6267.00 | 6362.25 | 6364.40 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 10:15:00 | 6396.00 | 6353.01 | 6350.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 13:15:00 | 6491.00 | 6409.51 | 6379.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 6720.50 | 6724.42 | 6653.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:45:00 | 6746.00 | 6724.42 | 6653.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 6799.50 | 6838.51 | 6778.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:45:00 | 6797.50 | 6838.51 | 6778.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 6798.00 | 6830.41 | 6780.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 6847.00 | 6830.41 | 6780.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 3332.95 | 2024-05-22 12:15:00 | 3361.70 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-05-29 15:15:00 | 3305.00 | 2024-06-03 14:15:00 | 3308.60 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-06-18 09:15:00 | 3568.40 | 2024-06-24 13:15:00 | 3678.20 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2024-06-18 11:00:00 | 3557.05 | 2024-06-24 13:15:00 | 3678.20 | STOP_HIT | 1.00 | 3.41% |
| BUY | retest2 | 2024-06-18 12:00:00 | 3552.00 | 2024-06-24 13:15:00 | 3678.20 | STOP_HIT | 1.00 | 3.55% |
| SELL | retest2 | 2024-06-28 10:15:00 | 3617.65 | 2024-07-01 11:15:00 | 3629.95 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-06-28 11:00:00 | 3600.75 | 2024-07-01 11:15:00 | 3629.95 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-07-01 10:00:00 | 3619.80 | 2024-07-01 11:15:00 | 3629.95 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-07-03 09:15:00 | 3648.05 | 2024-07-03 13:15:00 | 3606.55 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-07-03 10:15:00 | 3633.00 | 2024-07-03 13:15:00 | 3606.55 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-07-05 12:15:00 | 3692.00 | 2024-07-08 11:15:00 | 3630.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-07-08 10:15:00 | 3714.00 | 2024-07-08 11:15:00 | 3630.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-07-23 09:30:00 | 3575.00 | 2024-07-26 13:15:00 | 3554.70 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2024-07-23 11:15:00 | 3563.50 | 2024-07-26 13:15:00 | 3554.70 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2024-08-09 09:15:00 | 3577.90 | 2024-08-09 12:15:00 | 3540.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-08-19 14:45:00 | 3282.80 | 2024-08-21 09:15:00 | 3333.75 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-08-20 10:15:00 | 3289.50 | 2024-08-21 09:15:00 | 3333.75 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-08-20 14:30:00 | 3290.05 | 2024-08-21 09:15:00 | 3333.75 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-08-20 15:00:00 | 3287.35 | 2024-08-21 09:15:00 | 3333.75 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-08-21 11:15:00 | 3308.45 | 2024-08-21 11:15:00 | 3314.25 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-08-28 15:15:00 | 3281.00 | 2024-08-30 14:15:00 | 3300.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-08-29 10:15:00 | 3281.90 | 2024-08-30 14:15:00 | 3300.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-08-30 10:00:00 | 3284.40 | 2024-08-30 14:15:00 | 3300.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-09-03 11:30:00 | 3330.05 | 2024-09-06 14:15:00 | 3297.80 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-09-03 15:15:00 | 3331.05 | 2024-09-06 14:15:00 | 3297.80 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-09-06 11:30:00 | 3330.35 | 2024-09-06 14:15:00 | 3297.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-13 12:45:00 | 3277.95 | 2024-09-20 11:15:00 | 3323.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-09-13 13:15:00 | 3277.10 | 2024-09-20 11:15:00 | 3323.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-09-13 14:15:00 | 3274.00 | 2024-09-20 11:15:00 | 3323.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-09-16 09:45:00 | 3265.90 | 2024-09-20 11:15:00 | 3323.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-09-18 11:45:00 | 3245.40 | 2024-09-20 11:15:00 | 3323.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-10-09 13:15:00 | 3366.85 | 2024-10-10 09:15:00 | 3418.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-10-09 14:30:00 | 3369.90 | 2024-10-10 09:15:00 | 3418.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-10-09 15:00:00 | 3361.90 | 2024-10-10 09:15:00 | 3418.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-10-21 10:45:00 | 3332.70 | 2024-10-24 09:15:00 | 3517.00 | STOP_HIT | 1.00 | -5.53% |
| SELL | retest2 | 2024-10-21 12:00:00 | 3328.70 | 2024-10-24 09:15:00 | 3517.00 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2024-10-23 14:30:00 | 3332.85 | 2024-10-24 09:15:00 | 3517.00 | STOP_HIT | 1.00 | -5.53% |
| SELL | retest2 | 2024-10-23 15:00:00 | 3311.50 | 2024-10-24 09:15:00 | 3517.00 | STOP_HIT | 1.00 | -6.21% |
| BUY | retest2 | 2024-11-29 09:45:00 | 3500.05 | 2024-12-06 15:15:00 | 3559.50 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2024-12-02 09:30:00 | 3508.95 | 2024-12-06 15:15:00 | 3559.50 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2024-12-12 09:15:00 | 3616.35 | 2024-12-12 09:15:00 | 3586.35 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-12-16 10:15:00 | 3511.55 | 2024-12-20 13:15:00 | 3335.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:15:00 | 3512.05 | 2024-12-20 13:15:00 | 3336.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 3505.25 | 2024-12-20 13:15:00 | 3329.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 3511.55 | 2024-12-24 09:15:00 | 3409.00 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2024-12-16 11:15:00 | 3512.05 | 2024-12-24 09:15:00 | 3409.00 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2024-12-17 09:15:00 | 3505.25 | 2024-12-24 09:15:00 | 3409.00 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2024-12-27 15:15:00 | 3301.00 | 2025-01-02 12:15:00 | 3277.00 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-01-07 10:30:00 | 3476.45 | 2025-01-09 09:15:00 | 3824.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-07 11:30:00 | 3468.50 | 2025-01-09 09:15:00 | 3815.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-07 13:30:00 | 3478.85 | 2025-01-09 09:15:00 | 3826.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-08 13:45:00 | 3473.90 | 2025-01-09 09:15:00 | 3821.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-10 11:30:00 | 3684.15 | 2025-01-14 11:15:00 | 3602.60 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-01-10 12:15:00 | 3672.20 | 2025-01-14 11:15:00 | 3602.60 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-01-10 13:00:00 | 3684.20 | 2025-01-14 11:15:00 | 3602.60 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-01-20 14:15:00 | 3771.00 | 2025-01-22 09:15:00 | 3670.15 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-02-03 10:45:00 | 4246.00 | 2025-02-11 09:15:00 | 4173.55 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-02-03 11:30:00 | 4243.10 | 2025-02-11 09:15:00 | 4173.55 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-02-05 10:30:00 | 4243.15 | 2025-02-11 09:15:00 | 4173.55 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-02-05 12:00:00 | 4245.00 | 2025-02-11 09:15:00 | 4173.55 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-02-14 10:00:00 | 4011.15 | 2025-02-17 14:15:00 | 4113.60 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-02-14 15:00:00 | 4020.60 | 2025-02-17 14:15:00 | 4113.60 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-02-17 10:00:00 | 4006.55 | 2025-02-17 14:15:00 | 4113.60 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-02-24 09:15:00 | 4005.95 | 2025-02-25 14:15:00 | 3805.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 09:15:00 | 4005.95 | 2025-02-27 12:15:00 | 3605.36 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-10 09:15:00 | 4243.75 | 2025-03-11 09:15:00 | 4020.35 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest2 | 2025-03-10 14:45:00 | 4089.90 | 2025-03-11 09:15:00 | 4020.35 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-03-17 09:15:00 | 4177.00 | 2025-03-19 10:15:00 | 4082.80 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-03-17 13:45:00 | 4135.90 | 2025-03-19 10:15:00 | 4082.80 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-03-17 15:00:00 | 4130.50 | 2025-03-19 10:15:00 | 4082.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-03-27 09:15:00 | 4226.50 | 2025-03-28 10:15:00 | 4195.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-03-27 09:45:00 | 4260.85 | 2025-03-28 14:15:00 | 4206.30 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-03-27 11:00:00 | 4213.95 | 2025-03-28 14:15:00 | 4206.30 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-03-27 15:15:00 | 4228.00 | 2025-03-28 14:15:00 | 4206.30 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-03-28 09:15:00 | 4322.35 | 2025-03-28 14:15:00 | 4206.30 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-04-04 09:15:00 | 4145.00 | 2025-04-07 09:15:00 | 3730.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 09:45:00 | 4143.80 | 2025-04-07 09:15:00 | 3729.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-22 09:30:00 | 4268.90 | 2025-05-05 11:15:00 | 4695.79 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-12 12:15:00 | 4547.00 | 2025-05-13 12:15:00 | 4319.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-12 12:15:00 | 4547.00 | 2025-05-15 09:15:00 | 4344.00 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-05-28 14:00:00 | 4246.10 | 2025-05-28 15:15:00 | 4291.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-16 14:15:00 | 4523.70 | 2025-06-23 15:15:00 | 4641.00 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-06-17 09:15:00 | 4685.80 | 2025-06-23 15:15:00 | 4641.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-06-27 10:00:00 | 4766.70 | 2025-07-11 10:15:00 | 4971.50 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2025-06-27 11:30:00 | 4750.00 | 2025-07-11 10:15:00 | 4971.50 | STOP_HIT | 1.00 | 4.66% |
| BUY | retest2 | 2025-07-29 12:00:00 | 5127.00 | 2025-07-31 14:15:00 | 5051.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-29 13:30:00 | 5127.00 | 2025-07-31 14:15:00 | 5051.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-30 11:00:00 | 5133.70 | 2025-07-31 14:15:00 | 5051.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-19 09:15:00 | 5129.50 | 2025-08-20 14:15:00 | 5016.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-08-19 10:15:00 | 5112.00 | 2025-08-20 14:15:00 | 5016.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-08-19 12:00:00 | 5097.50 | 2025-08-20 14:15:00 | 5016.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest1 | 2025-09-01 09:15:00 | 4633.40 | 2025-09-02 10:15:00 | 4707.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest1 | 2025-09-01 15:00:00 | 4669.90 | 2025-09-02 10:15:00 | 4707.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest1 | 2025-09-02 09:15:00 | 4670.00 | 2025-09-02 10:15:00 | 4707.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-12 12:15:00 | 4817.00 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-09-12 13:30:00 | 4813.90 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-09-15 09:45:00 | 4817.00 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-09-16 09:15:00 | 4823.10 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2025-09-16 11:00:00 | 4847.10 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-16 11:30:00 | 4840.80 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-09-16 14:45:00 | 4868.00 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-09-19 10:00:00 | 4843.10 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-10-28 11:45:00 | 5020.90 | 2025-10-31 09:15:00 | 5722.80 | STOP_HIT | 1.00 | -13.98% |
| BUY | retest2 | 2025-11-11 09:15:00 | 6060.50 | 2025-11-17 12:15:00 | 6017.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-11-11 10:00:00 | 6046.00 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-11 10:30:00 | 6048.50 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-12 09:30:00 | 6046.50 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-13 09:15:00 | 6084.50 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-13 10:15:00 | 6043.00 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-13 10:45:00 | 6044.00 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-13 11:45:00 | 6043.50 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-11-17 09:15:00 | 6075.00 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-11-24 12:30:00 | 5898.00 | 2025-12-02 10:15:00 | 5818.00 | STOP_HIT | 1.00 | 1.36% |
| SELL | retest2 | 2025-12-19 11:45:00 | 5840.00 | 2025-12-23 11:15:00 | 5914.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-22 10:15:00 | 5832.50 | 2025-12-23 11:15:00 | 5914.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-30 10:45:00 | 5779.00 | 2025-12-30 13:15:00 | 5940.50 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2026-01-08 09:15:00 | 5779.50 | 2026-01-09 11:15:00 | 5862.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-01-09 13:00:00 | 5764.00 | 2026-01-12 15:15:00 | 5803.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-12 14:45:00 | 5794.50 | 2026-01-12 15:15:00 | 5803.50 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-01-30 10:30:00 | 6072.00 | 2026-02-01 15:15:00 | 5948.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-02-01 09:15:00 | 6125.00 | 2026-02-01 15:15:00 | 5948.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-02-01 12:45:00 | 6050.00 | 2026-02-01 15:15:00 | 5948.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-02-13 11:30:00 | 6463.00 | 2026-02-13 12:15:00 | 6410.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-03-04 09:15:00 | 6085.50 | 2026-03-05 09:15:00 | 6377.50 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest2 | 2026-03-10 09:15:00 | 6495.00 | 2026-03-11 13:15:00 | 6358.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2026-03-11 10:45:00 | 6490.50 | 2026-03-11 13:15:00 | 6358.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-03-11 11:15:00 | 6495.50 | 2026-03-11 13:15:00 | 6358.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-03-16 12:15:00 | 6159.00 | 2026-03-17 09:15:00 | 6285.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-03-19 10:15:00 | 6347.00 | 2026-03-20 09:15:00 | 6251.50 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-03-19 11:15:00 | 6339.50 | 2026-03-20 09:15:00 | 6251.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-03-23 09:15:00 | 6035.00 | 2026-03-25 09:15:00 | 6275.00 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2026-04-06 10:15:00 | 5839.50 | 2026-04-08 09:15:00 | 6110.00 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2026-04-06 11:00:00 | 5815.50 | 2026-04-08 09:15:00 | 6110.00 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2026-04-06 12:45:00 | 5830.00 | 2026-04-08 09:15:00 | 6110.00 | STOP_HIT | 1.00 | -4.80% |
| SELL | retest2 | 2026-04-07 10:00:00 | 5826.50 | 2026-04-08 09:15:00 | 6110.00 | STOP_HIT | 1.00 | -4.87% |
| BUY | retest2 | 2026-04-13 10:30:00 | 6187.00 | 2026-04-22 14:15:00 | 6267.00 | STOP_HIT | 1.00 | 1.29% |
