# LT (LT)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4996 bars)
- **Last close:** 3974.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 13 |
| PENDING | 32 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 3 |
| ENTRY2 | 21 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 22
- **Target hits / Stop hits / Partials:** 0 / 23 / 1
- **Avg / median % per leg:** -2.12% / -2.55%
- **Sum % (uncompounded):** -50.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 2 | 16.7% | 0 | 11 | 1 | -1.01% | -12.1% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.60% | -3.2% |
| BUY @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.89% | -8.9% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -3.22% | -38.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.01% | -2.0% |
| SELL @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -3.33% | -36.7% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.74% | -5.2% |
| retest2 (combined) | 21 | 2 | 9.5% | 0 | 20 | 1 | -2.17% | -45.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 11:15:00 | 3292.55 | 3542.51 | 3543.16 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 3647.55 | 3533.29 | 3533.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 09:15:00 | 3705.20 | 3543.38 | 3538.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3526.00 | 3576.53 | 3555.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3526.00 | 3576.53 | 3555.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3526.00 | 3576.53 | 3555.95 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-29 10:15:00 | 3773.50 | 3611.03 | 3592.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:15:00 | 3772.85 | 3612.64 | 3593.87 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-29 13:15:00 | 3768.40 | 3615.69 | 3595.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 14:15:00 | 3776.85 | 3617.30 | 3596.49 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-20 10:15:00 | 3776.00 | 3639.40 | 3626.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:15:00 | 3773.05 | 3640.73 | 3627.33 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-23 14:15:00 | 3789.65 | 3653.60 | 3634.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 15:15:00 | 3784.00 | 3654.90 | 3635.33 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 3652.50 | 3681.16 | 3653.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3507.95 | 3673.67 | 3651.01 | SL hit (close<static) qty=1.00 sl=3510.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3507.95 | 3673.67 | 3651.01 | SL hit (close<static) qty=1.00 sl=3510.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3507.95 | 3673.67 | 3651.01 | SL hit (close<static) qty=1.00 sl=3510.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3507.95 | 3673.67 | 3651.01 | SL hit (close<static) qty=1.00 sl=3510.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 3483.10 | 3629.81 | 3630.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 3478.70 | 3622.31 | 3626.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 3601.00 | 3593.35 | 3609.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 3601.00 | 3593.35 | 3609.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 3601.00 | 3593.35 | 3609.33 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-10-22 09:15:00 | 3551.75 | 3592.86 | 3608.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 3537.00 | 3592.30 | 3608.17 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-31 09:15:00 | 3627.15 | 3522.22 | 3565.97 | SL hit (close>static) qty=1.00 sl=3609.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-04 10:15:00 | 3554.05 | 3529.47 | 3567.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 3554.85 | 3529.72 | 3567.89 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-05 09:15:00 | 3557.50 | 3531.39 | 3567.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 10:15:00 | 3554.55 | 3531.62 | 3567.73 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-05 15:15:00 | 3564.75 | 3533.12 | 3567.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-06 09:15:00 | 3615.00 | 3533.93 | 3567.83 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 3615.00 | 3533.93 | 3567.83 | SL hit (close>static) qty=1.00 sl=3609.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 3615.00 | 3533.93 | 3567.83 | SL hit (close>static) qty=1.00 sl=3609.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-13 09:15:00 | 3556.60 | 3562.90 | 3578.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-13 10:15:00 | 3585.70 | 3563.13 | 3578.27 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-13 11:15:00 | 3566.75 | 3563.16 | 3578.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 12:15:00 | 3564.50 | 3563.18 | 3578.14 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 3575.75 | 3559.10 | 3574.71 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-19 14:15:00 | 3497.60 | 3559.05 | 3574.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:15:00 | 3511.20 | 3558.58 | 3574.00 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 3580.00 | 3552.99 | 3570.19 | SL hit (close>static) qty=1.00 sl=3579.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 3730.00 | 3555.73 | 3571.31 | SL hit (close>static) qty=1.00 sl=3609.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 3706.60 | 3586.11 | 3585.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 3738.25 | 3593.78 | 3589.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 3722.05 | 3745.13 | 3684.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 3677.65 | 3742.77 | 3685.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 3677.65 | 3742.77 | 3685.05 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 3520.00 | 3655.63 | 3655.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 3475.05 | 3653.84 | 3655.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 3561.95 | 3553.29 | 3594.10 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 3313.75 | 3551.58 | 3591.84 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-03 10:15:00 | 3312.45 | 3549.20 | 3590.45 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 3379.00 | 3279.06 | 3362.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 3379.00 | 3279.06 | 3362.71 | SL hit (close>ema400) qty=1.00 sl=3362.71 alert=retest1 |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 3277.40 | 3361.26 | 3387.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 3279.40 | 3360.45 | 3387.45 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-28 14:15:00 | 3325.00 | 3284.65 | 3327.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 15:15:00 | 3327.30 | 3285.07 | 3327.33 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-29 12:15:00 | 3333.00 | 3287.78 | 3327.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:15:00 | 3320.70 | 3288.10 | 3327.83 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-30 12:15:00 | 3331.70 | 3290.95 | 3328.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-30 13:15:00 | 3338.60 | 3291.42 | 3328.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-30 15:15:00 | 3330.00 | 3292.36 | 3328.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-02 09:15:00 | 3368.40 | 3293.12 | 3328.46 | ENTRY2 sustain failed after 2520m |
| Cross detected — sustain check pending | 2025-05-02 15:15:00 | 3325.00 | 3295.90 | 3328.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 3312.10 | 3296.06 | 3328.74 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 3960m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 3332.90 | 3296.43 | 3328.76 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 3306.50 | 3297.96 | 3328.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-06 10:15:00 | 3326.70 | 3298.24 | 3328.57 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-07 09:15:00 | 3300.00 | 3300.05 | 3328.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-07 10:15:00 | 3319.70 | 3300.25 | 3328.56 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 3443.00 | 3305.02 | 3329.21 | SL hit (close>static) qty=1.00 sl=3379.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 3443.00 | 3305.02 | 3329.21 | SL hit (close>static) qty=1.00 sl=3379.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 3443.00 | 3305.02 | 3329.21 | SL hit (close>static) qty=1.00 sl=3379.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 3443.00 | 3305.02 | 3329.21 | SL hit (close>static) qty=1.00 sl=3379.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 3593.70 | 3351.79 | 3351.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 3640.00 | 3374.03 | 3362.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 3553.50 | 3578.56 | 3503.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 3553.50 | 3578.56 | 3503.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3553.50 | 3578.56 | 3503.64 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-13 12:15:00 | 3567.30 | 3578.12 | 3504.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 3585.20 | 3578.19 | 3504.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 3490.70 | 3587.80 | 3555.99 | SL hit (close<static) qty=1.00 sl=3491.60 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 3422.00 | 3532.71 | 3533.04 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 3678.80 | 3533.33 | 3533.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3715.00 | 3578.11 | 3558.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 3589.00 | 3602.03 | 3575.19 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-08-21 09:15:00 | 3633.50 | 3602.19 | 3576.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:15:00 | 3634.50 | 3602.51 | 3576.48 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-21 15:15:00 | 3619.00 | 3603.35 | 3577.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:15:00 | 3629.10 | 3603.61 | 3577.80 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3573.60 | 3602.61 | 3579.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 3573.60 | 3602.61 | 3579.04 | SL hit (close<ema400) qty=1.00 sl=3579.04 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 3573.60 | 3602.61 | 3579.04 | SL hit (close<ema400) qty=1.00 sl=3579.04 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-29 10:15:00 | 3607.60 | 3597.63 | 3578.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 11:15:00 | 3613.00 | 3597.79 | 3578.33 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-01 11:15:00 | 3598.20 | 3597.86 | 3579.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 12:15:00 | 3603.70 | 3597.92 | 3579.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 3566.00 | 3598.07 | 3579.97 | SL hit (close<static) qty=1.00 sl=3568.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 3566.00 | 3598.07 | 3579.97 | SL hit (close<static) qty=1.00 sl=3568.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-03 14:15:00 | 3603.80 | 3596.81 | 3580.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:15:00 | 3601.70 | 3596.85 | 3580.14 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-04 10:15:00 | 3603.00 | 3596.92 | 3580.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-04 11:15:00 | 3589.70 | 3596.84 | 3580.39 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 3565.00 | 3595.81 | 3580.43 | SL hit (close<static) qty=1.00 sl=3568.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 3602.40 | 3581.99 | 3575.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 3600.00 | 3582.17 | 3575.63 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:15:00 | 4140.00 | 3945.23 | 3852.05 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 3979.30 | 3980.25 | 3894.92 | SL hit (close<ema200) qty=0.50 sl=3980.25 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 4029.60 | 4068.61 | 3996.29 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 3770.80 | 3951.04 | 3951.84 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 09:15:00 | 4045.30 | 3950.38 | 3949.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 4107.70 | 3965.56 | 3957.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4081.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4081.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4081.72 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3485.60 | 4023.65 | 4024.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 3474.90 | 4018.19 | 4021.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 3732.90 | 3727.26 | 3839.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3971.80 | 3728.75 | 3835.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3971.80 | 3728.75 | 3835.47 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 3933.90 | 3746.54 | 3840.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 3919.90 | 3748.26 | 3841.24 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 12:15:00 | 3941.30 | 3763.48 | 3844.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-10 13:15:00 | 3952.30 | 3765.36 | 3845.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 3923.00 | 3770.74 | 3846.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 3928.50 | 3772.31 | 3847.39 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4080.60 | 3784.00 | 3851.09 | SL hit (close>static) qty=1.00 sl=4021.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4080.60 | 3784.00 | 3851.09 | SL hit (close>static) qty=1.00 sl=4021.20 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 4079.50 | 3903.18 | 3903.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 4099.40 | 3919.20 | 3911.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 3920.00 | 3952.67 | 3930.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 3920.00 | 3952.67 | 3930.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3920.00 | 3952.67 | 3930.22 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-06 14:15:00 | 4010.80 | 3952.36 | 3930.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 15:15:00 | 4014.90 | 3952.98 | 3931.03 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-29 11:15:00 | 3772.85 | 2024-10-03 13:15:00 | 3507.95 | STOP_HIT | 1.00 | -7.02% |
| BUY | retest2 | 2024-07-29 14:15:00 | 3776.85 | 2024-10-03 13:15:00 | 3507.95 | STOP_HIT | 1.00 | -7.12% |
| BUY | retest2 | 2024-09-20 11:15:00 | 3773.05 | 2024-10-03 13:15:00 | 3507.95 | STOP_HIT | 1.00 | -7.03% |
| BUY | retest2 | 2024-09-23 15:15:00 | 3784.00 | 2024-10-03 13:15:00 | 3507.95 | STOP_HIT | 1.00 | -7.30% |
| SELL | retest2 | 2024-10-22 10:15:00 | 3537.00 | 2024-10-31 09:15:00 | 3627.15 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-11-04 11:15:00 | 3554.85 | 2024-11-06 09:15:00 | 3615.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-11-05 10:15:00 | 3554.55 | 2024-11-06 09:15:00 | 3615.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-11-13 12:15:00 | 3564.50 | 2024-11-22 13:15:00 | 3580.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-11-19 15:15:00 | 3511.20 | 2024-11-25 09:15:00 | 3730.00 | STOP_HIT | 1.00 | -6.23% |
| SELL | retest1 | 2025-02-03 10:15:00 | 3312.45 | 2025-03-21 09:15:00 | 3379.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-04-04 10:15:00 | 3279.40 | 2025-05-09 09:15:00 | 3443.00 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-04-28 15:15:00 | 3327.30 | 2025-05-09 09:15:00 | 3443.00 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-04-29 13:15:00 | 3320.70 | 2025-05-09 09:15:00 | 3443.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-05-05 09:15:00 | 3312.10 | 2025-05-09 09:15:00 | 3443.00 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2025-06-13 13:15:00 | 3585.20 | 2025-07-16 09:15:00 | 3490.70 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest1 | 2025-08-21 10:15:00 | 3634.50 | 2025-08-26 09:15:00 | 3573.60 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2025-08-22 09:15:00 | 3629.10 | 2025-08-26 09:15:00 | 3573.60 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-08-29 11:15:00 | 3613.00 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-01 12:15:00 | 3603.70 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-09-03 15:15:00 | 3601.70 | 2025-09-05 11:15:00 | 3565.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-09-15 10:15:00 | 3600.00 | 2025-11-27 10:15:00 | 4140.00 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-09-15 10:15:00 | 3600.00 | 2025-12-09 09:15:00 | 3979.30 | STOP_HIT | 0.50 | 10.54% |
| SELL | retest2 | 2026-04-09 10:15:00 | 3919.90 | 2026-04-15 09:15:00 | 4080.60 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2026-04-13 10:15:00 | 3928.50 | 2026-04-15 09:15:00 | 4080.60 | STOP_HIT | 1.00 | -3.87% |
