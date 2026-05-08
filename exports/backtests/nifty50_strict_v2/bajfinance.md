# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4997 bars)
- **Last close:** 955.35
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 7 |
| ALERT3 | 12 |
| PENDING | 31 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 11 |
| ENTRY2 | 14 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 14 / 17
- **Target hits / Stop hits / Partials:** 7 / 17 / 7
- **Avg / median % per leg:** -2.93% / -1.00%
- **Sum % (uncompounded):** -90.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 5 | 10 | 3 | -6.37% | -114.7% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 2.25% | 22.5% |
| BUY @ 3rd Alert (retest2) | 8 | 3 | 37.5% | 3 | 5 | 0 | -17.15% | -137.2% |
| SELL (all) | 13 | 6 | 46.2% | 2 | 7 | 4 | 1.84% | 23.9% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 0 | 4 | 4 | 1.63% | 13.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.18% | 10.9% |
| retest1 (combined) | 18 | 9 | 50.0% | 2 | 9 | 7 | 1.97% | 35.5% |
| retest2 (combined) | 13 | 5 | 38.5% | 5 | 8 | 0 | -9.71% | -126.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 13:15:00 | 3773.98 | 3654.94 | 3654.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-20 09:15:00 | 3787.32 | 3658.58 | 3656.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 10:15:00 | 3886.45 | 3893.40 | 3808.65 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-10-19 11:15:00 | 3910.15 | 3893.57 | 3809.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 12:15:00 | 3944.45 | 3894.07 | 3809.83 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-23 09:15:00 | 3915.10 | 3894.54 | 3814.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-23 10:15:00 | 3924.43 | 3894.84 | 3815.14 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-23 13:15:00 | 3914.00 | 3895.06 | 3816.44 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-23 14:15:00 | 3894.50 | 3895.06 | 3816.83 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 3756.68 | 3891.54 | 3818.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-26 09:15:00 | 3756.68 | 3891.54 | 3818.49 | SL hit (close<ema400) qty=1.00 sl=3818.49 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-10-26 09:15:00 | 3756.68 | 3891.54 | 3818.49 | SL hit (close<ema400) qty=1.00 sl=3818.49 alert=retest1 |

### Cycle 2 — SELL (started 2023-11-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 14:15:00 | 3678.50 | 3777.97 | 3778.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 09:15:00 | 3610.35 | 3775.35 | 3777.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 09:15:00 | 3673.60 | 3669.34 | 3714.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 3689.95 | 3671.46 | 3713.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 3689.95 | 3671.46 | 3713.72 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-12-05 13:15:00 | 3672.52 | 3671.89 | 3713.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-05 14:15:00 | 3675.55 | 3671.93 | 3712.91 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-12-08 09:15:00 | 3674.50 | 3676.19 | 3711.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 10:15:00 | 3660.48 | 3676.03 | 3711.73 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-12-14 09:15:00 | 3739.20 | 3669.66 | 3703.84 | SL hit (close>static) qty=1.00 sl=3717.43 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-14 09:15:00 | 3739.20 | 3669.66 | 3703.84 | SL hit (close>static) qty=1.00 sl=3717.43 alert=retest2 |
| Cross detected — sustain check pending | 2023-12-22 12:15:00 | 3649.93 | 3700.05 | 3714.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 13:15:00 | 3634.68 | 3699.40 | 3713.66 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-04 09:15:00 | 3824.50 | 3682.75 | 3700.39 | SL hit (close>static) qty=1.00 sl=3717.43 alert=retest2 |

### Cycle 3 — BUY (started 2024-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 10:15:00 | 3877.50 | 3717.58 | 3716.87 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 3541.50 | 3721.33 | 3721.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 12:15:00 | 3521.55 | 3713.59 | 3718.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 09:15:00 | 3320.00 | 3306.03 | 3407.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 3452.52 | 3317.17 | 3402.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 3452.52 | 3317.17 | 3402.88 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 12:15:00 | 3602.77 | 3460.86 | 3460.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 14:15:00 | 3610.12 | 3463.75 | 3462.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 3480.98 | 3481.04 | 3471.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 3480.98 | 3481.04 | 3471.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 3480.98 | 3481.04 | 3471.43 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-04-16 13:15:00 | 3513.18 | 3481.21 | 3471.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-16 14:15:00 | 3484.20 | 3481.24 | 3471.77 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-19 13:15:00 | 3531.35 | 3478.81 | 3471.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 14:15:00 | 3561.95 | 3479.64 | 3471.56 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 3404.10 | 3518.94 | 3493.85 | SL hit (close<static) qty=1.00 sl=3470.07 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-03 09:15:00 | 3625.98 | 3496.27 | 3484.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 3574.30 | 3497.05 | 3485.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-03 14:15:00 | 3467.25 | 3496.84 | 3485.35 | SL hit (close<static) qty=1.00 sl=3470.07 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 15:15:00 | 3302.20 | 3474.09 | 3474.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 3269.00 | 3416.10 | 3436.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 3414.18 | 3408.60 | 3431.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 3484.75 | 3409.45 | 3431.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 3484.75 | 3409.45 | 3431.36 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 3631.12 | 3449.95 | 3449.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 3639.00 | 3456.50 | 3453.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 10:15:00 | 3503.43 | 3513.98 | 3486.95 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-06-25 12:15:00 | 3540.85 | 3514.17 | 3487.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 13:15:00 | 3548.07 | 3514.50 | 3487.62 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-09 14:15:00 | 3536.18 | 3543.27 | 3512.87 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-09 15:15:00 | 3527.50 | 3543.11 | 3512.94 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-07-10 09:15:00 | 3546.25 | 3543.14 | 3513.11 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 10:15:00 | 3538.45 | 3543.10 | 3513.24 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 3531.25 | 3542.56 | 3513.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-11 09:15:00 | 3512.75 | 3542.11 | 3513.62 | SL hit (close<ema400) qty=1.00 sl=3513.62 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-07-11 09:15:00 | 3512.75 | 3542.11 | 3513.62 | SL hit (close<ema400) qty=1.00 sl=3513.62 alert=retest1 |
| Cross detected — sustain check pending | 2024-07-15 13:15:00 | 3534.23 | 3535.54 | 3512.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-15 14:15:00 | 3527.85 | 3535.46 | 3512.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-07-18 13:15:00 | 3544.62 | 3534.17 | 3513.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 14:15:00 | 3553.98 | 3534.36 | 3513.64 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 3502.75 | 3534.20 | 3513.76 | SL hit (close<static) qty=1.00 sl=3510.50 alert=retest2 |

### Cycle 8 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 11:15:00 | 3317.23 | 3495.76 | 3496.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 3304.98 | 3455.57 | 3473.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 13:15:00 | 3379.95 | 3370.66 | 3417.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 11:15:00 | 3412.88 | 3372.23 | 3411.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 3412.88 | 3372.23 | 3411.56 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 3656.00 | 3443.21 | 3442.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 11:15:00 | 3676.38 | 3470.47 | 3456.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 3648.32 | 3698.71 | 3608.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 3648.32 | 3698.71 | 3608.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 3648.32 | 3698.71 | 3608.62 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 3455.55 | 3570.43 | 3570.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 3422.00 | 3567.83 | 3569.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 3396.00 | 3384.01 | 3444.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 14:15:00 | 3433.15 | 3385.22 | 3444.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 3433.15 | 3385.22 | 3444.66 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 3697.88 | 3473.44 | 3472.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 09:15:00 | 3779.55 | 3480.88 | 3476.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 4247.50 | 4298.26 | 4133.41 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-04 10:15:00 | 4380.88 | 4298.79 | 4139.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 11:15:00 | 4368.92 | 4299.49 | 4140.48 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 4377.12 | 4298.15 | 4149.07 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 10:15:00 | 4383.48 | 4299.00 | 4150.24 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 14:15:00 | 4360.17 | 4307.12 | 4162.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 15:15:00 | 4367.90 | 4307.72 | 4163.42 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 09:15:00 | 4587.37 | 4320.59 | 4175.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 09:15:00 | 4586.29 | 4320.59 | 4175.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:15:00 | 4602.65 | 4363.94 | 4212.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-04-24 09:15:00 | 4805.82 | 4416.12 | 4255.41 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2025-04-24 09:15:00 | 4804.69 | 4416.12 | 4255.41 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 4306.75 | 4448.53 | 4293.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 4306.75 | 4448.53 | 4293.99 | SL hit (close<ema200) qty=0.50 sl=4448.53 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-02 13:15:00 | 4431.50 | 4436.51 | 4296.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 4434.75 | 4436.50 | 4296.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 13:15:00 | 4413.00 | 4436.77 | 4305.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 14:15:00 | 4407.25 | 4436.47 | 4306.23 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-07 09:15:00 | 4416.00 | 4435.79 | 4307.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:15:00 | 4445.50 | 4435.89 | 4307.87 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 4470.00 | 4431.88 | 4318.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 4470.00 | 4432.26 | 4318.85 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 4444.00 | 4529.99 | 4436.70 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-05 11:15:00 | 4476.50 | 4529.46 | 4436.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 4466.25 | 4528.83 | 4437.05 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-06-09 10:15:00 | 4878.23 | 4542.32 | 4449.21 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-09 10:15:00 | 4847.98 | 4542.32 | 4449.21 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-09 10:15:00 | 4890.05 | 4542.32 | 4449.21 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 931.50 | 4556.52 | 4473.41 | SL hit (close<static) qty=1.00 sl=4279.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 931.50 | 4556.52 | 4473.41 | SL hit (close<static) qty=1.00 sl=4435.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 14:15:00 | 938.00 | 4380.08 | 4386.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 09:15:00 | 927.00 | 4311.82 | 4351.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 961.00 | 960.24 | 1342.81 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-12 09:15:00 | 945.05 | 996.77 | 1025.93 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 10:15:00 | 944.35 | 996.25 | 1025.52 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-12 13:15:00 | 947.30 | 994.87 | 1024.39 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-12 14:15:00 | 950.80 | 994.43 | 1024.02 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-13 10:15:00 | 947.75 | 993.10 | 1022.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:15:00 | 944.60 | 992.62 | 1022.52 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-14 09:15:00 | 947.10 | 990.33 | 1020.62 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-14 10:15:00 | 948.00 | 989.91 | 1020.26 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-14 14:15:00 | 946.10 | 988.36 | 1018.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 15:15:00 | 945.95 | 987.93 | 1018.51 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-20 09:15:00 | 945.05 | 983.88 | 1014.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:15:00 | 944.00 | 983.49 | 1013.84 | SELL ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 897.13 | 961.45 | 994.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 897.37 | 961.45 | 994.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 898.65 | 961.45 | 994.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 896.80 | 961.45 | 994.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-03 12:15:00 | 961.15 | 957.83 | 990.63 | SL hit (close>ema200) qty=0.50 sl=957.83 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-03 12:15:00 | 961.15 | 957.83 | 990.63 | SL hit (close>ema200) qty=0.50 sl=957.83 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-03 12:15:00 | 961.15 | 957.83 | 990.63 | SL hit (close>ema200) qty=0.50 sl=957.83 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-03 12:15:00 | 961.15 | 957.83 | 990.63 | SL hit (close>ema200) qty=0.50 sl=957.83 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 987.80 | 962.69 | 986.44 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-03-02 11:15:00 | 976.15 | 992.92 | 996.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 12:15:00 | 972.70 | 992.72 | 996.45 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 948.70 | 991.86 | 995.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 10:15:00 | 953.85 | 991.48 | 995.73 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2026-03-12 09:15:00 | 875.43 | 972.62 | 984.74 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-12 15:15:00 | 858.46 | 966.75 | 981.40 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-07 09:15:00 | 975.95 | 919.86 | 927.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 10:15:00 | 972.75 | 920.39 | 927.33 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-19 12:15:00 | 3944.45 | 2023-10-26 09:15:00 | 3756.68 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest1 | 2023-10-23 10:15:00 | 3924.43 | 2023-10-26 09:15:00 | 3756.68 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2023-12-05 14:15:00 | 3675.55 | 2023-12-14 09:15:00 | 3739.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2023-12-08 10:15:00 | 3660.48 | 2023-12-14 09:15:00 | 3739.20 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2023-12-22 13:15:00 | 3634.68 | 2024-01-04 09:15:00 | 3824.50 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest2 | 2024-04-19 14:15:00 | 3561.95 | 2024-04-26 09:15:00 | 3404.10 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest2 | 2024-05-03 10:15:00 | 3574.30 | 2024-05-03 14:15:00 | 3467.25 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest1 | 2024-06-25 13:15:00 | 3548.07 | 2024-07-11 09:15:00 | 3512.75 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest1 | 2024-07-10 10:15:00 | 3538.45 | 2024-07-11 09:15:00 | 3512.75 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-07-18 14:15:00 | 3553.98 | 2024-07-19 09:15:00 | 3502.75 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-04-04 11:15:00 | 4368.92 | 2025-04-15 09:15:00 | 4587.37 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-08 10:15:00 | 4383.48 | 2025-04-15 09:15:00 | 4586.29 | PARTIAL | 0.50 | 4.63% |
| BUY | retest1 | 2025-04-09 15:15:00 | 4367.90 | 2025-04-21 09:15:00 | 4602.65 | PARTIAL | 0.50 | 5.37% |
| BUY | retest1 | 2025-04-04 11:15:00 | 4368.92 | 2025-04-24 09:15:00 | 4805.82 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-04-08 10:15:00 | 4383.48 | 2025-04-24 09:15:00 | 4804.69 | TARGET_HIT | 0.50 | 9.61% |
| BUY | retest1 | 2025-04-09 15:15:00 | 4367.90 | 2025-04-30 09:15:00 | 4306.75 | STOP_HIT | 0.50 | -1.40% |
| BUY | retest2 | 2025-05-02 14:15:00 | 4434.75 | 2025-06-09 10:15:00 | 4878.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-06 14:15:00 | 4407.25 | 2025-06-09 10:15:00 | 4847.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 10:15:00 | 4445.50 | 2025-06-09 10:15:00 | 4890.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 10:15:00 | 4470.00 | 2025-06-16 09:15:00 | 931.50 | STOP_HIT | 1.00 | -79.16% |
| BUY | retest2 | 2025-06-05 12:15:00 | 4466.25 | 2025-06-16 09:15:00 | 931.50 | STOP_HIT | 1.00 | -79.14% |
| SELL | retest1 | 2026-01-12 10:15:00 | 944.35 | 2026-02-02 09:15:00 | 897.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 11:15:00 | 944.60 | 2026-02-02 09:15:00 | 897.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-14 15:15:00 | 945.95 | 2026-02-02 09:15:00 | 898.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-20 10:15:00 | 944.00 | 2026-02-02 09:15:00 | 896.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-12 10:15:00 | 944.35 | 2026-02-03 12:15:00 | 961.15 | STOP_HIT | 0.50 | -1.78% |
| SELL | retest1 | 2026-01-13 11:15:00 | 944.60 | 2026-02-03 12:15:00 | 961.15 | STOP_HIT | 0.50 | -1.75% |
| SELL | retest1 | 2026-01-14 15:15:00 | 945.95 | 2026-02-03 12:15:00 | 961.15 | STOP_HIT | 0.50 | -1.61% |
| SELL | retest1 | 2026-01-20 10:15:00 | 944.00 | 2026-02-03 12:15:00 | 961.15 | STOP_HIT | 0.50 | -1.82% |
| SELL | retest2 | 2026-03-02 12:15:00 | 972.70 | 2026-03-12 09:15:00 | 875.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-04 10:15:00 | 953.85 | 2026-03-12 15:15:00 | 858.46 | TARGET_HIT | 1.00 | 10.00% |
