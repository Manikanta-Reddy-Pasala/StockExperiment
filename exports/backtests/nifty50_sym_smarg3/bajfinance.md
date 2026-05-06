# BAJFINANCE (BAJFINANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4997 bars)
- **Last close:** 980.75
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 10 |
| PENDING | 31 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 11 |
| ENTRY2 | 13 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 22
- **Target hits / Stop hits / Partials:** 0 / 24 / 2
- **Avg / median % per leg:** -0.76% / -1.69%
- **Sum % (uncompounded):** -19.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 0 | 16 | 0 | -1.88% | -30.1% |
| BUY @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.92% | -13.4% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.85% | -16.6% |
| SELL (all) | 10 | 3 | 30.0% | 0 | 8 | 2 | 1.03% | 10.3% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.42% | -17.7% |
| SELL @ 3rd Alert (retest2) | 6 | 3 | 50.0% | 0 | 4 | 2 | 4.67% | 28.0% |
| retest1 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.83% | -31.1% |
| retest2 (combined) | 15 | 4 | 26.7% | 0 | 13 | 2 | 0.76% | 11.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 14:15:00 | 3679.12 | 3612.95 | 3612.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 3694.00 | 3614.45 | 3613.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 10:15:00 | 3886.45 | 3893.69 | 3799.72 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2023-10-19 11:15:00 | 3910.15 | 3893.86 | 3800.27 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 12:15:00 | 3944.45 | 3894.36 | 3800.99 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-23 09:15:00 | 3915.10 | 3894.80 | 3806.22 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-23 10:15:00 | 3924.43 | 3895.09 | 3806.81 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-23 13:15:00 | 3914.00 | 3895.31 | 3808.24 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-23 14:15:00 | 3894.50 | 3895.30 | 3808.67 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 3756.68 | 3891.76 | 3810.69 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2023-10-26 09:15:00 | 3810.69 | 3891.76 | 3810.69 | SL hit qty=1.00 sl=3810.69 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-10-26 09:15:00 | 3810.69 | 3891.76 | 3810.69 | SL hit qty=1.00 sl=3810.69 alert=retest1 |
| CROSSOVER_SKIP | 2023-11-17 13:15:00 | 3607.50 | 3769.06 | 3769.35 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2023-12-20 10:15:00 | 3829.23 | 3696.31 | 3711.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-12-20 11:15:00 | 3822.00 | 3697.56 | 3711.77 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-04 11:15:00 | 3837.38 | 3685.64 | 3700.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 12:15:00 | 3832.50 | 3687.10 | 3700.83 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-08 15:15:00 | 3868.00 | 3714.34 | 3713.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 15:15:00 | 3868.00 | 3714.34 | 3713.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 09:15:00 | 3879.62 | 3715.98 | 3714.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 13:15:00 | 3745.50 | 3748.73 | 3732.79 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 14:15:00 | 3732.50 | 3748.57 | 3732.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 3732.50 | 3748.57 | 3732.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-01-16 09:15:00 | 3780.15 | 3748.79 | 3733.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 10:15:00 | 3755.60 | 3748.85 | 3733.17 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-16 12:15:00 | 3728.00 | 3748.82 | 3733.30 | SL hit qty=1.00 sl=3728.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 09:15:00 | 3526.07 | 3719.39 | 3720.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 09:15:00 | 3439.05 | 3686.41 | 3702.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 09:15:00 | 3320.00 | 3306.03 | 3406.91 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 3452.52 | 3317.17 | 3402.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 3452.52 | 3317.17 | 3402.65 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-04-10 12:15:00 | 3602.77 | 3460.86 | 3460.67 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2024-04-26 11:15:00 | 3373.73 | 3516.12 | 3492.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 12:15:00 | 3351.27 | 3514.48 | 3491.86 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-30 13:15:00 | 3467.50 | 3499.46 | 3485.65 | SL hit qty=1.00 sl=3467.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-09 10:15:00 | 3362.80 | 3481.93 | 3478.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 11:15:00 | 3351.12 | 3480.63 | 3478.10 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-09 15:15:00 | 3302.20 | 3474.09 | 3474.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 15:15:00 | 3302.20 | 3474.09 | 3474.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 3269.00 | 3416.10 | 3435.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 3414.18 | 3408.60 | 3431.11 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 3484.75 | 3409.45 | 3431.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 3484.75 | 3409.45 | 3431.31 | EMA400 retest candle locked |

### Cycle 5 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 3631.12 | 3449.95 | 3449.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 3639.00 | 3456.50 | 3453.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 10:15:00 | 3503.43 | 3513.98 | 3486.92 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-06-25 12:15:00 | 3540.85 | 3514.17 | 3487.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 13:15:00 | 3548.07 | 3514.50 | 3487.59 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-09 14:15:00 | 3536.18 | 3543.27 | 3512.85 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-09 15:15:00 | 3527.50 | 3543.11 | 3512.92 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-07-10 09:15:00 | 3546.25 | 3543.14 | 3513.09 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 10:15:00 | 3538.45 | 3543.10 | 3513.22 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 3531.25 | 3542.56 | 3513.54 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-07-10 14:15:00 | 3513.54 | 3542.56 | 3513.54 | SL hit qty=1.00 sl=3513.54 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-07-10 14:15:00 | 3513.54 | 3542.56 | 3513.54 | SL hit qty=1.00 sl=3513.54 alert=retest1 |
| Cross detected — sustain check pending | 2024-07-15 13:15:00 | 3534.23 | 3535.54 | 3512.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-15 14:15:00 | 3527.85 | 3535.46 | 3512.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-07-18 13:15:00 | 3544.62 | 3534.17 | 3513.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 14:15:00 | 3553.98 | 3534.36 | 3513.62 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 3510.50 | 3534.20 | 3513.75 | SL hit qty=1.00 sl=3510.50 alert=retest2 |

### Cycle 6 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 11:15:00 | 3317.23 | 3495.76 | 3496.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 3304.98 | 3455.57 | 3473.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 13:15:00 | 3379.95 | 3370.66 | 3417.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 11:15:00 | 3412.88 | 3372.23 | 3411.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 3412.88 | 3372.23 | 3411.56 | EMA400 retest candle locked |

### Cycle 7 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 3656.00 | 3443.21 | 3442.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 11:15:00 | 3676.38 | 3470.47 | 3456.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 3648.32 | 3698.71 | 3608.62 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 3648.32 | 3698.71 | 3608.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 3648.32 | 3698.71 | 3608.62 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-10-28 14:15:00 | 3455.55 | 3570.43 | 3570.95 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-01-02 13:15:00 | 3714.38 | 3456.45 | 3463.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-02 14:15:00 | 3694.55 | 3458.82 | 3465.14 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-03 11:15:00 | 3746.48 | 3468.71 | 3470.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 12:15:00 | 3717.60 | 3471.18 | 3471.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-03 13:15:00 | 3697.88 | 3473.44 | 3472.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 3697.88 | 3473.44 | 3472.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 09:15:00 | 3779.55 | 3480.88 | 3476.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 4247.50 | 4298.26 | 4133.41 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-04-04 10:15:00 | 4380.88 | 4298.79 | 4139.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 11:15:00 | 4368.92 | 4299.49 | 4140.48 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 4377.12 | 4298.15 | 4149.07 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 10:15:00 | 4383.48 | 4299.00 | 4150.24 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 14:15:00 | 4360.17 | 4307.12 | 4162.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 15:15:00 | 4367.90 | 4307.72 | 4163.42 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 4306.75 | 4448.53 | 4293.99 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 4293.99 | 4448.53 | 4293.99 | SL hit qty=1.00 sl=4293.99 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 4293.99 | 4448.53 | 4293.99 | SL hit qty=1.00 sl=4293.99 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 4293.99 | 4448.53 | 4293.99 | SL hit qty=1.00 sl=4293.99 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-02 13:15:00 | 4431.50 | 4436.51 | 4296.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 4434.75 | 4436.50 | 4296.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 13:15:00 | 4413.00 | 4436.77 | 4305.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 14:15:00 | 4407.25 | 4436.47 | 4306.23 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-07 09:15:00 | 4416.00 | 4435.79 | 4307.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:15:00 | 4445.50 | 4435.89 | 4307.87 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 4470.00 | 4431.88 | 4318.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 4470.00 | 4432.26 | 4318.85 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 4444.00 | 4529.99 | 4436.70 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-05 11:15:00 | 4476.50 | 4529.46 | 4436.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 4466.25 | 4528.83 | 4437.05 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 4279.75 | 4556.52 | 4473.41 | SL hit qty=1.00 sl=4279.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 4279.75 | 4556.52 | 4473.41 | SL hit qty=1.00 sl=4279.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 4279.75 | 4556.52 | 4473.41 | SL hit qty=1.00 sl=4279.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 4279.75 | 4556.52 | 4473.41 | SL hit qty=1.00 sl=4279.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 4435.00 | 4556.52 | 4473.41 | SL hit qty=1.00 sl=4435.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 14:15:00 | 938.00 | 4380.08 | 4386.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 09:15:00 | 927.00 | 4311.82 | 4351.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 961.00 | 960.24 | 1342.81 | EMA200 retest candle locked |
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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 987.80 | 962.69 | 986.44 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 986.44 | 962.69 | 986.44 | SL hit qty=1.00 sl=986.44 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 986.44 | 962.69 | 986.44 | SL hit qty=1.00 sl=986.44 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 986.44 | 962.69 | 986.44 | SL hit qty=1.00 sl=986.44 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 986.44 | 962.69 | 986.44 | SL hit qty=1.00 sl=986.44 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-02 11:15:00 | 976.15 | 992.92 | 996.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 12:15:00 | 972.70 | 992.72 | 996.45 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 948.70 | 991.86 | 995.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 10:15:00 | 953.85 | 991.48 | 995.73 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-20 09:15:00 | 826.80 | 934.33 | 961.38 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-23 09:15:00 | 810.77 | 926.77 | 956.60 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 972.70 | 907.78 | 922.64 | SL hit qty=0.50 sl=972.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 953.85 | 907.78 | 922.64 | SL hit qty=0.50 sl=953.85 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-19 12:15:00 | 3944.45 | 2023-10-26 09:15:00 | 3810.69 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest1 | 2023-10-23 10:15:00 | 3924.43 | 2023-10-26 09:15:00 | 3810.69 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-01-04 12:15:00 | 3832.50 | 2024-01-08 15:15:00 | 3868.00 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2024-01-16 10:15:00 | 3755.60 | 2024-01-16 12:15:00 | 3728.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-04-26 12:15:00 | 3351.27 | 2024-04-30 13:15:00 | 3467.50 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2024-05-09 11:15:00 | 3351.12 | 2024-05-09 15:15:00 | 3302.20 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest1 | 2024-06-25 13:15:00 | 3548.07 | 2024-07-10 14:15:00 | 3513.54 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2024-07-10 10:15:00 | 3538.45 | 2024-07-10 14:15:00 | 3513.54 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-07-18 14:15:00 | 3553.98 | 2024-07-19 09:15:00 | 3510.50 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-01-03 12:15:00 | 3717.60 | 2025-01-03 13:15:00 | 3697.88 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-04-04 11:15:00 | 4368.92 | 2025-04-30 09:15:00 | 4293.99 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2025-04-08 10:15:00 | 4383.48 | 2025-04-30 09:15:00 | 4293.99 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest1 | 2025-04-09 15:15:00 | 4367.90 | 2025-04-30 09:15:00 | 4293.99 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-02 14:15:00 | 4434.75 | 2025-06-16 09:15:00 | 4279.75 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2025-05-06 14:15:00 | 4407.25 | 2025-06-16 09:15:00 | 4279.75 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-05-07 10:15:00 | 4445.50 | 2025-06-16 09:15:00 | 4279.75 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-05-12 10:15:00 | 4470.00 | 2025-06-16 09:15:00 | 4279.75 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2025-06-05 12:15:00 | 4466.25 | 2025-06-16 09:15:00 | 4435.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest1 | 2026-01-12 10:15:00 | 944.35 | 2026-02-12 10:15:00 | 986.44 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest1 | 2026-01-13 11:15:00 | 944.60 | 2026-02-12 10:15:00 | 986.44 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest1 | 2026-01-14 15:15:00 | 945.95 | 2026-02-12 10:15:00 | 986.44 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest1 | 2026-01-20 10:15:00 | 944.00 | 2026-02-12 10:15:00 | 986.44 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2026-03-02 12:15:00 | 972.70 | 2026-03-20 09:15:00 | 826.80 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-03-04 10:15:00 | 953.85 | 2026-03-23 09:15:00 | 810.77 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-03-02 12:15:00 | 972.70 | 2026-04-30 09:15:00 | 972.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest2 | 2026-03-04 10:15:00 | 953.85 | 2026-04-30 09:15:00 | 953.85 | STOP_HIT | 0.50 | 0.00% |
