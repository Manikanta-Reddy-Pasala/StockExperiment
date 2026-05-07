# LT (LT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 4023.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 20 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Target hits / Stop hits / Partials:** 0 / 13 / 0
- **Avg / median % per leg:** -2.37% / -1.97%
- **Sum % (uncompounded):** -30.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 2 | 15.4% | 0 | 13 | 0 | -2.37% | -30.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 0 | 13 | 0 | -2.37% | -30.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 2 | 15.4% | 0 | 13 | 0 | -2.37% | -30.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 3510.40 | 3626.32 | 3626.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 3497.45 | 3625.04 | 3625.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 3601.00 | 3593.46 | 3607.67 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 3601.00 | 3593.46 | 3607.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 3601.00 | 3593.46 | 3607.67 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-22 09:15:00 | 3551.75 | 3592.95 | 3606.93 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:15:00 | 3540.05 | 3591.87 | 3606.25 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-10-31 09:15:00 | 3627.15 | 3522.47 | 3564.82 | SL hit (close>static) qty=1.00 sl=3609.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-04 10:15:00 | 3555.65 | 3531.62 | 3567.46 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 12:15:00 | 3553.50 | 3532.06 | 3567.32 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-11-05 09:15:00 | 3557.50 | 3533.43 | 3567.32 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:15:00 | 3550.60 | 3533.81 | 3567.17 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-11-05 15:15:00 | 3564.75 | 3535.04 | 3567.13 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-06 09:15:00 | 3614.60 | 3535.83 | 3567.37 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 3614.60 | 3535.83 | 3567.37 | SL hit (close>static) qty=1.00 sl=3609.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 3614.60 | 3535.83 | 3567.37 | SL hit (close>static) qty=1.00 sl=3609.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-13 09:15:00 | 3556.60 | 3564.19 | 3577.82 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-13 10:15:00 | 3585.70 | 3564.41 | 3577.86 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-13 11:15:00 | 3566.80 | 3564.43 | 3577.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 13:15:00 | 3557.65 | 3564.37 | 3577.64 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 3575.75 | 3560.17 | 3574.36 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-19 14:15:00 | 3497.55 | 3560.09 | 3573.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 3462.10 | 3558.63 | 3573.10 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 2580m) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 3580.05 | 3553.96 | 3569.90 | SL hit (close>static) qty=1.00 sl=3579.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 3731.00 | 3556.70 | 3571.04 | SL hit (close>static) qty=1.00 sl=3609.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-09 09:15:00 | 3524.00 | 3678.02 | 3666.62 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:15:00 | 3531.55 | 3674.98 | 3665.21 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-13 11:15:00 | 3519.15 | 3655.77 | 3655.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 3519.15 | 3655.77 | 3655.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 3475.45 | 3653.97 | 3655.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 3561.95 | 3553.70 | 3594.26 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 3619.00 | 3555.01 | 3593.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 3619.00 | 3555.01 | 3593.52 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-01 11:15:00 | 3512.00 | 3555.13 | 3593.20 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 3415.95 | 3552.36 | 3591.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-12 12:15:00 | 3566.90 | 3322.56 | 3336.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-12 13:15:00 | 3590.00 | 3325.22 | 3337.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-13 09:15:00 | 3568.20 | 3332.74 | 3341.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-13 10:15:00 | 3575.90 | 3335.16 | 3342.68 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-13 13:15:00 | 3565.00 | 3342.30 | 3346.17 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 15:15:00 | 3570.20 | 3346.75 | 3348.37 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-15 09:15:00 | 3556.90 | 3365.03 | 3357.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 11:15:00 | 3568.60 | 3369.01 | 3359.70 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 3639.00 | 3373.98 | 3362.29 | SL hit (close>static) qty=1.00 sl=3623.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 3639.00 | 3373.98 | 3362.29 | SL hit (close>static) qty=1.00 sl=3623.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 3639.00 | 3373.98 | 3362.29 | SL hit (close>static) qty=1.00 sl=3623.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-20 14:15:00 | 3571.40 | 3418.86 | 3387.22 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-20 15:15:00 | 3575.00 | 3420.42 | 3388.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-22 09:15:00 | 3564.70 | 3433.22 | 3395.95 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 3562.20 | 3435.73 | 3397.58 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 3554.30 | 3436.91 | 3398.37 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-22 13:15:00 | 3547.60 | 3438.01 | 3399.11 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-22 14:15:00 | 3550.70 | 3439.13 | 3399.87 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 3646.70 | 3453.01 | 3408.68 | SL hit (close>static) qty=1.00 sl=3623.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-11 11:15:00 | 3544.90 | 3604.39 | 3560.57 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:15:00 | 3547.00 | 3603.19 | 3560.40 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 3422.00 | 3532.73 | 3532.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 3422.00 | 3532.73 | 3532.98 | EMA200 below EMA400 |

### Cycle 4 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 3770.80 | 3950.99 | 3951.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 3724.70 | 4046.10 | 4035.43 | Break + close below crossover candle low |

### Cycle 5 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3485.60 | 4023.27 | 4024.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 3474.90 | 4017.82 | 4021.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 3732.90 | 3727.11 | 3839.35 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3971.00 | 3728.58 | 3835.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3971.00 | 3728.58 | 3835.14 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 10:15:00 | 3919.80 | 3748.11 | 3840.93 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 3902.80 | 3751.44 | 3841.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 3923.00 | 3770.59 | 3846.68 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-13 11:15:00 | 3941.80 | 3773.85 | 3847.56 | ENTRY2 sustain failed after 120m |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4081.60 | 3783.87 | 3850.80 | SL hit (close>static) qty=1.00 sl=4023.40 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:15:00 | 3920.60 | 3955.92 | 3932.38 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-05-07 12:15:00 | 4033.50 | 3958.80 | 3934.76 | SL hit (close>static) qty=1.00 sl=4023.40 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-22 11:15:00 | 3540.05 | 2024-10-31 09:15:00 | 3627.15 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-11-04 12:15:00 | 3553.50 | 2024-11-06 09:15:00 | 3614.60 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-11-05 11:15:00 | 3550.60 | 2024-11-06 09:15:00 | 3614.60 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-11-13 13:15:00 | 3557.65 | 2024-11-22 13:15:00 | 3580.05 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-11-21 09:15:00 | 3462.10 | 2024-11-25 09:15:00 | 3731.00 | STOP_HIT | 1.00 | -7.77% |
| SELL | retest2 | 2025-01-09 11:15:00 | 3531.55 | 2025-01-13 11:15:00 | 3519.15 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-02-01 13:15:00 | 3415.95 | 2025-05-15 13:15:00 | 3639.00 | STOP_HIT | 1.00 | -6.53% |
| SELL | retest2 | 2025-05-13 15:15:00 | 3570.20 | 2025-05-15 13:15:00 | 3639.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-05-15 11:15:00 | 3568.60 | 2025-05-15 13:15:00 | 3639.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-05-22 11:15:00 | 3562.20 | 2025-05-26 09:15:00 | 3646.70 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-07-11 13:15:00 | 3547.00 | 2025-07-28 14:15:00 | 3422.00 | STOP_HIT | 1.00 | 3.52% |
| SELL | retest2 | 2026-04-09 12:15:00 | 3902.80 | 2026-04-15 09:15:00 | 4081.60 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2026-05-06 11:15:00 | 3920.60 | 2026-05-07 12:15:00 | 4033.50 | STOP_HIT | 1.00 | -2.88% |
