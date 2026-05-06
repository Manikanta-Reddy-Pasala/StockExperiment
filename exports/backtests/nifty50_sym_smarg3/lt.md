# LT (LT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 4008.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 6 |
| ALERT3 | 9 |
| PENDING | 18 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 14 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 13
- **Target hits / Stop hits / Partials:** 0 / 15 / 1
- **Avg / median % per leg:** -0.91% / -1.38%
- **Sum % (uncompounded):** -14.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 0 | 13 | 1 | -0.69% | -9.7% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.45% | -2.9% |
| BUY @ 3rd Alert (retest2) | 12 | 3 | 25.0% | 0 | 11 | 1 | -0.56% | -6.7% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.47% | -4.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.47% | -4.9% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.45% | -2.9% |
| retest2 (combined) | 14 | 3 | 21.4% | 0 | 13 | 1 | -0.84% | -11.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 3647.55 | 3533.29 | 3533.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 09:15:00 | 3705.20 | 3543.38 | 3538.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3526.00 | 3576.53 | 3555.94 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3526.00 | 3576.53 | 3555.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3526.00 | 3576.53 | 3555.94 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-29 10:15:00 | 3773.50 | 3611.03 | 3592.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:15:00 | 3772.85 | 3612.64 | 3593.87 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-29 13:15:00 | 3768.40 | 3615.69 | 3595.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 14:15:00 | 3776.85 | 3617.30 | 3596.49 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-20 10:15:00 | 3776.00 | 3639.40 | 3626.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:15:00 | 3773.05 | 3640.73 | 3627.33 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-23 14:15:00 | 3789.65 | 3653.60 | 3634.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 15:15:00 | 3784.00 | 3654.90 | 3635.33 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 3652.50 | 3681.16 | 3653.86 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3510.00 | 3673.67 | 3651.01 | SL hit qty=1.00 sl=3510.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3510.00 | 3673.67 | 3651.01 | SL hit qty=1.00 sl=3510.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3510.00 | 3673.67 | 3651.01 | SL hit qty=1.00 sl=3510.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3510.00 | 3673.67 | 3651.01 | SL hit qty=1.00 sl=3510.00 alert=retest2 |
| CROSSOVER_SKIP | 2024-10-09 14:15:00 | 3483.10 | 3629.81 | 3630.49 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2024-11-25 09:15:00 | 3730.00 | 3555.73 | 3571.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 10:15:00 | 3749.00 | 3557.66 | 3572.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 3706.60 | 3586.11 | 3585.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 3706.60 | 3586.11 | 3585.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 3738.25 | 3593.78 | 3589.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 3722.05 | 3745.13 | 3684.15 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 3677.65 | 3742.77 | 3685.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 3677.65 | 3742.77 | 3685.05 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-01-13 11:15:00 | 3520.00 | 3655.63 | 3655.98 | HTF filter: close above htf_sma |

### Cycle 3 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 3593.70 | 3351.79 | 3351.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 3640.00 | 3374.03 | 3362.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 3553.50 | 3578.56 | 3503.64 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 3553.50 | 3578.56 | 3503.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3553.50 | 3578.56 | 3503.64 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-13 12:15:00 | 3567.30 | 3578.12 | 3504.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 3585.20 | 3578.19 | 3504.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 3491.60 | 3598.20 | 3559.24 | SL hit qty=1.00 sl=3491.60 alert=retest2 |
| CROSSOVER_SKIP | 2025-07-28 14:15:00 | 3422.00 | 3532.71 | 3533.04 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 3637.80 | 3528.27 | 3530.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 3642.70 | 3529.41 | 3531.29 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 3678.80 | 3533.33 | 3533.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 3678.80 | 3533.33 | 3533.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3715.00 | 3578.11 | 3558.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 3589.00 | 3602.03 | 3575.19 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-21 09:15:00 | 3633.50 | 3602.19 | 3576.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:15:00 | 3634.50 | 3602.51 | 3576.48 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-21 15:15:00 | 3619.00 | 3603.35 | 3577.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:15:00 | 3629.10 | 3603.61 | 3577.80 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3573.60 | 3602.61 | 3579.04 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 3579.04 | 3602.61 | 3579.04 | SL hit qty=1.00 sl=3579.04 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 3579.04 | 3602.61 | 3579.04 | SL hit qty=1.00 sl=3579.04 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-29 10:15:00 | 3607.60 | 3597.63 | 3578.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 11:15:00 | 3613.00 | 3597.79 | 3578.33 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-01 11:15:00 | 3598.20 | 3597.86 | 3579.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 12:15:00 | 3603.70 | 3597.92 | 3579.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 3568.60 | 3598.07 | 3579.97 | SL hit qty=1.00 sl=3568.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 3568.60 | 3598.07 | 3579.97 | SL hit qty=1.00 sl=3568.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-03 14:15:00 | 3603.80 | 3596.81 | 3580.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:15:00 | 3601.70 | 3596.85 | 3580.14 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-04 10:15:00 | 3603.00 | 3596.92 | 3580.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-04 11:15:00 | 3589.70 | 3596.84 | 3580.39 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 3568.60 | 3596.12 | 3580.51 | SL hit qty=1.00 sl=3568.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 3602.40 | 3581.99 | 3575.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 3600.00 | 3582.17 | 3575.63 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-11-27 10:15:00 | 4140.00 | 3945.23 | 3852.05 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 4029.60 | 4068.61 | 3996.29 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2026-01-27 11:15:00 | 3770.80 | 3951.04 | 3951.84 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 4045.30 | 3950.38 | 3949.94 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 09:15:00 | 4045.30 | 3950.38 | 3949.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 4107.70 | 3965.56 | 3957.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4081.72 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4081.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4081.72 | EMA400 retest candle locked |

### Cycle 6 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3485.60 | 4023.65 | 4024.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 3474.90 | 4018.19 | 4021.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 3732.90 | 3727.26 | 3839.68 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3971.80 | 3728.75 | 3835.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3971.80 | 3728.75 | 3835.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 3933.90 | 3746.54 | 3840.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 3919.90 | 3748.26 | 3841.24 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 12:15:00 | 3941.30 | 3763.48 | 3844.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-10 13:15:00 | 3952.30 | 3765.36 | 3845.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 3923.00 | 3770.74 | 3846.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 3928.50 | 3772.31 | 3847.39 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4021.20 | 3784.00 | 3851.09 | SL hit qty=1.00 sl=4021.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4021.20 | 3784.00 | 3851.09 | SL hit qty=1.00 sl=4021.20 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 4079.50 | 3903.18 | 3903.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 4099.40 | 3919.20 | 3911.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 3920.00 | 3952.67 | 3930.22 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 3920.00 | 3952.67 | 3930.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3920.00 | 3952.67 | 3930.22 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-05-06 14:15:00 | 4010.80 | 3952.36 | 3930.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 15:15:00 | 4014.90 | 3952.98 | 3931.03 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-29 11:15:00 | 3772.85 | 2024-10-03 13:15:00 | 3510.00 | STOP_HIT | 1.00 | -6.97% |
| BUY | retest2 | 2024-07-29 14:15:00 | 3776.85 | 2024-10-03 13:15:00 | 3510.00 | STOP_HIT | 1.00 | -7.07% |
| BUY | retest2 | 2024-09-20 11:15:00 | 3773.05 | 2024-10-03 13:15:00 | 3510.00 | STOP_HIT | 1.00 | -6.97% |
| BUY | retest2 | 2024-09-23 15:15:00 | 3784.00 | 2024-10-03 13:15:00 | 3510.00 | STOP_HIT | 1.00 | -7.24% |
| BUY | retest2 | 2024-11-25 10:15:00 | 3749.00 | 2024-11-28 09:15:00 | 3706.60 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-13 13:15:00 | 3585.20 | 2025-07-14 12:15:00 | 3491.60 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-07-30 10:15:00 | 3642.70 | 2025-07-30 13:15:00 | 3678.80 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest1 | 2025-08-21 10:15:00 | 3634.50 | 2025-08-26 09:15:00 | 3579.04 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest1 | 2025-08-22 09:15:00 | 3629.10 | 2025-08-26 09:15:00 | 3579.04 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-08-29 11:15:00 | 3613.00 | 2025-09-02 13:15:00 | 3568.60 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-09-01 12:15:00 | 3603.70 | 2025-09-02 13:15:00 | 3568.60 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-03 15:15:00 | 3601.70 | 2025-09-05 10:15:00 | 3568.60 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-09-15 10:15:00 | 3600.00 | 2025-11-27 10:15:00 | 4140.00 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-09-15 10:15:00 | 3600.00 | 2026-02-05 09:15:00 | 4045.30 | STOP_HIT | 0.50 | 12.37% |
| SELL | retest2 | 2026-04-09 10:15:00 | 3919.90 | 2026-04-15 09:15:00 | 4021.20 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-04-13 10:15:00 | 3928.50 | 2026-04-15 09:15:00 | 4021.20 | STOP_HIT | 1.00 | -2.36% |
