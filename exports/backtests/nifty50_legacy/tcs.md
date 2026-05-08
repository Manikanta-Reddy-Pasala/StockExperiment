# TCS (TCS)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4996 bars)
- **Last close:** 2394.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 11 |
| PENDING | 41 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 14 |
| ENTRY2 | 19 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 11 / 21
- **Target hits / Stop hits / Partials:** 0 / 30 / 2
- **Avg / median % per leg:** 1.20% / -1.26%
- **Sum % (uncompounded):** 38.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 7 | 38.9% | 0 | 18 | 0 | 0.61% | 11.0% |
| BUY @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 0 | 11 | 0 | 2.17% | 23.9% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.85% | -13.0% |
| SELL (all) | 14 | 4 | 28.6% | 0 | 12 | 2 | 1.96% | 27.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 4 | 28.6% | 0 | 12 | 2 | 1.96% | 27.4% |
| retest1 (combined) | 11 | 7 | 63.6% | 0 | 11 | 0 | 2.17% | 23.9% |
| retest2 (combined) | 21 | 4 | 19.0% | 0 | 19 | 2 | 0.69% | 14.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 11:15:00 | 3360.20 | 3458.22 | 3458.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 3351.80 | 3455.27 | 3456.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 3442.00 | 3420.24 | 3436.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 3442.00 | 3420.24 | 3436.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 3442.00 | 3420.24 | 3436.88 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 13:15:00 | 3515.00 | 3451.16 | 3450.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 12:15:00 | 3518.00 | 3467.62 | 3459.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 09:15:00 | 3665.50 | 3688.40 | 3607.50 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-05 10:15:00 | 3724.00 | 3687.68 | 3610.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 11:15:00 | 3718.20 | 3687.99 | 3610.83 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-09 09:15:00 | 3741.80 | 3690.27 | 3616.51 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:15:00 | 3726.30 | 3690.63 | 3617.06 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-10 10:15:00 | 3718.00 | 3691.58 | 3620.07 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-10 11:15:00 | 3708.40 | 3691.74 | 3620.51 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-11 10:15:00 | 3749.70 | 3692.94 | 3623.22 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 11:15:00 | 3740.10 | 3693.40 | 3623.80 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 3977.90 | 4074.07 | 3972.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-19 15:15:00 | 3971.00 | 4073.04 | 3972.98 | SL hit (close<ema400) qty=1.00 sl=3972.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-19 15:15:00 | 3971.00 | 4073.04 | 3972.98 | SL hit (close<ema400) qty=1.00 sl=3972.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-19 15:15:00 | 3971.00 | 4073.04 | 3972.98 | SL hit (close<ema400) qty=1.00 sl=3972.98 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-20 09:15:00 | 3999.90 | 4072.32 | 3973.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 10:15:00 | 4005.00 | 4071.65 | 3973.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-21 11:15:00 | 3949.00 | 4064.68 | 3973.58 | SL hit (close<static) qty=1.00 sl=3968.60 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-04 13:15:00 | 4005.00 | 3998.35 | 3958.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 14:15:00 | 4004.15 | 3998.41 | 3958.49 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-08 11:15:00 | 4016.00 | 3997.33 | 3960.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 12:15:00 | 4014.90 | 3997.50 | 3960.35 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-08 15:15:00 | 3964.90 | 3997.04 | 3960.67 | SL hit (close<static) qty=1.00 sl=3968.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-08 15:15:00 | 3964.90 | 3997.04 | 3960.67 | SL hit (close<static) qty=1.00 sl=3968.60 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-09 10:15:00 | 3996.55 | 3996.94 | 3960.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-09 11:15:00 | 3965.90 | 3996.63 | 3961.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-12 10:15:00 | 3995.90 | 3992.09 | 3960.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 11:15:00 | 4000.25 | 3992.17 | 3961.08 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 3960.10 | 3992.07 | 3962.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-15 12:15:00 | 3960.10 | 3992.07 | 3962.25 | SL hit (close<static) qty=1.00 sl=3968.60 alert=retest2 |

### Cycle 3 — SELL (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 10:15:00 | 3856.40 | 3939.58 | 3939.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 11:15:00 | 3848.00 | 3938.66 | 3939.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 09:15:00 | 3923.00 | 3913.82 | 3925.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 09:15:00 | 3923.00 | 3913.82 | 3925.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 3923.00 | 3913.82 | 3925.79 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-16 12:15:00 | 3850.95 | 3917.65 | 3925.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:15:00 | 3853.95 | 3917.02 | 3925.02 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-17 13:15:00 | 3848.70 | 3914.43 | 3923.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 14:15:00 | 3832.60 | 3913.61 | 3922.98 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-24 09:15:00 | 3854.65 | 3899.02 | 3914.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-24 10:15:00 | 3860.15 | 3898.64 | 3913.90 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-24 11:15:00 | 3857.00 | 3898.22 | 3913.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 12:15:00 | 3858.65 | 3897.83 | 3913.34 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-27 14:15:00 | 3845.40 | 3895.43 | 3911.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 15:15:00 | 3847.05 | 3894.95 | 3911.11 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 3896.70 | 3836.13 | 3873.78 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-06-12 14:15:00 | 3830.95 | 3843.47 | 3873.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 15:15:00 | 3837.00 | 3843.41 | 3872.98 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-14 10:15:00 | 3840.00 | 3845.39 | 3872.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-14 11:15:00 | 3843.90 | 3845.38 | 3872.56 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-14 12:15:00 | 3836.40 | 3845.29 | 3872.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 13:15:00 | 3830.75 | 3845.15 | 3872.17 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-21 10:15:00 | 3839.75 | 3837.18 | 3864.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:15:00 | 3837.15 | 3837.18 | 3864.65 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-24 13:15:00 | 3825.65 | 3836.41 | 3863.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:15:00 | 3815.00 | 3836.20 | 3862.80 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 3857.10 | 3835.31 | 3861.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 3910.00 | 3837.84 | 3861.32 | SL hit (close>static) qty=1.00 sl=3903.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 3910.00 | 3837.84 | 3861.32 | SL hit (close>static) qty=1.00 sl=3903.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 3910.00 | 3837.84 | 3861.32 | SL hit (close>static) qty=1.00 sl=3903.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 3910.00 | 3837.84 | 3861.32 | SL hit (close>static) qty=1.00 sl=3903.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 3933.20 | 3840.25 | 3862.18 | SL hit (close>static) qty=1.00 sl=3932.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 3933.20 | 3840.25 | 3862.18 | SL hit (close>static) qty=1.00 sl=3932.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 3933.20 | 3840.25 | 3862.18 | SL hit (close>static) qty=1.00 sl=3932.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 3933.20 | 3840.25 | 3862.18 | SL hit (close>static) qty=1.00 sl=3932.80 alert=retest2 |

### Cycle 4 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 4020.00 | 3881.21 | 3880.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 4038.10 | 3907.86 | 3895.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 4132.10 | 4175.88 | 4066.63 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-08-06 09:15:00 | 4229.65 | 4175.24 | 4069.53 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:15:00 | 4207.05 | 4175.56 | 4070.22 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-07 09:15:00 | 4210.00 | 4176.04 | 4073.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:15:00 | 4192.90 | 4176.21 | 4074.17 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-07 13:15:00 | 4196.35 | 4176.71 | 4075.95 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 14:15:00 | 4200.00 | 4176.95 | 4076.56 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-08 10:15:00 | 4192.85 | 4177.47 | 4078.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:15:00 | 4222.55 | 4177.92 | 4079.04 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 4348.80 | 4434.09 | 4322.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 4312.15 | 4430.95 | 4322.67 | SL hit (close<ema400) qty=1.00 sl=4322.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 4312.15 | 4430.95 | 4322.67 | SL hit (close<ema400) qty=1.00 sl=4322.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 4312.15 | 4430.95 | 4322.67 | SL hit (close<ema400) qty=1.00 sl=4322.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 4312.15 | 4430.95 | 4322.67 | SL hit (close<ema400) qty=1.00 sl=4322.67 alert=retest1 |

### Cycle 5 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 4104.00 | 4278.09 | 4278.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 4084.65 | 4246.17 | 4261.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 4139.00 | 4136.25 | 4192.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 4101.35 | 4135.97 | 4191.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 4101.35 | 4135.97 | 4191.13 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-18 09:15:00 | 4040.45 | 4143.54 | 4184.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 4008.00 | 4142.20 | 4184.05 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-19 12:15:00 | 4097.10 | 4135.12 | 4178.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:15:00 | 4090.05 | 4134.67 | 4178.13 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 4216.10 | 4129.35 | 4172.37 | SL hit (close>static) qty=1.00 sl=4205.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 4216.10 | 4129.35 | 4172.37 | SL hit (close>static) qty=1.00 sl=4205.10 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 4352.55 | 4204.69 | 4204.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 4382.60 | 4207.83 | 4206.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 4304.30 | 4309.41 | 4266.09 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-12-18 11:15:00 | 4330.05 | 4309.61 | 4266.41 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 12:15:00 | 4336.00 | 4309.87 | 4266.76 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 4283.70 | 4310.83 | 4268.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 4265.15 | 4309.21 | 4268.96 | SL hit (close<ema400) qty=1.00 sl=4268.96 alert=retest1 |

### Cycle 7 — SELL (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 15:15:00 | 4112.45 | 4239.05 | 4239.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 4099.70 | 4226.29 | 4232.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 09:15:00 | 4198.60 | 4185.89 | 4210.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 4198.60 | 4185.89 | 4210.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 4198.60 | 4185.89 | 4210.00 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-01-17 09:15:00 | 4141.15 | 4203.41 | 4215.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:15:00 | 4140.00 | 4202.78 | 4215.52 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-23 10:15:00 | 4157.40 | 4179.47 | 4201.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 4164.40 | 4179.32 | 4201.13 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 3519.00 | 3931.39 | 4036.16 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 3539.74 | 3931.39 | 4036.16 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 3675.90 | 3675.23 | 3827.23 | SL hit (close>ema200) qty=0.50 sl=3675.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 3675.90 | 3675.23 | 3827.23 | SL hit (close>ema200) qty=0.50 sl=3675.23 alert=retest2 |

### Cycle 8 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 3160.90 | 3082.17 | 3082.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 10:15:00 | 3190.40 | 3097.86 | 3090.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 3204.30 | 3212.34 | 3168.31 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-01 10:15:00 | 3221.00 | 3212.38 | 3168.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 3224.80 | 3212.50 | 3169.27 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 3223.10 | 3215.13 | 3173.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:15:00 | 3228.30 | 3215.27 | 3173.62 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 3238.10 | 3215.66 | 3174.84 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 3235.00 | 3215.85 | 3175.14 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-08 11:15:00 | 3221.20 | 3221.74 | 3181.15 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-08 12:15:00 | 3218.80 | 3221.71 | 3181.33 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3191.80 | 3219.80 | 3182.70 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 3213.20 | 3219.54 | 3182.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 3211.40 | 3219.46 | 3183.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 3208.50 | 3220.80 | 3186.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-16 10:15:00 | 3200.10 | 3220.60 | 3186.85 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-16 11:15:00 | 3206.80 | 3220.46 | 3186.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 3209.90 | 3220.36 | 3187.07 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 15:15:00 | 3208.00 | 3219.94 | 3187.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-19 09:15:00 | 3184.90 | 3219.59 | 3187.34 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 3184.90 | 3219.59 | 3187.34 | SL hit (close<ema400) qty=1.00 sl=3187.34 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 3184.90 | 3219.59 | 3187.34 | SL hit (close<ema400) qty=1.00 sl=3187.34 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 3184.90 | 3219.59 | 3187.34 | SL hit (close<ema400) qty=1.00 sl=3187.34 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.09 | 3187.25 | SL hit (close<static) qty=1.00 sl=3181.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.09 | 3187.25 | SL hit (close<static) qty=1.00 sl=3181.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 10:15:00 | 3226.90 | 3185.22 | 3176.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:15:00 | 3234.70 | 3185.71 | 3176.85 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 3048.70 | 3185.85 | 3177.15 | SL hit (close<static) qty=1.00 sl=3181.60 alert=retest2 |

### Cycle 9 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 2991.80 | 3168.76 | 3168.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2934.00 | 3161.24 | 3165.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.00 | 2525.99 | 2688.88 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 2510.60 | 2531.89 | 2677.18 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 2507.60 | 2531.65 | 2676.33 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2480.60 | 2530.49 | 2671.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2486.70 | 2530.05 | 2670.55 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 11:15:00 | 2517.90 | 2540.80 | 2651.30 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-22 12:15:00 | 2523.30 | 2540.62 | 2650.66 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 2517.00 | 2540.25 | 2645.11 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:15:00 | 2460.20 | 2539.46 | 2644.19 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-05 11:15:00 | 3718.20 | 2024-03-19 15:15:00 | 3971.00 | STOP_HIT | 1.00 | 6.80% |
| BUY | retest1 | 2024-01-09 10:15:00 | 3726.30 | 2024-03-19 15:15:00 | 3971.00 | STOP_HIT | 1.00 | 6.57% |
| BUY | retest1 | 2024-01-11 11:15:00 | 3740.10 | 2024-03-19 15:15:00 | 3971.00 | STOP_HIT | 1.00 | 6.17% |
| BUY | retest2 | 2024-03-20 10:15:00 | 4005.00 | 2024-03-21 11:15:00 | 3949.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-04-04 14:15:00 | 4004.15 | 2024-04-08 15:15:00 | 3964.90 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-04-08 12:15:00 | 4014.90 | 2024-04-08 15:15:00 | 3964.90 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-04-12 11:15:00 | 4000.25 | 2024-04-15 12:15:00 | 3960.10 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-05-16 13:15:00 | 3853.95 | 2024-06-27 11:15:00 | 3910.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-05-17 14:15:00 | 3832.60 | 2024-06-27 11:15:00 | 3910.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-05-24 12:15:00 | 3858.65 | 2024-06-27 11:15:00 | 3910.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-05-27 15:15:00 | 3847.05 | 2024-06-27 11:15:00 | 3910.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-06-12 15:15:00 | 3837.00 | 2024-06-27 14:15:00 | 3933.20 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-06-14 13:15:00 | 3830.75 | 2024-06-27 14:15:00 | 3933.20 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-06-21 11:15:00 | 3837.15 | 2024-06-27 14:15:00 | 3933.20 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-06-24 14:15:00 | 3815.00 | 2024-06-27 14:15:00 | 3933.20 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest1 | 2024-08-06 10:15:00 | 4207.05 | 2024-09-19 10:15:00 | 4312.15 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest1 | 2024-08-07 10:15:00 | 4192.90 | 2024-09-19 10:15:00 | 4312.15 | STOP_HIT | 1.00 | 2.84% |
| BUY | retest1 | 2024-08-07 14:15:00 | 4200.00 | 2024-09-19 10:15:00 | 4312.15 | STOP_HIT | 1.00 | 2.67% |
| BUY | retest1 | 2024-08-08 11:15:00 | 4222.55 | 2024-09-19 10:15:00 | 4312.15 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2024-11-18 10:15:00 | 4008.00 | 2024-11-22 13:15:00 | 4216.10 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2024-11-19 13:15:00 | 4090.05 | 2024-11-22 13:15:00 | 4216.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest1 | 2024-12-18 12:15:00 | 4336.00 | 2024-12-20 10:15:00 | 4265.15 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-01-17 10:15:00 | 4140.00 | 2025-02-28 09:15:00 | 3519.00 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 4164.40 | 2025-02-28 09:15:00 | 3539.74 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-01-17 10:15:00 | 4140.00 | 2025-03-25 09:15:00 | 3675.90 | STOP_HIT | 0.50 | 11.21% |
| SELL | retest2 | 2025-01-23 11:15:00 | 4164.40 | 2025-03-25 09:15:00 | 3675.90 | STOP_HIT | 0.50 | 11.73% |
| BUY | retest1 | 2026-01-01 11:15:00 | 3224.80 | 2026-01-19 09:15:00 | 3184.90 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest1 | 2026-01-05 11:15:00 | 3228.30 | 2026-01-19 09:15:00 | 3184.90 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest1 | 2026-01-06 10:15:00 | 3235.00 | 2026-01-19 09:15:00 | 3184.90 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2026-01-12 13:15:00 | 3211.40 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-01-16 12:15:00 | 3209.90 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-02-03 11:15:00 | 3234.70 | 2026-02-04 09:15:00 | 3048.70 | STOP_HIT | 1.00 | -5.75% |
