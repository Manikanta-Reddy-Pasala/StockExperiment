# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 7302.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 9 |
| ALERT3 | 15 |
| PENDING | 38 |
| PENDING_CANCEL | 10 |
| ENTRY1 | 4 |
| ENTRY2 | 24 |
| PARTIAL | 3 |
| TARGET_HIT | 7 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 19
- **Target hits / Stop hits / Partials:** 7 / 21 / 3
- **Avg / median % per leg:** 0.94% / -1.96%
- **Sum % (uncompounded):** 29.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 12 | 48.0% | 7 | 15 | 3 | 2.14% | 53.4% |
| BUY @ 2nd Alert (retest1) | 7 | 7 | 100.0% | 3 | 1 | 3 | 6.42% | 45.0% |
| BUY @ 3rd Alert (retest2) | 18 | 5 | 27.8% | 4 | 14 | 0 | 0.47% | 8.4% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -4.05% | -24.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -4.05% | -24.3% |
| retest1 (combined) | 7 | 7 | 100.0% | 3 | 1 | 3 | 6.42% | 45.0% |
| retest2 (combined) | 24 | 5 | 20.8% | 4 | 20 | 0 | -0.66% | -15.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 10:15:00 | 3466.05 | 3408.99 | 3408.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 3476.20 | 3416.01 | 3412.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 15:15:00 | 3446.35 | 3448.56 | 3432.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 15:15:00 | 3446.35 | 3448.56 | 3432.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 15:15:00 | 3446.35 | 3448.56 | 3432.15 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 10:15:00 | 3303.30 | 3418.78 | 3418.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 14:15:00 | 3282.80 | 3414.05 | 3416.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 10:15:00 | 3407.30 | 3406.33 | 3412.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 11:15:00 | 3411.00 | 3406.37 | 3412.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 11:15:00 | 3411.00 | 3406.37 | 3412.40 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 11:15:00 | 3522.15 | 3417.98 | 3417.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 12:15:00 | 3553.50 | 3426.22 | 3422.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 3896.10 | 3904.65 | 3754.65 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-12-21 09:15:00 | 3949.60 | 3905.10 | 3755.62 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 10:15:00 | 3944.60 | 3905.49 | 3756.56 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 09:15:00 | 4141.83 | 3946.00 | 3801.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-02 09:15:00 | 3952.30 | 3962.93 | 3820.05 | SL hit (close<ema200) qty=0.50 sl=3962.93 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-08 09:15:00 | 3917.50 | 3944.18 | 3828.71 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-08 10:15:00 | 3892.85 | 3943.67 | 3829.03 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 3816.90 | 3935.73 | 3832.16 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-11 09:15:00 | 3889.25 | 3929.15 | 3832.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 10:15:00 | 3889.10 | 3928.75 | 3832.65 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 12:15:00 | 3875.55 | 3924.96 | 3834.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 13:15:00 | 3873.30 | 3924.45 | 3835.14 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 15:15:00 | 3870.85 | 3923.34 | 3835.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-15 09:15:00 | 3823.60 | 3922.35 | 3835.42 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2024-01-16 11:15:00 | 3785.10 | 3913.90 | 3834.93 | SL hit (close<static) qty=1.00 sl=3790.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-16 11:15:00 | 3785.10 | 3913.90 | 3834.93 | SL hit (close<static) qty=1.00 sl=3790.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-01 09:15:00 | 3925.90 | 3809.05 | 3796.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 10:15:00 | 3910.10 | 3810.05 | 3796.75 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-01 12:15:00 | 3904.40 | 3811.56 | 3797.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 13:15:00 | 3909.40 | 3812.54 | 3798.20 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 3798.35 | 3836.35 | 3813.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-09 10:15:00 | 3776.15 | 3834.94 | 3813.17 | SL hit (close<static) qty=1.00 sl=3790.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-09 10:15:00 | 3776.15 | 3834.94 | 3813.17 | SL hit (close<static) qty=1.00 sl=3790.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-09 14:15:00 | 3840.20 | 3834.56 | 3813.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 15:15:00 | 3840.95 | 3834.62 | 3813.54 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-14 11:15:00 | 3852.00 | 3839.26 | 3817.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 12:15:00 | 3882.90 | 3839.69 | 3818.00 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-20 10:15:00 | 3831.90 | 3857.92 | 3830.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 11:15:00 | 3848.85 | 3857.83 | 3830.47 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-20 15:15:00 | 3830.05 | 3856.87 | 3830.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 09:15:00 | 3873.85 | 3857.04 | 3830.75 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 3842.95 | 3857.47 | 3831.62 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-02-22 09:15:00 | 3901.80 | 3857.70 | 3831.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 10:15:00 | 3895.20 | 3858.07 | 3832.31 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-29 11:15:00 | 3811.40 | 3880.21 | 3848.75 | SL hit (close<static) qty=1.00 sl=3821.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-29 12:15:00 | 3773.50 | 3879.15 | 3848.37 | SL hit (close<static) qty=1.00 sl=3791.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-29 12:15:00 | 3773.50 | 3879.15 | 3848.37 | SL hit (close<static) qty=1.00 sl=3791.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-29 12:15:00 | 3773.50 | 3879.15 | 3848.37 | SL hit (close<static) qty=1.00 sl=3791.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-29 12:15:00 | 3773.50 | 3879.15 | 3848.37 | SL hit (close<static) qty=1.00 sl=3791.80 alert=retest2 |

### Cycle 4 — SELL (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 15:15:00 | 3745.65 | 3827.17 | 3827.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 09:15:00 | 3702.20 | 3825.93 | 3826.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 09:15:00 | 3859.90 | 3811.78 | 3819.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 3859.90 | 3811.78 | 3819.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 3859.90 | 3811.78 | 3819.51 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 3998.25 | 3827.34 | 3826.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 11:15:00 | 4008.20 | 3850.16 | 3838.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 4541.00 | 4610.99 | 4420.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 4400.00 | 4608.89 | 4420.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 4400.00 | 4608.89 | 4420.36 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-06-05 11:15:00 | 4626.00 | 4604.52 | 4424.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-05 12:15:00 | 4589.65 | 4604.37 | 4425.42 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-06 10:15:00 | 4646.10 | 4603.56 | 4429.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 11:15:00 | 4653.00 | 4604.05 | 4430.54 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-02 12:15:00 | 4625.80 | 4727.96 | 4592.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:15:00 | 4629.75 | 4726.99 | 4592.52 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-05 11:15:00 | 4639.25 | 4841.51 | 4740.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 12:15:00 | 4639.05 | 4839.50 | 4740.01 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-07 10:15:00 | 4625.85 | 4819.62 | 4735.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 11:15:00 | 4627.35 | 4817.70 | 4735.04 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 4760.65 | 4797.55 | 4736.45 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-08-16 11:15:00 | 4789.00 | 4793.76 | 4736.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 12:15:00 | 4794.75 | 4793.77 | 4737.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 4726.10 | 4846.78 | 4795.91 | SL hit (close<static) qty=1.00 sl=4736.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-12 13:15:00 | 4814.80 | 4824.00 | 4789.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:15:00 | 4886.00 | 4824.62 | 4790.25 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2024-09-27 12:15:00 | 5092.73 | 4868.69 | 4826.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-09-27 12:15:00 | 5102.95 | 4868.69 | 4826.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-09-27 12:15:00 | 5090.09 | 4868.69 | 4826.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-03 10:15:00 | 4789.00 | 4888.52 | 4840.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 11:15:00 | 4840.00 | 4888.04 | 4840.42 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-03 15:15:00 | 4782.50 | 4884.02 | 4839.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-04 09:15:00 | 4773.00 | 4882.92 | 4839.00 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2024-10-04 11:15:00 | 4789.05 | 4880.58 | 4838.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-04 12:15:00 | 4763.70 | 4879.41 | 4837.89 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-10-04 13:15:00 | 4721.50 | 4877.84 | 4837.31 | SL hit (close<static) qty=1.00 sl=4736.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 13:15:00 | 4721.50 | 4877.84 | 4837.31 | SL hit (close<static) qty=1.00 sl=4736.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-11 09:15:00 | 4802.40 | 4830.52 | 4817.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-11 10:15:00 | 4729.10 | 4829.51 | 4817.06 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-14 09:15:00 | 4784.00 | 4824.23 | 4814.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-14 10:15:00 | 4753.00 | 4823.52 | 4814.44 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-14 14:15:00 | 4780.15 | 4821.61 | 4813.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-14 15:15:00 | 4777.55 | 4821.17 | 4813.48 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-10-16 15:15:00 | 4678.25 | 4805.25 | 4805.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 4678.25 | 4805.25 | 4805.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 4567.35 | 4802.88 | 4804.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 4845.55 | 4790.91 | 4798.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 4845.55 | 4790.91 | 4798.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 4845.55 | 4790.91 | 4798.15 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-10-22 09:15:00 | 4772.95 | 4791.60 | 4798.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-22 10:15:00 | 4799.90 | 4791.68 | 4798.29 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-22 14:15:00 | 4752.95 | 4791.27 | 4797.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 15:15:00 | 4759.05 | 4790.95 | 4797.76 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 4908.05 | 4758.98 | 4779.06 | SL hit (close>static) qty=1.00 sl=4851.90 alert=retest2 |

### Cycle 7 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 4912.25 | 4796.91 | 4796.40 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 4598.00 | 4796.17 | 4796.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 4566.60 | 4791.79 | 4794.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 4945.70 | 4784.97 | 4790.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 4945.70 | 4784.97 | 4790.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 4945.70 | 4784.97 | 4790.67 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 4893.10 | 4796.32 | 4796.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 4975.00 | 4800.50 | 4798.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 4821.35 | 4850.05 | 4826.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 4821.35 | 4850.05 | 4826.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 4821.35 | 4850.05 | 4826.39 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-12-06 09:15:00 | 4864.50 | 4840.23 | 4825.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:15:00 | 4895.55 | 4840.78 | 4825.72 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-11 13:15:00 | 4803.45 | 4839.15 | 4826.64 | SL hit (close<static) qty=1.00 sl=4804.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 4724.15 | 4816.63 | 4817.05 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 4879.75 | 4817.01 | 4816.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 4900.45 | 4821.54 | 4819.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 10:15:00 | 4943.40 | 4960.77 | 4899.21 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-01-14 10:15:00 | 5004.25 | 4960.47 | 4901.17 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-14 11:15:00 | 4992.35 | 4960.79 | 4901.63 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-14 12:15:00 | 5039.10 | 4961.57 | 4902.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 13:15:00 | 5027.50 | 4962.23 | 4902.94 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-20 11:15:00 | 5015.45 | 4980.35 | 4919.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 12:15:00 | 5016.45 | 4980.71 | 4920.39 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-22 12:15:00 | 5006.50 | 4983.43 | 4925.90 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 13:15:00 | 5005.40 | 4983.65 | 4926.29 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:15:00 | 5255.67 | 5036.97 | 4965.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 5278.88 | 5060.51 | 4982.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-03 09:15:00 | 5518.10 | 5060.51 | 4982.07 | Target hit (10%) qty=1.00 alert=retest1 |
| Target hit | 2025-02-03 09:15:00 | 5505.94 | 5060.51 | 4982.07 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2025-02-03 10:15:00 | 5530.25 | 5065.02 | 4984.72 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 5055.50 | 5173.74 | 5060.21 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 09:15:00 | 4847.00 | 4986.61 | 4986.65 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 11:15:00 | 5106.85 | 4986.69 | 4986.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 5138.45 | 5005.60 | 4996.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 5104.50 | 5192.71 | 5112.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 5104.50 | 5192.71 | 5112.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 5104.50 | 5192.71 | 5112.29 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 5181.30 | 5185.69 | 5111.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 5183.80 | 5185.67 | 5111.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-04-17 13:15:00 | 5702.18 | 5270.12 | 5169.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 6854.00 | 7353.65 | 7355.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 7329.94 | 7343.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 11:15:00 | 7072.00 | 7049.56 | 7178.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 7321.50 | 7059.42 | 7176.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 7321.50 | 7059.42 | 7176.16 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 7142.50 | 7079.32 | 7182.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 7115.00 | 7079.68 | 7181.95 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 7165.00 | 7104.19 | 7177.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 7141.50 | 7104.56 | 7177.35 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 7178.50 | 7110.59 | 7178.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 7094.50 | 7110.43 | 7177.84 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-28 10:15:00 | 7200.50 | 7115.60 | 7173.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 7065.50 | 7115.11 | 7173.19 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 7233.50 | 7114.90 | 7171.64 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 6984.00 | 7120.12 | 7172.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 7009.50 | 7119.02 | 7171.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-04 13:15:00 | 7308.00 | 7122.13 | 7170.54 | SL hit (close>static) qty=1.00 sl=7294.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7149.57 | 7180.89 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7149.57 | 7180.89 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7149.57 | 7180.89 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7149.57 | 7180.89 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-21 10:15:00 | 3944.60 | 2023-12-29 09:15:00 | 4141.83 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-12-21 10:15:00 | 3944.60 | 2024-01-02 09:15:00 | 3952.30 | STOP_HIT | 0.50 | 0.20% |
| BUY | retest2 | 2024-01-11 10:15:00 | 3889.10 | 2024-01-16 11:15:00 | 3785.10 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-01-12 13:15:00 | 3873.30 | 2024-01-16 11:15:00 | 3785.10 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-02-01 10:15:00 | 3910.10 | 2024-02-09 10:15:00 | 3776.15 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2024-02-01 13:15:00 | 3909.40 | 2024-02-09 10:15:00 | 3776.15 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2024-02-09 15:15:00 | 3840.95 | 2024-02-29 11:15:00 | 3811.40 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-02-14 12:15:00 | 3882.90 | 2024-02-29 12:15:00 | 3773.50 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-02-20 11:15:00 | 3848.85 | 2024-02-29 12:15:00 | 3773.50 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-02-21 09:15:00 | 3873.85 | 2024-02-29 12:15:00 | 3773.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-02-22 10:15:00 | 3895.20 | 2024-02-29 12:15:00 | 3773.50 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-06-06 11:15:00 | 4653.00 | 2024-09-09 09:15:00 | 4726.10 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2024-07-02 13:15:00 | 4629.75 | 2024-09-27 12:15:00 | 5092.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-05 12:15:00 | 4639.05 | 2024-09-27 12:15:00 | 5102.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-07 11:15:00 | 4627.35 | 2024-09-27 12:15:00 | 5090.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-16 12:15:00 | 4794.75 | 2024-10-04 13:15:00 | 4721.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-09-12 14:15:00 | 4886.00 | 2024-10-04 13:15:00 | 4721.50 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2024-10-03 11:15:00 | 4840.00 | 2024-10-16 15:15:00 | 4678.25 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2024-10-22 15:15:00 | 4759.05 | 2024-10-29 14:15:00 | 4908.05 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2024-12-06 10:15:00 | 4895.55 | 2024-12-11 13:15:00 | 4803.45 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest1 | 2025-01-14 13:15:00 | 5027.50 | 2025-01-30 09:15:00 | 5255.67 | PARTIAL | 0.50 | 4.54% |
| BUY | retest1 | 2025-01-20 12:15:00 | 5016.45 | 2025-02-03 09:15:00 | 5278.88 | PARTIAL | 0.50 | 5.23% |
| BUY | retest1 | 2025-01-14 13:15:00 | 5027.50 | 2025-02-03 09:15:00 | 5518.10 | TARGET_HIT | 0.50 | 9.76% |
| BUY | retest1 | 2025-01-20 12:15:00 | 5016.45 | 2025-02-03 09:15:00 | 5505.94 | TARGET_HIT | 0.50 | 9.76% |
| BUY | retest1 | 2025-01-22 13:15:00 | 5005.40 | 2025-02-03 10:15:00 | 5530.25 | TARGET_HIT | 1.00 | 10.49% |
| BUY | retest2 | 2025-04-08 10:15:00 | 5183.80 | 2025-04-17 13:15:00 | 5702.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-13 10:15:00 | 7115.00 | 2026-05-04 13:15:00 | 7308.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-22 10:15:00 | 7141.50 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-04-23 10:15:00 | 7094.50 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-04-28 11:15:00 | 7065.50 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest2 | 2026-04-30 10:15:00 | 7009.50 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -5.62% |
