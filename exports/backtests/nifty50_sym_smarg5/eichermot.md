# EICHERMOT (EICHERMOT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 7310.50
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
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 8 |
| PENDING | 31 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 22 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 7
- **Winners / losers:** 12 / 8
- **Target hits / Stop hits / Partials:** 1 / 18 / 1
- **Avg / median % per leg:** 6.16% / 5.19%
- **Sum % (uncompounded):** 123.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 12 | 60.0% | 1 | 18 | 1 | 6.16% | 123.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.83% | -2.8% |
| BUY @ 3rd Alert (retest2) | 19 | 12 | 63.2% | 1 | 17 | 1 | 6.64% | 126.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.83% | -2.8% |
| retest2 (combined) | 19 | 12 | 63.2% | 1 | 17 | 1 | 6.64% | 126.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 11:15:00 | 3496.35 | 3420.56 | 3420.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 12:15:00 | 3512.00 | 3421.47 | 3420.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 15:15:00 | 3446.35 | 3448.32 | 3436.16 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 15:15:00 | 3446.35 | 3448.32 | 3436.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 15:15:00 | 3446.35 | 3448.32 | 3436.16 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2023-10-31 11:15:00 | 3324.55 | 3425.86 | 3425.90 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2023-11-06 09:15:00 | 3499.00 | 3408.01 | 3416.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 10:15:00 | 3526.80 | 3409.20 | 3416.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-08 11:15:00 | 3537.15 | 3424.83 | 3424.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 11:15:00 | 3537.15 | 3424.83 | 3424.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 12:15:00 | 3553.50 | 3426.11 | 3424.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 3896.10 | 3904.64 | 3755.67 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2023-12-21 09:15:00 | 3949.60 | 3905.08 | 3756.64 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 10:15:00 | 3944.60 | 3905.48 | 3757.58 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-08 09:15:00 | 3917.50 | 3944.17 | 3829.40 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-08 10:15:00 | 3892.85 | 3943.66 | 3829.72 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 3816.90 | 3935.72 | 3832.81 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-10 09:15:00 | 3832.81 | 3935.72 | 3832.81 | SL hit qty=1.00 sl=3832.81 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-11 09:15:00 | 3889.25 | 3929.14 | 3832.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 10:15:00 | 3889.10 | 3928.74 | 3833.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 12:15:00 | 3875.55 | 3924.95 | 3835.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 13:15:00 | 3873.30 | 3924.44 | 3835.73 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-12 15:15:00 | 3870.85 | 3923.34 | 3836.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-15 09:15:00 | 3823.60 | 3922.35 | 3836.00 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2024-01-16 09:15:00 | 3790.65 | 3916.22 | 3835.86 | SL hit qty=1.00 sl=3790.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-16 09:15:00 | 3790.65 | 3916.22 | 3835.86 | SL hit qty=1.00 sl=3790.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-01 09:15:00 | 3925.90 | 3809.05 | 3796.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 10:15:00 | 3910.10 | 3810.05 | 3797.15 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-01 12:15:00 | 3904.40 | 3811.56 | 3798.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 13:15:00 | 3909.40 | 3812.54 | 3798.59 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 3798.35 | 3836.35 | 3813.75 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 3790.65 | 3835.53 | 3813.67 | SL hit qty=1.00 sl=3790.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 3790.65 | 3835.53 | 3813.67 | SL hit qty=1.00 sl=3790.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-09 14:15:00 | 3840.20 | 3834.56 | 3813.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 15:15:00 | 3840.95 | 3834.62 | 3813.85 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-13 14:15:00 | 3791.80 | 3837.97 | 3816.89 | SL hit qty=1.00 sl=3791.80 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-14 11:15:00 | 3852.00 | 3839.26 | 3817.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 12:15:00 | 3882.90 | 3839.69 | 3818.29 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-20 09:15:00 | 3791.80 | 3858.18 | 3830.63 | SL hit qty=1.00 sl=3791.80 alert=retest2 |
| Cross detected — sustain check pending | 2024-02-20 10:15:00 | 3831.90 | 3857.92 | 3830.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 11:15:00 | 3848.85 | 3857.83 | 3830.72 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-20 15:15:00 | 3830.05 | 3856.87 | 3830.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 09:15:00 | 3873.85 | 3857.04 | 3830.99 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| ALERT3_SKIP | 2024-02-21 14:15:00 | 3842.95 | 3857.47 | 3831.85 | max_alert3_locks_per_cycle=2 reached — end cycle |
| CROSSOVER_SKIP | 2024-03-15 14:15:00 | 3744.60 | 3827.99 | 3828.10 | HTF filter: close above htf_sma |

### Cycle 3 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 3998.25 | 3827.34 | 3827.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 11:15:00 | 4008.20 | 3850.16 | 3839.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 4541.00 | 4610.99 | 4420.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 4400.00 | 4608.89 | 4420.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 4400.00 | 4608.89 | 4420.38 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-05 11:15:00 | 4626.00 | 4604.52 | 4424.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-05 12:15:00 | 4589.65 | 4604.37 | 4425.44 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-06 10:15:00 | 4646.10 | 4603.56 | 4429.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 11:15:00 | 4653.00 | 4604.05 | 4430.56 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-02 12:15:00 | 4625.80 | 4727.96 | 4592.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:15:00 | 4629.75 | 4726.99 | 4592.53 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-05 11:15:00 | 4639.25 | 4841.51 | 4740.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 12:15:00 | 4639.05 | 4839.50 | 4740.01 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-07 10:15:00 | 4625.85 | 4819.62 | 4735.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 11:15:00 | 4627.35 | 4817.70 | 4735.05 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 4595.00 | 4815.49 | 4734.35 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-08-09 09:15:00 | 4773.80 | 4793.92 | 4727.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 4785.90 | 4793.84 | 4727.88 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 4593.75 | 4858.67 | 4829.48 | SL hit qty=1.00 sl=4593.75 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-08 10:15:00 | 4648.85 | 4856.58 | 4828.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 11:15:00 | 4640.55 | 4854.43 | 4827.64 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| CROSSOVER_SKIP | 2024-10-16 15:15:00 | 4678.25 | 4805.25 | 4805.78 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 4593.75 | 4802.88 | 4804.59 | SL hit qty=1.00 sl=4593.75 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-17 12:15:00 | 4647.35 | 4797.43 | 4801.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-17 13:15:00 | 4610.95 | 4795.58 | 4800.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-18 09:15:00 | 4745.25 | 4791.74 | 4798.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:15:00 | 4771.25 | 4791.54 | 4798.72 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 4593.75 | 4773.92 | 4788.47 | SL hit qty=1.00 sl=4593.75 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-28 11:15:00 | 4680.20 | 4757.72 | 4779.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 12:15:00 | 4689.80 | 4757.05 | 4779.08 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2024-10-28 13:15:00 | 4717.90 | 4756.66 | 4778.77 | max_alert3_locks_per_cycle=2 reached — end cycle |

### Cycle 4 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 4912.25 | 4796.91 | 4796.40 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2024-11-13 09:15:00 | 4598.00 | 4796.17 | 4796.41 | HTF filter: close above htf_sma |
| First Alert — break + close above crossover candle high | 2024-11-14 09:15:00 | 4945.70 | 4784.97 | 4790.67 | Break + close above crossover candle high |

### Cycle 5 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 4893.10 | 4796.32 | 4796.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 4975.00 | 4800.50 | 4798.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 4821.35 | 4850.05 | 4826.40 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 4821.35 | 4850.05 | 4826.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 4821.35 | 4850.05 | 4826.40 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-06 09:15:00 | 4864.50 | 4840.23 | 4825.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:15:00 | 4895.55 | 4840.78 | 4825.72 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-10 13:15:00 | 4804.00 | 4841.05 | 4827.13 | SL hit qty=1.00 sl=4804.00 alert=retest2 |
| CROSSOVER_SKIP | 2024-12-20 14:15:00 | 4724.15 | 4816.63 | 4817.05 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2024-12-27 09:15:00 | 4935.45 | 4809.81 | 4813.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 10:15:00 | 4883.95 | 4810.55 | 4813.65 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-27 12:15:00 | 4877.45 | 4811.58 | 4814.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 13:15:00 | 4880.35 | 4812.27 | 4814.47 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 4879.75 | 4817.01 | 4816.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 4879.75 | 4817.01 | 4816.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 4879.75 | 4817.01 | 4816.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 4900.45 | 4821.54 | 4819.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 10:15:00 | 4943.40 | 4960.77 | 4899.21 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-14 10:15:00 | 5004.25 | 4960.47 | 4901.17 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-14 11:15:00 | 4992.35 | 4960.79 | 4901.63 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-14 12:15:00 | 5039.10 | 4961.57 | 4902.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 13:15:00 | 5027.50 | 4962.23 | 4902.94 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-20 11:15:00 | 5015.45 | 4980.35 | 4919.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 12:15:00 | 5016.45 | 4980.71 | 4920.39 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-22 12:15:00 | 5006.50 | 4983.43 | 4925.90 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 13:15:00 | 5005.40 | 4983.65 | 4926.29 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 5055.50 | 5173.74 | 5060.21 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 5060.21 | 5173.74 | 5060.21 | SL hit qty=1.00 sl=5060.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 5060.21 | 5173.74 | 5060.21 | SL hit qty=1.00 sl=5060.21 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 5060.21 | 5173.74 | 5060.21 | SL hit qty=1.00 sl=5060.21 alert=retest1 |
| CROSSOVER_SKIP | 2025-03-04 09:15:00 | 4847.00 | 4986.61 | 4986.65 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-03-07 09:15:00 | 5115.00 | 4984.08 | 4985.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 5125.40 | 4985.49 | 4985.74 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-07 11:15:00 | 5106.85 | 4986.69 | 4986.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 11:15:00 | 5106.85 | 4986.69 | 4986.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 5138.45 | 5005.60 | 4996.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 5104.50 | 5192.71 | 5112.29 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 5104.50 | 5192.71 | 5112.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 5104.50 | 5192.71 | 5112.29 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 5181.30 | 5185.69 | 5111.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 5183.80 | 5185.67 | 5111.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 5961.37 | 5601.77 | 5546.40 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2025-09-08 11:15:00 | 6738.94 | 5994.57 | 5800.20 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2026-03-20 11:15:00 | 6854.00 | 7353.65 | 7355.44 | HTF filter: close above htf_sma |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-11-06 10:15:00 | 3526.80 | 2023-11-08 11:15:00 | 3537.15 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest1 | 2023-12-21 10:15:00 | 3944.60 | 2024-01-10 09:15:00 | 3832.81 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-01-11 10:15:00 | 3889.10 | 2024-01-16 09:15:00 | 3790.65 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2024-01-12 13:15:00 | 3873.30 | 2024-01-16 09:15:00 | 3790.65 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-02-01 10:15:00 | 3910.10 | 2024-02-09 09:15:00 | 3790.65 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-02-01 13:15:00 | 3909.40 | 2024-02-09 09:15:00 | 3790.65 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-02-09 15:15:00 | 3840.95 | 2024-02-13 14:15:00 | 3791.80 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-02-14 12:15:00 | 3882.90 | 2024-02-20 09:15:00 | 3791.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-02-20 11:15:00 | 3848.85 | 2024-10-08 09:15:00 | 4593.75 | STOP_HIT | 1.00 | 19.35% |
| BUY | retest2 | 2024-02-21 09:15:00 | 3873.85 | 2024-10-17 09:15:00 | 4593.75 | STOP_HIT | 1.00 | 18.58% |
| BUY | retest2 | 2024-06-06 11:15:00 | 4653.00 | 2024-10-25 09:15:00 | 4593.75 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-07-02 13:15:00 | 4629.75 | 2024-12-10 13:15:00 | 4804.00 | STOP_HIT | 1.00 | 3.76% |
| BUY | retest2 | 2024-08-05 12:15:00 | 4639.05 | 2024-12-30 13:15:00 | 4879.75 | STOP_HIT | 1.00 | 5.19% |
| BUY | retest2 | 2024-08-07 11:15:00 | 4627.35 | 2024-12-30 13:15:00 | 4879.75 | STOP_HIT | 1.00 | 5.45% |
| BUY | retest2 | 2024-08-09 10:15:00 | 4785.90 | 2025-02-11 09:15:00 | 5060.21 | STOP_HIT | 1.00 | 5.73% |
| BUY | retest2 | 2024-10-08 11:15:00 | 4640.55 | 2025-02-11 09:15:00 | 5060.21 | STOP_HIT | 1.00 | 9.04% |
| BUY | retest2 | 2024-10-18 10:15:00 | 4771.25 | 2025-02-11 09:15:00 | 5060.21 | STOP_HIT | 1.00 | 6.06% |
| BUY | retest2 | 2024-10-28 12:15:00 | 4689.80 | 2025-03-07 11:15:00 | 5106.85 | STOP_HIT | 1.00 | 8.89% |
| BUY | retest2 | 2024-12-06 10:15:00 | 4895.55 | 2025-08-18 09:15:00 | 5961.37 | PARTIAL | 0.50 | 21.77% |
| BUY | retest2 | 2024-12-06 10:15:00 | 4895.55 | 2025-09-08 11:15:00 | 6738.94 | TARGET_HIT | 0.50 | 37.65% |
