# PI Industries Ltd. (PIIND)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 3103.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 86 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 14 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 119 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 72
- **Target hits / Stop hits / Partials:** 14 / 72 / 33
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 14.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 19 | 32.8% | 5 | 39 | 14 | 0.07% | 4.0% |
| BUY @ 2nd Alert (retest1) | 58 | 19 | 32.8% | 5 | 39 | 14 | 0.07% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 61 | 28 | 45.9% | 9 | 33 | 19 | 0.17% | 10.2% |
| SELL @ 2nd Alert (retest1) | 61 | 28 | 45.9% | 9 | 33 | 19 | 0.17% | 10.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 119 | 47 | 39.5% | 14 | 72 | 33 | 0.12% | 14.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 11:10:00 | 3641.60 | 3649.29 | 0.00 | ORB-short ORB[3644.20,3667.90] vol=2.4x ATR=8.06 |
| Stop hit — per-position SL triggered | 2025-05-15 11:20:00 | 3649.66 | 3649.13 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 10:20:00 | 3687.50 | 3703.06 | 0.00 | ORB-short ORB[3687.90,3725.00] vol=3.2x ATR=12.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 10:30:00 | 3668.37 | 3685.36 | 0.00 | T1 1.5R @ 3668.37 |
| Stop hit — per-position SL triggered | 2025-05-16 10:45:00 | 3687.50 | 3683.69 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 11:10:00 | 3642.00 | 3663.19 | 0.00 | ORB-short ORB[3658.80,3701.10] vol=1.7x ATR=9.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 12:20:00 | 3628.16 | 3656.22 | 0.00 | T1 1.5R @ 3628.16 |
| Stop hit — per-position SL triggered | 2025-05-22 13:20:00 | 3642.00 | 3641.53 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:05:00 | 3715.80 | 3691.22 | 0.00 | ORB-long ORB[3664.20,3694.10] vol=2.0x ATR=10.76 |
| Stop hit — per-position SL triggered | 2025-05-27 10:10:00 | 3705.04 | 3700.21 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:05:00 | 3795.50 | 3773.34 | 0.00 | ORB-long ORB[3740.00,3793.30] vol=2.0x ATR=12.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:25:00 | 3814.12 | 3791.60 | 0.00 | T1 1.5R @ 3814.12 |
| Target hit | 2025-05-28 15:20:00 | 3861.30 | 3828.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:55:00 | 3843.40 | 3868.59 | 0.00 | ORB-short ORB[3866.90,3920.00] vol=1.6x ATR=9.09 |
| Stop hit — per-position SL triggered | 2025-05-30 11:00:00 | 3852.49 | 3866.07 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 10:50:00 | 3781.20 | 3798.63 | 0.00 | ORB-short ORB[3800.50,3826.00] vol=2.0x ATR=11.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 11:25:00 | 3763.29 | 3791.73 | 0.00 | T1 1.5R @ 3763.29 |
| Stop hit — per-position SL triggered | 2025-06-02 11:50:00 | 3781.20 | 3786.46 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:50:00 | 3878.80 | 3859.65 | 0.00 | ORB-long ORB[3829.90,3873.80] vol=2.2x ATR=11.83 |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 3866.97 | 3865.53 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 09:55:00 | 3834.70 | 3810.29 | 0.00 | ORB-long ORB[3785.50,3815.40] vol=1.8x ATR=12.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 10:00:00 | 3852.73 | 3821.36 | 0.00 | T1 1.5R @ 3852.73 |
| Target hit | 2025-06-04 14:05:00 | 3960.00 | 3964.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2025-06-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:00:00 | 3948.70 | 3937.16 | 0.00 | ORB-long ORB[3918.90,3945.10] vol=2.2x ATR=8.23 |
| Stop hit — per-position SL triggered | 2025-06-10 10:55:00 | 3940.47 | 3944.80 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:10:00 | 4111.00 | 4088.72 | 0.00 | ORB-long ORB[4066.60,4105.40] vol=1.6x ATR=12.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 14:40:00 | 4129.36 | 4102.48 | 0.00 | T1 1.5R @ 4129.36 |
| Target hit | 2025-06-23 15:20:00 | 4152.70 | 4112.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:40:00 | 4232.40 | 4205.16 | 0.00 | ORB-long ORB[4156.70,4200.00] vol=2.4x ATR=11.73 |
| Stop hit — per-position SL triggered | 2025-06-25 09:45:00 | 4220.67 | 4209.49 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 11:15:00 | 4150.00 | 4141.02 | 0.00 | ORB-long ORB[4112.10,4139.70] vol=1.7x ATR=7.76 |
| Stop hit — per-position SL triggered | 2025-06-27 13:00:00 | 4142.24 | 4143.00 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:35:00 | 4191.00 | 4178.28 | 0.00 | ORB-long ORB[4161.00,4189.00] vol=1.6x ATR=9.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 09:45:00 | 4204.66 | 4186.27 | 0.00 | T1 1.5R @ 4204.66 |
| Stop hit — per-position SL triggered | 2025-07-03 09:50:00 | 4191.00 | 4188.27 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:15:00 | 4175.20 | 4188.79 | 0.00 | ORB-short ORB[4184.30,4220.00] vol=2.4x ATR=8.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:05:00 | 4162.84 | 4182.99 | 0.00 | T1 1.5R @ 4162.84 |
| Target hit | 2025-07-07 15:20:00 | 4161.10 | 4163.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-07-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:50:00 | 4184.50 | 4158.15 | 0.00 | ORB-long ORB[4126.00,4163.00] vol=2.6x ATR=7.85 |
| Stop hit — per-position SL triggered | 2025-07-09 11:05:00 | 4176.65 | 4162.36 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:00:00 | 4139.70 | 4110.06 | 0.00 | ORB-long ORB[4059.90,4099.00] vol=1.6x ATR=8.33 |
| Stop hit — per-position SL triggered | 2025-07-15 11:20:00 | 4131.37 | 4112.12 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:55:00 | 4116.70 | 4137.07 | 0.00 | ORB-short ORB[4141.10,4159.40] vol=1.5x ATR=7.14 |
| Stop hit — per-position SL triggered | 2025-07-16 11:25:00 | 4123.84 | 4129.80 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 11:05:00 | 4118.90 | 4100.72 | 0.00 | ORB-long ORB[4073.90,4116.50] vol=5.3x ATR=9.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 11:25:00 | 4133.41 | 4104.99 | 0.00 | T1 1.5R @ 4133.41 |
| Target hit | 2025-07-17 15:20:00 | 4131.70 | 4118.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-07-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 11:05:00 | 4155.80 | 4136.58 | 0.00 | ORB-long ORB[4100.00,4128.10] vol=1.9x ATR=8.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 12:20:00 | 4168.41 | 4143.69 | 0.00 | T1 1.5R @ 4168.41 |
| Stop hit — per-position SL triggered | 2025-07-21 13:50:00 | 4155.80 | 4148.45 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 11:05:00 | 4026.30 | 4045.32 | 0.00 | ORB-short ORB[4038.00,4085.00] vol=3.3x ATR=8.51 |
| Stop hit — per-position SL triggered | 2025-07-23 11:45:00 | 4034.81 | 4040.08 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:45:00 | 4096.50 | 4086.86 | 0.00 | ORB-long ORB[4054.60,4089.00] vol=2.1x ATR=12.44 |
| Stop hit — per-position SL triggered | 2025-07-28 09:55:00 | 4084.06 | 4088.56 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:30:00 | 4286.50 | 4272.28 | 0.00 | ORB-long ORB[4249.50,4279.30] vol=1.9x ATR=10.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 09:40:00 | 4302.30 | 4286.45 | 0.00 | T1 1.5R @ 4302.30 |
| Stop hit — per-position SL triggered | 2025-07-30 09:50:00 | 4286.50 | 4291.26 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 09:35:00 | 4028.60 | 4054.95 | 0.00 | ORB-short ORB[4040.00,4100.30] vol=2.5x ATR=13.25 |
| Stop hit — per-position SL triggered | 2025-08-07 09:45:00 | 4041.85 | 4049.40 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:55:00 | 3864.00 | 3844.69 | 0.00 | ORB-long ORB[3813.10,3857.60] vol=3.1x ATR=12.21 |
| Stop hit — per-position SL triggered | 2025-08-12 11:10:00 | 3851.79 | 3845.81 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:50:00 | 3914.00 | 3877.28 | 0.00 | ORB-long ORB[3844.00,3878.90] vol=1.9x ATR=11.05 |
| Stop hit — per-position SL triggered | 2025-08-25 10:00:00 | 3902.95 | 3880.78 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:45:00 | 3718.40 | 3703.84 | 0.00 | ORB-long ORB[3692.00,3713.80] vol=3.6x ATR=8.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 11:35:00 | 3731.21 | 3710.02 | 0.00 | T1 1.5R @ 3731.21 |
| Stop hit — per-position SL triggered | 2025-08-29 12:00:00 | 3718.40 | 3711.71 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:45:00 | 3719.40 | 3708.94 | 0.00 | ORB-long ORB[3696.00,3717.50] vol=3.7x ATR=9.29 |
| Stop hit — per-position SL triggered | 2025-09-01 11:50:00 | 3710.11 | 3713.91 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 11:05:00 | 3727.30 | 3740.70 | 0.00 | ORB-short ORB[3739.10,3765.80] vol=1.9x ATR=7.40 |
| Stop hit — per-position SL triggered | 2025-09-03 11:10:00 | 3734.70 | 3740.57 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 10:15:00 | 3761.20 | 3774.26 | 0.00 | ORB-short ORB[3773.40,3797.60] vol=2.8x ATR=9.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 10:35:00 | 3747.26 | 3771.98 | 0.00 | T1 1.5R @ 3747.26 |
| Target hit | 2025-09-04 15:20:00 | 3745.20 | 3745.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-09-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:35:00 | 3725.00 | 3738.59 | 0.00 | ORB-short ORB[3738.70,3760.00] vol=2.5x ATR=6.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:45:00 | 3714.66 | 3737.66 | 0.00 | T1 1.5R @ 3714.66 |
| Target hit | 2025-09-05 15:20:00 | 3661.10 | 3699.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-09-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:40:00 | 3702.70 | 3682.96 | 0.00 | ORB-long ORB[3654.00,3676.70] vol=1.7x ATR=8.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 11:10:00 | 3715.66 | 3690.27 | 0.00 | T1 1.5R @ 3715.66 |
| Stop hit — per-position SL triggered | 2025-09-08 12:40:00 | 3702.70 | 3696.48 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:50:00 | 3669.50 | 3682.34 | 0.00 | ORB-short ORB[3680.00,3708.60] vol=2.6x ATR=5.98 |
| Stop hit — per-position SL triggered | 2025-09-09 11:00:00 | 3675.48 | 3681.91 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:45:00 | 3737.80 | 3728.48 | 0.00 | ORB-long ORB[3705.30,3735.00] vol=2.3x ATR=9.33 |
| Stop hit — per-position SL triggered | 2025-09-10 10:20:00 | 3728.47 | 3733.96 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:05:00 | 3715.60 | 3722.19 | 0.00 | ORB-short ORB[3717.00,3738.80] vol=2.1x ATR=4.47 |
| Stop hit — per-position SL triggered | 2025-09-16 11:10:00 | 3720.07 | 3722.08 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:15:00 | 3787.60 | 3764.90 | 0.00 | ORB-long ORB[3714.50,3740.00] vol=3.0x ATR=9.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 11:25:00 | 3801.96 | 3781.65 | 0.00 | T1 1.5R @ 3801.96 |
| Stop hit — per-position SL triggered | 2025-09-17 11:30:00 | 3787.60 | 3782.21 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 10:05:00 | 3715.30 | 3729.21 | 0.00 | ORB-short ORB[3725.50,3776.00] vol=2.9x ATR=9.80 |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 3725.10 | 3727.66 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:10:00 | 3598.50 | 3631.97 | 0.00 | ORB-short ORB[3650.50,3681.00] vol=1.7x ATR=9.03 |
| Stop hit — per-position SL triggered | 2025-09-23 11:25:00 | 3607.53 | 3628.95 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:15:00 | 3601.00 | 3616.10 | 0.00 | ORB-short ORB[3605.50,3639.90] vol=3.5x ATR=7.33 |
| Stop hit — per-position SL triggered | 2025-09-24 12:50:00 | 3608.33 | 3611.22 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:45:00 | 3543.00 | 3552.27 | 0.00 | ORB-short ORB[3548.00,3593.90] vol=2.1x ATR=8.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:00:00 | 3529.86 | 3548.16 | 0.00 | T1 1.5R @ 3529.86 |
| Target hit | 2025-09-26 14:10:00 | 3508.30 | 3506.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — BUY (started 2025-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:35:00 | 3610.90 | 3591.52 | 0.00 | ORB-long ORB[3562.20,3593.00] vol=1.9x ATR=9.60 |
| Stop hit — per-position SL triggered | 2025-10-03 09:40:00 | 3601.30 | 3594.40 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:05:00 | 3557.00 | 3572.29 | 0.00 | ORB-short ORB[3563.60,3605.00] vol=2.6x ATR=7.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:40:00 | 3545.37 | 3567.67 | 0.00 | T1 1.5R @ 3545.37 |
| Target hit | 2025-10-08 15:20:00 | 3500.20 | 3512.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-10-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:05:00 | 3562.70 | 3545.98 | 0.00 | ORB-long ORB[3524.00,3553.50] vol=2.1x ATR=7.06 |
| Stop hit — per-position SL triggered | 2025-10-10 10:20:00 | 3555.64 | 3549.07 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 11:00:00 | 3568.70 | 3535.18 | 0.00 | ORB-long ORB[3503.20,3550.00] vol=2.5x ATR=12.00 |
| Stop hit — per-position SL triggered | 2025-10-13 11:05:00 | 3556.70 | 3536.67 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:55:00 | 3540.70 | 3560.71 | 0.00 | ORB-short ORB[3550.00,3573.50] vol=1.8x ATR=6.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 11:35:00 | 3530.87 | 3556.36 | 0.00 | T1 1.5R @ 3530.87 |
| Stop hit — per-position SL triggered | 2025-10-16 12:05:00 | 3540.70 | 3552.28 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:55:00 | 3519.10 | 3529.49 | 0.00 | ORB-short ORB[3529.60,3549.80] vol=2.7x ATR=7.13 |
| Stop hit — per-position SL triggered | 2025-10-17 10:05:00 | 3526.23 | 3528.59 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 3562.80 | 3548.42 | 0.00 | ORB-long ORB[3529.70,3554.50] vol=3.2x ATR=9.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 09:45:00 | 3576.65 | 3556.40 | 0.00 | T1 1.5R @ 3576.65 |
| Stop hit — per-position SL triggered | 2025-10-23 09:50:00 | 3562.80 | 3556.76 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:10:00 | 3592.80 | 3597.43 | 0.00 | ORB-short ORB[3600.50,3627.00] vol=2.3x ATR=7.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:30:00 | 3582.08 | 3596.66 | 0.00 | T1 1.5R @ 3582.08 |
| Stop hit — per-position SL triggered | 2025-10-30 12:50:00 | 3592.80 | 3591.36 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:40:00 | 3784.60 | 3748.50 | 0.00 | ORB-long ORB[3728.50,3759.80] vol=1.9x ATR=11.86 |
| Stop hit — per-position SL triggered | 2025-11-07 11:10:00 | 3772.74 | 3755.91 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:55:00 | 3781.40 | 3764.28 | 0.00 | ORB-long ORB[3745.70,3775.70] vol=5.6x ATR=8.65 |
| Stop hit — per-position SL triggered | 2025-11-10 11:05:00 | 3772.75 | 3766.06 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 11:05:00 | 3535.80 | 3558.65 | 0.00 | ORB-short ORB[3554.20,3582.00] vol=1.5x ATR=6.29 |
| Stop hit — per-position SL triggered | 2025-11-17 12:10:00 | 3542.09 | 3553.61 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:35:00 | 3511.80 | 3522.39 | 0.00 | ORB-short ORB[3513.00,3554.90] vol=2.0x ATR=7.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:00:00 | 3500.04 | 3513.99 | 0.00 | T1 1.5R @ 3500.04 |
| Target hit | 2025-11-18 15:20:00 | 3460.00 | 3480.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2025-11-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 11:10:00 | 3419.50 | 3435.65 | 0.00 | ORB-short ORB[3444.00,3463.60] vol=2.6x ATR=5.47 |
| Stop hit — per-position SL triggered | 2025-11-19 11:30:00 | 3424.97 | 3434.99 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:55:00 | 3455.00 | 3450.65 | 0.00 | ORB-long ORB[3422.70,3453.00] vol=1.9x ATR=7.03 |
| Stop hit — per-position SL triggered | 2025-11-21 10:00:00 | 3447.97 | 3451.26 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:45:00 | 3371.40 | 3384.31 | 0.00 | ORB-short ORB[3389.90,3410.50] vol=1.6x ATR=5.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:15:00 | 3362.80 | 3380.50 | 0.00 | T1 1.5R @ 3362.80 |
| Stop hit — per-position SL triggered | 2025-12-01 11:55:00 | 3371.40 | 3377.77 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:55:00 | 3393.80 | 3382.83 | 0.00 | ORB-long ORB[3370.00,3385.80] vol=2.9x ATR=4.88 |
| Stop hit — per-position SL triggered | 2025-12-04 11:10:00 | 3388.92 | 3384.32 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:00:00 | 3353.00 | 3371.65 | 0.00 | ORB-short ORB[3375.40,3396.00] vol=2.1x ATR=5.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:05:00 | 3344.42 | 3370.18 | 0.00 | T1 1.5R @ 3344.42 |
| Stop hit — per-position SL triggered | 2025-12-08 11:30:00 | 3353.00 | 3367.45 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:35:00 | 3399.90 | 3394.87 | 0.00 | ORB-long ORB[3350.00,3390.10] vol=1.7x ATR=5.81 |
| Stop hit — per-position SL triggered | 2025-12-12 10:40:00 | 3394.09 | 3395.26 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 11:05:00 | 3261.00 | 3252.78 | 0.00 | ORB-long ORB[3225.50,3248.10] vol=6.0x ATR=6.92 |
| Stop hit — per-position SL triggered | 2025-12-16 11:20:00 | 3254.08 | 3253.59 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:40:00 | 3214.60 | 3220.20 | 0.00 | ORB-short ORB[3221.00,3240.70] vol=1.7x ATR=6.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:40:00 | 3205.45 | 3216.10 | 0.00 | T1 1.5R @ 3205.45 |
| Target hit | 2025-12-30 15:20:00 | 3178.10 | 3199.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2026-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:00:00 | 3221.50 | 3227.96 | 0.00 | ORB-short ORB[3226.00,3255.00] vol=1.8x ATR=5.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:30:00 | 3213.30 | 3224.63 | 0.00 | T1 1.5R @ 3213.30 |
| Target hit | 2026-01-01 14:15:00 | 3218.10 | 3216.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — SELL (started 2026-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 09:30:00 | 3190.10 | 3202.38 | 0.00 | ORB-short ORB[3196.10,3227.20] vol=3.2x ATR=8.70 |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 3198.80 | 3198.13 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 3270.00 | 3297.21 | 0.00 | ORB-short ORB[3292.30,3332.70] vol=1.8x ATR=8.09 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 3278.09 | 3289.67 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 09:50:00 | 3216.40 | 3196.55 | 0.00 | ORB-long ORB[3180.00,3213.00] vol=2.1x ATR=10.57 |
| Stop hit — per-position SL triggered | 2026-01-12 09:55:00 | 3205.83 | 3197.87 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-01-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:55:00 | 3244.10 | 3245.31 | 0.00 | ORB-short ORB[3252.70,3273.90] vol=2.6x ATR=8.70 |
| Stop hit — per-position SL triggered | 2026-01-13 11:25:00 | 3252.80 | 3245.19 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:30:00 | 3205.90 | 3210.97 | 0.00 | ORB-short ORB[3206.00,3236.40] vol=2.0x ATR=8.30 |
| Stop hit — per-position SL triggered | 2026-01-20 10:40:00 | 3214.20 | 3210.90 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:35:00 | 3123.80 | 3139.21 | 0.00 | ORB-short ORB[3125.50,3165.50] vol=2.2x ATR=13.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:50:00 | 3103.51 | 3131.00 | 0.00 | T1 1.5R @ 3103.51 |
| Stop hit — per-position SL triggered | 2026-01-21 11:35:00 | 3123.80 | 3124.57 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-01-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 11:00:00 | 3189.40 | 3177.08 | 0.00 | ORB-long ORB[3157.20,3186.40] vol=1.8x ATR=6.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:05:00 | 3199.35 | 3178.50 | 0.00 | T1 1.5R @ 3199.35 |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 3189.40 | 3179.17 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:35:00 | 3150.30 | 3178.87 | 0.00 | ORB-short ORB[3172.50,3208.50] vol=1.7x ATR=10.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:50:00 | 3133.92 | 3166.71 | 0.00 | T1 1.5R @ 3133.92 |
| Target hit | 2026-01-29 13:50:00 | 3125.00 | 3124.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — BUY (started 2026-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:55:00 | 3177.60 | 3160.02 | 0.00 | ORB-long ORB[3130.00,3160.90] vol=1.6x ATR=6.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 11:25:00 | 3187.51 | 3167.02 | 0.00 | T1 1.5R @ 3187.51 |
| Stop hit — per-position SL triggered | 2026-01-30 13:05:00 | 3177.60 | 3178.35 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:10:00 | 3152.10 | 3138.64 | 0.00 | ORB-long ORB[3123.80,3140.70] vol=1.9x ATR=7.10 |
| Stop hit — per-position SL triggered | 2026-02-09 10:30:00 | 3145.00 | 3143.59 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 3182.20 | 3191.33 | 0.00 | ORB-short ORB[3188.00,3230.00] vol=6.8x ATR=9.76 |
| Stop hit — per-position SL triggered | 2026-02-10 11:20:00 | 3191.96 | 3189.73 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 3126.90 | 3119.40 | 0.00 | ORB-long ORB[3094.10,3114.90] vol=2.3x ATR=5.37 |
| Stop hit — per-position SL triggered | 2026-02-26 11:00:00 | 3121.53 | 3120.20 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 2909.00 | 2928.48 | 0.00 | ORB-short ORB[2931.10,2958.00] vol=1.8x ATR=8.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 2895.76 | 2925.13 | 0.00 | T1 1.5R @ 2895.76 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 2909.00 | 2913.42 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-03-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:45:00 | 2859.90 | 2868.97 | 0.00 | ORB-short ORB[2865.80,2900.00] vol=3.6x ATR=10.63 |
| Stop hit — per-position SL triggered | 2026-03-16 10:55:00 | 2870.53 | 2868.42 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-03-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:55:00 | 2907.40 | 2893.87 | 0.00 | ORB-long ORB[2882.00,2901.60] vol=1.6x ATR=10.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:10:00 | 2922.70 | 2898.04 | 0.00 | T1 1.5R @ 2922.70 |
| Target hit | 2026-03-17 13:20:00 | 2912.00 | 2914.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 77 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 2918.40 | 2901.07 | 0.00 | ORB-long ORB[2866.30,2895.10] vol=9.8x ATR=8.28 |
| Stop hit — per-position SL triggered | 2026-03-18 11:10:00 | 2910.12 | 2908.44 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 2826.90 | 2854.17 | 0.00 | ORB-short ORB[2860.20,2897.00] vol=2.3x ATR=9.19 |
| Stop hit — per-position SL triggered | 2026-03-27 11:10:00 | 2836.09 | 2851.47 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-04-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:50:00 | 2828.30 | 2856.42 | 0.00 | ORB-short ORB[2844.30,2885.40] vol=1.6x ATR=8.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 10:10:00 | 2814.98 | 2832.59 | 0.00 | T1 1.5R @ 2814.98 |
| Stop hit — per-position SL triggered | 2026-04-09 10:30:00 | 2828.30 | 2829.39 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:50:00 | 3103.00 | 3088.60 | 0.00 | ORB-long ORB[3071.10,3098.90] vol=2.9x ATR=11.18 |
| Stop hit — per-position SL triggered | 2026-04-24 09:55:00 | 3091.82 | 3090.23 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-04-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:40:00 | 3116.40 | 3104.04 | 0.00 | ORB-long ORB[3080.00,3115.40] vol=1.6x ATR=10.71 |
| Stop hit — per-position SL triggered | 2026-04-27 10:50:00 | 3105.69 | 3104.31 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:10:00 | 3138.00 | 3100.14 | 0.00 | ORB-long ORB[3072.40,3100.00] vol=4.1x ATR=11.02 |
| Stop hit — per-position SL triggered | 2026-04-28 10:55:00 | 3126.98 | 3122.36 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:45:00 | 3027.70 | 3042.59 | 0.00 | ORB-short ORB[3051.60,3090.00] vol=4.4x ATR=8.83 |
| Stop hit — per-position SL triggered | 2026-04-30 13:05:00 | 3036.53 | 3035.92 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:50:00 | 3005.00 | 3011.74 | 0.00 | ORB-short ORB[3007.00,3035.00] vol=2.6x ATR=6.33 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 3011.33 | 3011.11 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-05-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:30:00 | 3080.30 | 3061.93 | 0.00 | ORB-long ORB[3021.10,3058.40] vol=2.1x ATR=9.93 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 3070.37 | 3073.47 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-05-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:00:00 | 3115.10 | 3095.64 | 0.00 | ORB-long ORB[3072.50,3107.50] vol=2.1x ATR=11.11 |
| Stop hit — per-position SL triggered | 2026-05-07 10:35:00 | 3103.99 | 3099.70 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-15 11:10:00 | 3641.60 | 2025-05-15 11:20:00 | 3649.66 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-16 10:20:00 | 3687.50 | 2025-05-16 10:30:00 | 3668.37 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-05-16 10:20:00 | 3687.50 | 2025-05-16 10:45:00 | 3687.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-22 11:10:00 | 3642.00 | 2025-05-22 12:20:00 | 3628.16 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-22 11:10:00 | 3642.00 | 2025-05-22 13:20:00 | 3642.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-27 10:05:00 | 3715.80 | 2025-05-27 10:10:00 | 3705.04 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-28 10:05:00 | 3795.50 | 2025-05-28 10:25:00 | 3814.12 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-05-28 10:05:00 | 3795.50 | 2025-05-28 15:20:00 | 3861.30 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2025-05-30 10:55:00 | 3843.40 | 2025-05-30 11:00:00 | 3852.49 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-02 10:50:00 | 3781.20 | 2025-06-02 11:25:00 | 3763.29 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-06-02 10:50:00 | 3781.20 | 2025-06-02 11:50:00 | 3781.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-03 09:50:00 | 3878.80 | 2025-06-03 10:15:00 | 3866.97 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-04 09:55:00 | 3834.70 | 2025-06-04 10:00:00 | 3852.73 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-04 09:55:00 | 3834.70 | 2025-06-04 14:05:00 | 3960.00 | TARGET_HIT | 0.50 | 3.27% |
| BUY | retest1 | 2025-06-10 10:00:00 | 3948.70 | 2025-06-10 10:55:00 | 3940.47 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-06-23 11:10:00 | 4111.00 | 2025-06-23 14:40:00 | 4129.36 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-06-23 11:10:00 | 4111.00 | 2025-06-23 15:20:00 | 4152.70 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2025-06-25 09:40:00 | 4232.40 | 2025-06-25 09:45:00 | 4220.67 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-27 11:15:00 | 4150.00 | 2025-06-27 13:00:00 | 4142.24 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-03 09:35:00 | 4191.00 | 2025-07-03 09:45:00 | 4204.66 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-07-03 09:35:00 | 4191.00 | 2025-07-03 09:50:00 | 4191.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-07 10:15:00 | 4175.20 | 2025-07-07 11:05:00 | 4162.84 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-07 10:15:00 | 4175.20 | 2025-07-07 15:20:00 | 4161.10 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-07-09 10:50:00 | 4184.50 | 2025-07-09 11:05:00 | 4176.65 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-15 11:00:00 | 4139.70 | 2025-07-15 11:20:00 | 4131.37 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-16 10:55:00 | 4116.70 | 2025-07-16 11:25:00 | 4123.84 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-17 11:05:00 | 4118.90 | 2025-07-17 11:25:00 | 4133.41 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-07-17 11:05:00 | 4118.90 | 2025-07-17 15:20:00 | 4131.70 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2025-07-21 11:05:00 | 4155.80 | 2025-07-21 12:20:00 | 4168.41 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-07-21 11:05:00 | 4155.80 | 2025-07-21 13:50:00 | 4155.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 11:05:00 | 4026.30 | 2025-07-23 11:45:00 | 4034.81 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-28 09:45:00 | 4096.50 | 2025-07-28 09:55:00 | 4084.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-30 09:30:00 | 4286.50 | 2025-07-30 09:40:00 | 4302.30 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-30 09:30:00 | 4286.50 | 2025-07-30 09:50:00 | 4286.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 09:35:00 | 4028.60 | 2025-08-07 09:45:00 | 4041.85 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-12 10:55:00 | 3864.00 | 2025-08-12 11:10:00 | 3851.79 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-25 09:50:00 | 3914.00 | 2025-08-25 10:00:00 | 3902.95 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-29 10:45:00 | 3718.40 | 2025-08-29 11:35:00 | 3731.21 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-08-29 10:45:00 | 3718.40 | 2025-08-29 12:00:00 | 3718.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 09:45:00 | 3719.40 | 2025-09-01 11:50:00 | 3710.11 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-03 11:05:00 | 3727.30 | 2025-09-03 11:10:00 | 3734.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-04 10:15:00 | 3761.20 | 2025-09-04 10:35:00 | 3747.26 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-04 10:15:00 | 3761.20 | 2025-09-04 15:20:00 | 3745.20 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-05 10:35:00 | 3725.00 | 2025-09-05 10:45:00 | 3714.66 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-09-05 10:35:00 | 3725.00 | 2025-09-05 15:20:00 | 3661.10 | TARGET_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2025-09-08 10:40:00 | 3702.70 | 2025-09-08 11:10:00 | 3715.66 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-08 10:40:00 | 3702.70 | 2025-09-08 12:40:00 | 3702.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-09 10:50:00 | 3669.50 | 2025-09-09 11:00:00 | 3675.48 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-10 09:45:00 | 3737.80 | 2025-09-10 10:20:00 | 3728.47 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-16 11:05:00 | 3715.60 | 2025-09-16 11:10:00 | 3720.07 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-09-17 10:15:00 | 3787.60 | 2025-09-17 11:25:00 | 3801.96 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-09-17 10:15:00 | 3787.60 | 2025-09-17 11:30:00 | 3787.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-18 10:05:00 | 3715.30 | 2025-09-18 10:15:00 | 3725.10 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-23 11:10:00 | 3598.50 | 2025-09-23 11:25:00 | 3607.53 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-24 11:15:00 | 3601.00 | 2025-09-24 12:50:00 | 3608.33 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-26 10:45:00 | 3543.00 | 2025-09-26 11:00:00 | 3529.86 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-26 10:45:00 | 3543.00 | 2025-09-26 14:10:00 | 3508.30 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2025-10-03 09:35:00 | 3610.90 | 2025-10-03 09:40:00 | 3601.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-08 11:05:00 | 3557.00 | 2025-10-08 11:40:00 | 3545.37 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-08 11:05:00 | 3557.00 | 2025-10-08 15:20:00 | 3500.20 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2025-10-10 10:05:00 | 3562.70 | 2025-10-10 10:20:00 | 3555.64 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-13 11:00:00 | 3568.70 | 2025-10-13 11:05:00 | 3556.70 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-10-16 10:55:00 | 3540.70 | 2025-10-16 11:35:00 | 3530.87 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-10-16 10:55:00 | 3540.70 | 2025-10-16 12:05:00 | 3540.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 09:55:00 | 3519.10 | 2025-10-17 10:05:00 | 3526.23 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-23 09:30:00 | 3562.80 | 2025-10-23 09:45:00 | 3576.65 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-10-23 09:30:00 | 3562.80 | 2025-10-23 09:50:00 | 3562.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 10:10:00 | 3592.80 | 2025-10-30 10:30:00 | 3582.08 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-30 10:10:00 | 3592.80 | 2025-10-30 12:50:00 | 3592.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-07 10:40:00 | 3784.60 | 2025-11-07 11:10:00 | 3772.74 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-10 10:55:00 | 3781.40 | 2025-11-10 11:05:00 | 3772.75 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-17 11:05:00 | 3535.80 | 2025-11-17 12:10:00 | 3542.09 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-18 09:35:00 | 3511.80 | 2025-11-18 10:00:00 | 3500.04 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-18 09:35:00 | 3511.80 | 2025-11-18 15:20:00 | 3460.00 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2025-11-19 11:10:00 | 3419.50 | 2025-11-19 11:30:00 | 3424.97 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-11-21 09:55:00 | 3455.00 | 2025-11-21 10:00:00 | 3447.97 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-01 10:45:00 | 3371.40 | 2025-12-01 11:15:00 | 3362.80 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-01 10:45:00 | 3371.40 | 2025-12-01 11:55:00 | 3371.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-04 10:55:00 | 3393.80 | 2025-12-04 11:10:00 | 3388.92 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-12-08 11:00:00 | 3353.00 | 2025-12-08 11:05:00 | 3344.42 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-08 11:00:00 | 3353.00 | 2025-12-08 11:30:00 | 3353.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-12 10:35:00 | 3399.90 | 2025-12-12 10:40:00 | 3394.09 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-16 11:05:00 | 3261.00 | 2025-12-16 11:20:00 | 3254.08 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-30 10:40:00 | 3214.60 | 2025-12-30 11:40:00 | 3205.45 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-12-30 10:40:00 | 3214.60 | 2025-12-30 15:20:00 | 3178.10 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2026-01-01 11:00:00 | 3221.50 | 2026-01-01 11:30:00 | 3213.30 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-01-01 11:00:00 | 3221.50 | 2026-01-01 14:15:00 | 3218.10 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2026-01-02 09:30:00 | 3190.10 | 2026-01-02 10:15:00 | 3198.80 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-08 11:10:00 | 3270.00 | 2026-01-08 11:35:00 | 3278.09 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-12 09:50:00 | 3216.40 | 2026-01-12 09:55:00 | 3205.83 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-13 10:55:00 | 3244.10 | 2026-01-13 11:25:00 | 3252.80 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-20 10:30:00 | 3205.90 | 2026-01-20 10:40:00 | 3214.20 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-21 10:35:00 | 3123.80 | 2026-01-21 10:50:00 | 3103.51 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-01-21 10:35:00 | 3123.80 | 2026-01-21 11:35:00 | 3123.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-28 11:00:00 | 3189.40 | 2026-01-28 11:05:00 | 3199.35 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-01-28 11:00:00 | 3189.40 | 2026-01-28 11:15:00 | 3189.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 09:35:00 | 3150.30 | 2026-01-29 09:50:00 | 3133.92 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-01-29 09:35:00 | 3150.30 | 2026-01-29 13:50:00 | 3125.00 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2026-01-30 10:55:00 | 3177.60 | 2026-01-30 11:25:00 | 3187.51 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-01-30 10:55:00 | 3177.60 | 2026-01-30 13:05:00 | 3177.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-09 10:10:00 | 3152.10 | 2026-02-09 10:30:00 | 3145.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-10 11:00:00 | 3182.20 | 2026-02-10 11:20:00 | 3191.96 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-26 10:50:00 | 3126.90 | 2026-02-26 11:00:00 | 3121.53 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-03-13 10:00:00 | 2909.00 | 2026-03-13 10:10:00 | 2895.76 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-13 10:00:00 | 2909.00 | 2026-03-13 10:50:00 | 2909.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:45:00 | 2859.90 | 2026-03-16 10:55:00 | 2870.53 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-17 09:55:00 | 2907.40 | 2026-03-17 10:10:00 | 2922.70 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-17 09:55:00 | 2907.40 | 2026-03-17 13:20:00 | 2912.00 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-03-18 11:00:00 | 2918.40 | 2026-03-18 11:10:00 | 2910.12 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-27 11:05:00 | 2826.90 | 2026-03-27 11:10:00 | 2836.09 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-09 09:50:00 | 2828.30 | 2026-04-09 10:10:00 | 2814.98 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-09 09:50:00 | 2828.30 | 2026-04-09 10:30:00 | 2828.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 09:50:00 | 3103.00 | 2026-04-24 09:55:00 | 3091.82 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-27 10:40:00 | 3116.40 | 2026-04-27 10:50:00 | 3105.69 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-28 10:10:00 | 3138.00 | 2026-04-28 10:55:00 | 3126.98 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-30 10:45:00 | 3027.70 | 2026-04-30 13:05:00 | 3036.53 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-05 10:50:00 | 3005.00 | 2026-05-05 11:10:00 | 3011.33 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-05-06 10:30:00 | 3080.30 | 2026-05-06 11:10:00 | 3070.37 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-07 10:00:00 | 3115.10 | 2026-05-07 10:35:00 | 3103.99 | STOP_HIT | 1.00 | -0.36% |
