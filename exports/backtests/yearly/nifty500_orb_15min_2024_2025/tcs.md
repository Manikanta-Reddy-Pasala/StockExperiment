# Tata Consultancy Services Ltd. (TCS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:25:00 (36496 bars)
- **Last close:** 2474.80
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
| ENTRY1 | 59 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 11 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 48
- **Target hits / Stop hits / Partials:** 11 / 48 / 27
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 14.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 18 | 42.9% | 6 | 24 | 12 | 0.17% | 7.3% |
| BUY @ 2nd Alert (retest1) | 42 | 18 | 42.9% | 6 | 24 | 12 | 0.17% | 7.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 20 | 45.5% | 5 | 24 | 15 | 0.16% | 7.0% |
| SELL @ 2nd Alert (retest1) | 44 | 20 | 45.5% | 5 | 24 | 15 | 0.16% | 7.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 86 | 38 | 44.2% | 11 | 48 | 27 | 0.17% | 14.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 11:00:00 | 3895.30 | 3898.78 | 0.00 | ORB-short ORB[3904.50,3926.95] vol=3.2x ATR=7.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 11:25:00 | 3883.74 | 3897.90 | 0.00 | T1 1.5R @ 3883.74 |
| Stop hit — per-position SL triggered | 2024-05-15 11:50:00 | 3895.30 | 3897.35 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:00:00 | 3883.55 | 3887.15 | 0.00 | ORB-short ORB[3885.00,3909.50] vol=1.6x ATR=8.53 |
| Stop hit — per-position SL triggered | 2024-05-16 12:10:00 | 3892.08 | 3886.23 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 3852.00 | 3841.58 | 0.00 | ORB-long ORB[3825.50,3847.00] vol=1.8x ATR=5.68 |
| Stop hit — per-position SL triggered | 2024-05-23 10:00:00 | 3846.32 | 3846.23 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:05:00 | 3895.55 | 3873.36 | 0.00 | ORB-long ORB[3848.40,3871.30] vol=1.5x ATR=8.23 |
| Stop hit — per-position SL triggered | 2024-05-27 11:30:00 | 3887.32 | 3885.06 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:00:00 | 3708.20 | 3728.47 | 0.00 | ORB-short ORB[3721.20,3749.80] vol=2.0x ATR=11.29 |
| Stop hit — per-position SL triggered | 2024-05-31 11:10:00 | 3719.49 | 3719.24 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 11:00:00 | 3860.80 | 3868.63 | 0.00 | ORB-short ORB[3871.80,3905.90] vol=1.8x ATR=8.43 |
| Stop hit — per-position SL triggered | 2024-06-10 11:25:00 | 3869.23 | 3868.27 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 11:00:00 | 3886.90 | 3871.43 | 0.00 | ORB-long ORB[3852.00,3867.40] vol=1.6x ATR=6.85 |
| Stop hit — per-position SL triggered | 2024-06-13 11:25:00 | 3880.05 | 3873.06 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 11:10:00 | 3870.30 | 3839.37 | 0.00 | ORB-long ORB[3821.55,3846.70] vol=1.8x ATR=7.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 11:25:00 | 3881.53 | 3843.94 | 0.00 | T1 1.5R @ 3881.53 |
| Target hit | 2024-06-27 15:20:00 | 3938.20 | 3896.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-07-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:45:00 | 3972.95 | 3939.62 | 0.00 | ORB-long ORB[3884.00,3933.80] vol=2.3x ATR=9.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 12:00:00 | 3987.94 | 3955.59 | 0.00 | T1 1.5R @ 3987.94 |
| Stop hit — per-position SL triggered | 2024-07-01 14:05:00 | 3972.95 | 3965.30 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:40:00 | 4023.00 | 4007.25 | 0.00 | ORB-long ORB[3982.10,4019.00] vol=1.5x ATR=8.16 |
| Stop hit — per-position SL triggered | 2024-07-04 09:55:00 | 4014.84 | 4009.02 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:40:00 | 4004.70 | 4005.82 | 0.00 | ORB-short ORB[4005.00,4031.25] vol=3.0x ATR=7.66 |
| Stop hit — per-position SL triggered | 2024-07-08 10:45:00 | 4012.36 | 4005.88 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:40:00 | 3969.15 | 3985.65 | 0.00 | ORB-short ORB[3981.00,4003.30] vol=2.1x ATR=8.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:15:00 | 3956.90 | 3973.47 | 0.00 | T1 1.5R @ 3956.90 |
| Target hit | 2024-07-10 15:20:00 | 3915.75 | 3926.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2024-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:50:00 | 4183.35 | 4171.10 | 0.00 | ORB-long ORB[4144.90,4180.00] vol=4.2x ATR=9.71 |
| Stop hit — per-position SL triggered | 2024-07-16 10:35:00 | 4173.64 | 4175.37 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 10:40:00 | 4234.00 | 4202.25 | 0.00 | ORB-long ORB[4160.00,4222.00] vol=1.6x ATR=9.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 10:50:00 | 4247.88 | 4212.52 | 0.00 | T1 1.5R @ 4247.88 |
| Target hit | 2024-07-18 15:20:00 | 4301.25 | 4270.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-07-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:00:00 | 4315.95 | 4289.60 | 0.00 | ORB-long ORB[4271.05,4311.20] vol=2.6x ATR=8.32 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 4307.63 | 4292.40 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:45:00 | 4277.70 | 4290.08 | 0.00 | ORB-short ORB[4281.10,4305.75] vol=1.6x ATR=9.63 |
| Stop hit — per-position SL triggered | 2024-07-25 09:55:00 | 4287.33 | 4289.96 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:05:00 | 4400.25 | 4376.84 | 0.00 | ORB-long ORB[4337.05,4379.90] vol=4.7x ATR=10.55 |
| Stop hit — per-position SL triggered | 2024-07-26 11:25:00 | 4389.70 | 4381.91 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 11:15:00 | 4398.70 | 4399.61 | 0.00 | ORB-short ORB[4405.00,4431.00] vol=1.6x ATR=8.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 12:10:00 | 4386.13 | 4398.57 | 0.00 | T1 1.5R @ 4386.13 |
| Stop hit — per-position SL triggered | 2024-07-29 12:15:00 | 4398.70 | 4398.47 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:20:00 | 4335.40 | 4355.31 | 0.00 | ORB-short ORB[4361.70,4398.00] vol=4.3x ATR=8.00 |
| Stop hit — per-position SL triggered | 2024-07-30 10:55:00 | 4343.40 | 4351.83 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 11:05:00 | 4138.55 | 4188.94 | 0.00 | ORB-short ORB[4195.30,4239.00] vol=2.0x ATR=11.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 11:15:00 | 4120.94 | 4183.85 | 0.00 | T1 1.5R @ 4120.94 |
| Stop hit — per-position SL triggered | 2024-08-05 11:25:00 | 4138.55 | 4178.51 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:00:00 | 4242.40 | 4215.17 | 0.00 | ORB-long ORB[4170.95,4216.40] vol=2.5x ATR=15.15 |
| Stop hit — per-position SL triggered | 2024-08-06 10:20:00 | 4227.25 | 4217.62 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:35:00 | 4401.30 | 4367.91 | 0.00 | ORB-long ORB[4325.10,4378.20] vol=1.7x ATR=13.27 |
| Stop hit — per-position SL triggered | 2024-08-16 09:40:00 | 4388.03 | 4371.15 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 4559.00 | 4541.16 | 0.00 | ORB-long ORB[4505.00,4555.00] vol=1.8x ATR=11.21 |
| Stop hit — per-position SL triggered | 2024-08-20 09:40:00 | 4547.79 | 4543.29 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:50:00 | 4469.25 | 4478.21 | 0.00 | ORB-short ORB[4482.85,4506.00] vol=2.1x ATR=8.46 |
| Stop hit — per-position SL triggered | 2024-08-23 09:55:00 | 4477.71 | 4477.77 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:40:00 | 4546.00 | 4521.58 | 0.00 | ORB-long ORB[4485.00,4530.00] vol=1.7x ATR=11.83 |
| Stop hit — per-position SL triggered | 2024-08-26 09:55:00 | 4534.17 | 4525.92 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:35:00 | 4527.00 | 4484.85 | 0.00 | ORB-long ORB[4466.15,4499.00] vol=1.6x ATR=10.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 11:35:00 | 4542.45 | 4500.52 | 0.00 | T1 1.5R @ 4542.45 |
| Stop hit — per-position SL triggered | 2024-08-28 12:30:00 | 4527.00 | 4513.03 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:55:00 | 4494.00 | 4462.14 | 0.00 | ORB-long ORB[4442.45,4488.00] vol=2.1x ATR=9.02 |
| Stop hit — per-position SL triggered | 2024-09-10 11:00:00 | 4484.98 | 4463.86 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:05:00 | 4331.30 | 4378.26 | 0.00 | ORB-short ORB[4380.00,4422.20] vol=1.8x ATR=17.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:40:00 | 4305.20 | 4365.72 | 0.00 | T1 1.5R @ 4305.20 |
| Target hit | 2024-09-19 15:20:00 | 4297.50 | 4321.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2024-09-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 09:50:00 | 4284.00 | 4292.48 | 0.00 | ORB-short ORB[4286.65,4330.70] vol=1.9x ATR=11.12 |
| Stop hit — per-position SL triggered | 2024-09-20 09:55:00 | 4295.12 | 4292.33 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 4246.65 | 4267.12 | 0.00 | ORB-short ORB[4250.00,4294.45] vol=1.7x ATR=7.34 |
| Stop hit — per-position SL triggered | 2024-10-01 11:20:00 | 4253.99 | 4266.66 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:50:00 | 4245.00 | 4271.05 | 0.00 | ORB-short ORB[4262.25,4279.45] vol=1.5x ATR=10.64 |
| Stop hit — per-position SL triggered | 2024-10-07 11:25:00 | 4255.64 | 4264.21 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:25:00 | 4120.00 | 4143.61 | 0.00 | ORB-short ORB[4131.05,4169.95] vol=1.6x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 11:20:00 | 4108.22 | 4135.19 | 0.00 | T1 1.5R @ 4108.22 |
| Stop hit — per-position SL triggered | 2024-10-15 15:15:00 | 4120.00 | 4119.30 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 11:05:00 | 4124.45 | 4110.42 | 0.00 | ORB-long ORB[4096.80,4124.00] vol=1.8x ATR=8.13 |
| Stop hit — per-position SL triggered | 2024-10-17 11:30:00 | 4116.32 | 4111.72 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:35:00 | 4070.70 | 4041.55 | 0.00 | ORB-long ORB[3995.15,4038.65] vol=1.6x ATR=11.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 11:00:00 | 4087.39 | 4050.72 | 0.00 | T1 1.5R @ 4087.39 |
| Stop hit — per-position SL triggered | 2024-10-23 13:40:00 | 4070.70 | 4069.98 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-11-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:05:00 | 3951.85 | 3957.89 | 0.00 | ORB-short ORB[3952.20,3991.90] vol=1.9x ATR=9.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:30:00 | 3938.18 | 3956.07 | 0.00 | T1 1.5R @ 3938.18 |
| Stop hit — per-position SL triggered | 2024-11-04 12:00:00 | 3951.85 | 3947.56 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:40:00 | 4042.00 | 4003.47 | 0.00 | ORB-long ORB[3975.25,4009.00] vol=2.3x ATR=11.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:45:00 | 4059.00 | 4020.57 | 0.00 | T1 1.5R @ 4059.00 |
| Target hit | 2024-11-06 15:20:00 | 4140.00 | 4102.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2024-11-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 09:35:00 | 4185.15 | 4159.98 | 0.00 | ORB-long ORB[4117.65,4170.00] vol=1.6x ATR=10.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 10:00:00 | 4200.75 | 4173.69 | 0.00 | T1 1.5R @ 4200.75 |
| Target hit | 2024-11-11 14:20:00 | 4194.40 | 4194.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 4280.80 | 4307.09 | 0.00 | ORB-short ORB[4298.40,4339.95] vol=2.1x ATR=11.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:45:00 | 4264.19 | 4303.40 | 0.00 | T1 1.5R @ 4264.19 |
| Stop hit — per-position SL triggered | 2024-11-28 11:00:00 | 4280.80 | 4300.80 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:35:00 | 4352.65 | 4326.18 | 0.00 | ORB-long ORB[4289.05,4331.70] vol=2.1x ATR=10.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 09:50:00 | 4369.00 | 4338.59 | 0.00 | T1 1.5R @ 4369.00 |
| Stop hit — per-position SL triggered | 2024-12-04 11:55:00 | 4352.65 | 4356.70 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:10:00 | 4413.80 | 4428.50 | 0.00 | ORB-short ORB[4417.50,4458.00] vol=2.3x ATR=11.48 |
| Stop hit — per-position SL triggered | 2024-12-13 11:15:00 | 4425.28 | 4428.22 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 11:00:00 | 4361.40 | 4380.27 | 0.00 | ORB-short ORB[4384.00,4432.95] vol=1.6x ATR=10.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:15:00 | 4345.10 | 4377.28 | 0.00 | T1 1.5R @ 4345.10 |
| Target hit | 2024-12-17 15:20:00 | 4324.45 | 4347.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2024-12-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 10:50:00 | 4149.45 | 4170.75 | 0.00 | ORB-short ORB[4181.40,4217.00] vol=1.8x ATR=14.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 12:00:00 | 4127.98 | 4161.96 | 0.00 | T1 1.5R @ 4127.98 |
| Stop hit — per-position SL triggered | 2024-12-23 13:45:00 | 4149.45 | 4149.14 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:35:00 | 4163.00 | 4167.82 | 0.00 | ORB-short ORB[4164.65,4199.95] vol=1.6x ATR=8.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:45:00 | 4150.05 | 4166.91 | 0.00 | T1 1.5R @ 4150.05 |
| Stop hit — per-position SL triggered | 2024-12-26 11:25:00 | 4163.00 | 4161.43 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:50:00 | 4138.85 | 4121.27 | 0.00 | ORB-long ORB[4096.95,4134.10] vol=1.6x ATR=9.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:55:00 | 4152.77 | 4132.30 | 0.00 | T1 1.5R @ 4152.77 |
| Stop hit — per-position SL triggered | 2025-01-02 11:50:00 | 4138.85 | 4136.85 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 11:10:00 | 4109.80 | 4125.65 | 0.00 | ORB-short ORB[4141.00,4179.95] vol=2.0x ATR=7.72 |
| Stop hit — per-position SL triggered | 2025-01-03 11:20:00 | 4117.52 | 4125.71 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-01-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 09:30:00 | 4145.75 | 4133.15 | 0.00 | ORB-long ORB[4099.90,4138.10] vol=3.4x ATR=9.47 |
| Stop hit — per-position SL triggered | 2025-01-06 09:35:00 | 4136.28 | 4133.44 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:55:00 | 4037.85 | 4075.42 | 0.00 | ORB-short ORB[4093.15,4140.35] vol=1.5x ATR=12.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 11:45:00 | 4018.91 | 4065.45 | 0.00 | T1 1.5R @ 4018.91 |
| Stop hit — per-position SL triggered | 2025-01-07 13:30:00 | 4037.85 | 4053.76 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:50:00 | 4086.60 | 4113.61 | 0.00 | ORB-short ORB[4090.15,4126.35] vol=1.7x ATR=14.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 12:40:00 | 4064.90 | 4097.68 | 0.00 | T1 1.5R @ 4064.90 |
| Target hit | 2025-01-09 15:20:00 | 4037.00 | 4073.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2025-01-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 10:35:00 | 4122.90 | 4089.46 | 0.00 | ORB-long ORB[4044.20,4079.90] vol=1.6x ATR=10.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:45:00 | 4139.10 | 4095.55 | 0.00 | T1 1.5R @ 4139.10 |
| Stop hit — per-position SL triggered | 2025-01-22 10:55:00 | 4122.90 | 4098.26 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-02-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 11:05:00 | 4090.00 | 4113.34 | 0.00 | ORB-short ORB[4107.55,4150.00] vol=2.4x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 13:25:00 | 4078.22 | 4102.41 | 0.00 | T1 1.5R @ 4078.22 |
| Stop hit — per-position SL triggered | 2025-02-05 15:00:00 | 4090.00 | 4097.51 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:00:00 | 4090.00 | 4095.13 | 0.00 | ORB-short ORB[4097.90,4139.50] vol=2.2x ATR=7.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:55:00 | 4078.82 | 4092.57 | 0.00 | T1 1.5R @ 4078.82 |
| Target hit | 2025-02-06 15:00:00 | 4079.95 | 4079.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — BUY (started 2025-02-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-10 11:10:00 | 4047.45 | 4030.83 | 0.00 | ORB-long ORB[4014.35,4040.40] vol=2.1x ATR=7.54 |
| Stop hit — per-position SL triggered | 2025-02-10 11:15:00 | 4039.91 | 4031.32 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-02-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 10:05:00 | 3804.00 | 3797.44 | 0.00 | ORB-long ORB[3775.00,3803.80] vol=1.9x ATR=7.27 |
| Stop hit — per-position SL triggered | 2025-02-21 10:10:00 | 3796.73 | 3797.52 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 11:15:00 | 3485.10 | 3506.38 | 0.00 | ORB-short ORB[3489.00,3530.45] vol=2.3x ATR=7.16 |
| Stop hit — per-position SL triggered | 2025-03-17 12:00:00 | 3492.26 | 3502.60 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 3522.05 | 3505.28 | 0.00 | ORB-long ORB[3478.45,3513.00] vol=3.4x ATR=7.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:40:00 | 3533.59 | 3506.76 | 0.00 | T1 1.5R @ 3533.59 |
| Target hit | 2025-03-18 15:20:00 | 3552.35 | 3531.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2025-03-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:50:00 | 3677.00 | 3663.22 | 0.00 | ORB-long ORB[3640.05,3672.95] vol=2.3x ATR=8.59 |
| Stop hit — per-position SL triggered | 2025-03-26 11:00:00 | 3668.41 | 3663.99 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:55:00 | 3535.05 | 3549.78 | 0.00 | ORB-short ORB[3536.15,3584.00] vol=2.1x ATR=10.12 |
| Stop hit — per-position SL triggered | 2025-04-01 12:15:00 | 3545.17 | 3546.29 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:55:00 | 3318.40 | 3308.33 | 0.00 | ORB-long ORB[3284.10,3313.70] vol=1.8x ATR=7.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:55:00 | 3329.55 | 3312.47 | 0.00 | T1 1.5R @ 3329.55 |
| Target hit | 2025-04-21 13:40:00 | 3324.70 | 3325.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2025-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:35:00 | 3325.80 | 3312.20 | 0.00 | ORB-long ORB[3295.00,3325.00] vol=1.7x ATR=8.19 |
| Stop hit — per-position SL triggered | 2025-04-22 09:40:00 | 3317.61 | 3313.42 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 11:00:00 | 3895.30 | 2024-05-15 11:25:00 | 3883.74 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-05-15 11:00:00 | 3895.30 | 2024-05-15 11:50:00 | 3895.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 11:00:00 | 3883.55 | 2024-05-16 12:10:00 | 3892.08 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-05-23 09:35:00 | 3852.00 | 2024-05-23 10:00:00 | 3846.32 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-05-27 10:05:00 | 3895.55 | 2024-05-27 11:30:00 | 3887.32 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-05-31 10:00:00 | 3708.20 | 2024-05-31 11:10:00 | 3719.49 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-10 11:00:00 | 3860.80 | 2024-06-10 11:25:00 | 3869.23 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-06-13 11:00:00 | 3886.90 | 2024-06-13 11:25:00 | 3880.05 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-06-27 11:10:00 | 3870.30 | 2024-06-27 11:25:00 | 3881.53 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-06-27 11:10:00 | 3870.30 | 2024-06-27 15:20:00 | 3938.20 | TARGET_HIT | 0.50 | 1.75% |
| BUY | retest1 | 2024-07-01 10:45:00 | 3972.95 | 2024-07-01 12:00:00 | 3987.94 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-07-01 10:45:00 | 3972.95 | 2024-07-01 14:05:00 | 3972.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 09:40:00 | 4023.00 | 2024-07-04 09:55:00 | 4014.84 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-08 10:40:00 | 4004.70 | 2024-07-08 10:45:00 | 4012.36 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-07-10 09:40:00 | 3969.15 | 2024-07-10 10:15:00 | 3956.90 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-07-10 09:40:00 | 3969.15 | 2024-07-10 15:20:00 | 3915.75 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2024-07-16 09:50:00 | 4183.35 | 2024-07-16 10:35:00 | 4173.64 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-18 10:40:00 | 4234.00 | 2024-07-18 10:50:00 | 4247.88 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-07-18 10:40:00 | 4234.00 | 2024-07-18 15:20:00 | 4301.25 | TARGET_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2024-07-23 11:00:00 | 4315.95 | 2024-07-23 11:15:00 | 4307.63 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-07-25 09:45:00 | 4277.70 | 2024-07-25 09:55:00 | 4287.33 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-26 11:05:00 | 4400.25 | 2024-07-26 11:25:00 | 4389.70 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-07-29 11:15:00 | 4398.70 | 2024-07-29 12:10:00 | 4386.13 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-07-29 11:15:00 | 4398.70 | 2024-07-29 12:15:00 | 4398.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-30 10:20:00 | 4335.40 | 2024-07-30 10:55:00 | 4343.40 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-08-05 11:05:00 | 4138.55 | 2024-08-05 11:15:00 | 4120.94 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-05 11:05:00 | 4138.55 | 2024-08-05 11:25:00 | 4138.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-06 10:00:00 | 4242.40 | 2024-08-06 10:20:00 | 4227.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-16 09:35:00 | 4401.30 | 2024-08-16 09:40:00 | 4388.03 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-20 09:30:00 | 4559.00 | 2024-08-20 09:40:00 | 4547.79 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-23 09:50:00 | 4469.25 | 2024-08-23 09:55:00 | 4477.71 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-08-26 09:40:00 | 4546.00 | 2024-08-26 09:55:00 | 4534.17 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-28 10:35:00 | 4527.00 | 2024-08-28 11:35:00 | 4542.45 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-08-28 10:35:00 | 4527.00 | 2024-08-28 12:30:00 | 4527.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 10:55:00 | 4494.00 | 2024-09-10 11:00:00 | 4484.98 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-09-19 10:05:00 | 4331.30 | 2024-09-19 10:40:00 | 4305.20 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-09-19 10:05:00 | 4331.30 | 2024-09-19 15:20:00 | 4297.50 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2024-09-20 09:50:00 | 4284.00 | 2024-09-20 09:55:00 | 4295.12 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-01 11:10:00 | 4246.65 | 2024-10-01 11:20:00 | 4253.99 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-10-07 10:50:00 | 4245.00 | 2024-10-07 11:25:00 | 4255.64 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-15 10:25:00 | 4120.00 | 2024-10-15 11:20:00 | 4108.22 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-10-15 10:25:00 | 4120.00 | 2024-10-15 15:15:00 | 4120.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-17 11:05:00 | 4124.45 | 2024-10-17 11:30:00 | 4116.32 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-10-23 10:35:00 | 4070.70 | 2024-10-23 11:00:00 | 4087.39 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-10-23 10:35:00 | 4070.70 | 2024-10-23 13:40:00 | 4070.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-04 10:05:00 | 3951.85 | 2024-11-04 10:30:00 | 3938.18 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-11-04 10:05:00 | 3951.85 | 2024-11-04 12:00:00 | 3951.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-06 09:40:00 | 4042.00 | 2024-11-06 09:45:00 | 4059.00 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-11-06 09:40:00 | 4042.00 | 2024-11-06 15:20:00 | 4140.00 | TARGET_HIT | 0.50 | 2.42% |
| BUY | retest1 | 2024-11-11 09:35:00 | 4185.15 | 2024-11-11 10:00:00 | 4200.75 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-11-11 09:35:00 | 4185.15 | 2024-11-11 14:20:00 | 4194.40 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2024-11-28 10:35:00 | 4280.80 | 2024-11-28 10:45:00 | 4264.19 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-11-28 10:35:00 | 4280.80 | 2024-11-28 11:00:00 | 4280.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 09:35:00 | 4352.65 | 2024-12-04 09:50:00 | 4369.00 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-04 09:35:00 | 4352.65 | 2024-12-04 11:55:00 | 4352.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 11:10:00 | 4413.80 | 2024-12-13 11:15:00 | 4425.28 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-17 11:00:00 | 4361.40 | 2024-12-17 11:15:00 | 4345.10 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-17 11:00:00 | 4361.40 | 2024-12-17 15:20:00 | 4324.45 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2024-12-23 10:50:00 | 4149.45 | 2024-12-23 12:00:00 | 4127.98 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-12-23 10:50:00 | 4149.45 | 2024-12-23 13:45:00 | 4149.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 10:35:00 | 4163.00 | 2024-12-26 10:45:00 | 4150.05 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-26 10:35:00 | 4163.00 | 2024-12-26 11:25:00 | 4163.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 09:50:00 | 4138.85 | 2025-01-02 10:55:00 | 4152.77 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-01-02 09:50:00 | 4138.85 | 2025-01-02 11:50:00 | 4138.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 11:10:00 | 4109.80 | 2025-01-03 11:20:00 | 4117.52 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-01-06 09:30:00 | 4145.75 | 2025-01-06 09:35:00 | 4136.28 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-07 10:55:00 | 4037.85 | 2025-01-07 11:45:00 | 4018.91 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-07 10:55:00 | 4037.85 | 2025-01-07 13:30:00 | 4037.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-09 10:50:00 | 4086.60 | 2025-01-09 12:40:00 | 4064.90 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-01-09 10:50:00 | 4086.60 | 2025-01-09 15:20:00 | 4037.00 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2025-01-22 10:35:00 | 4122.90 | 2025-01-22 10:45:00 | 4139.10 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-01-22 10:35:00 | 4122.90 | 2025-01-22 10:55:00 | 4122.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-05 11:05:00 | 4090.00 | 2025-02-05 13:25:00 | 4078.22 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-02-05 11:05:00 | 4090.00 | 2025-02-05 15:00:00 | 4090.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-06 11:00:00 | 4090.00 | 2025-02-06 11:55:00 | 4078.82 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-02-06 11:00:00 | 4090.00 | 2025-02-06 15:00:00 | 4079.95 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-02-10 11:10:00 | 4047.45 | 2025-02-10 11:15:00 | 4039.91 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-02-21 10:05:00 | 3804.00 | 2025-02-21 10:10:00 | 3796.73 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-03-17 11:15:00 | 3485.10 | 2025-03-17 12:00:00 | 3492.26 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-03-18 09:35:00 | 3522.05 | 2025-03-18 09:40:00 | 3533.59 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-03-18 09:35:00 | 3522.05 | 2025-03-18 15:20:00 | 3552.35 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2025-03-26 10:50:00 | 3677.00 | 2025-03-26 11:00:00 | 3668.41 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-04-01 10:55:00 | 3535.05 | 2025-04-01 12:15:00 | 3545.17 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-21 10:55:00 | 3318.40 | 2025-04-21 11:55:00 | 3329.55 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-04-21 10:55:00 | 3318.40 | 2025-04-21 13:40:00 | 3324.70 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-04-22 09:35:00 | 3325.80 | 2025-04-22 09:40:00 | 3317.61 | STOP_HIT | 1.00 | -0.25% |
