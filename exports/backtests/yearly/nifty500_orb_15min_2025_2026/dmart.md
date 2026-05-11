# Avenue Supermarts Ltd. (DMART)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16963 bars)
- **Last close:** 4396.10
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
| ENTRY1 | 91 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 15 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 125 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 49 / 76
- **Target hits / Stop hits / Partials:** 15 / 76 / 34
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 6.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 19 | 30.6% | 6 | 43 | 13 | -0.01% | -0.6% |
| BUY @ 2nd Alert (retest1) | 62 | 19 | 30.6% | 6 | 43 | 13 | -0.01% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 63 | 30 | 47.6% | 9 | 33 | 21 | 0.11% | 7.1% |
| SELL @ 2nd Alert (retest1) | 63 | 30 | 47.6% | 9 | 33 | 21 | 0.11% | 7.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 125 | 49 | 39.2% | 15 | 76 | 34 | 0.05% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:35:00 | 4094.10 | 4068.10 | 0.00 | ORB-long ORB[4033.00,4074.00] vol=2.4x ATR=12.46 |
| Stop hit — per-position SL triggered | 2025-05-13 09:40:00 | 4081.64 | 4071.78 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:35:00 | 4130.00 | 4105.16 | 0.00 | ORB-long ORB[4042.50,4101.80] vol=2.9x ATR=12.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 11:15:00 | 4148.50 | 4117.11 | 0.00 | T1 1.5R @ 4148.50 |
| Target hit | 2025-05-16 15:20:00 | 4187.50 | 4152.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2025-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:30:00 | 4098.00 | 4113.27 | 0.00 | ORB-short ORB[4101.00,4149.50] vol=2.6x ATR=10.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 09:40:00 | 4082.77 | 4104.64 | 0.00 | T1 1.5R @ 4082.77 |
| Stop hit — per-position SL triggered | 2025-05-27 10:10:00 | 4098.00 | 4098.01 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:05:00 | 4040.20 | 4050.15 | 0.00 | ORB-short ORB[4046.40,4070.00] vol=2.1x ATR=5.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 11:15:00 | 4031.52 | 4049.30 | 0.00 | T1 1.5R @ 4031.52 |
| Stop hit — per-position SL triggered | 2025-05-29 11:30:00 | 4040.20 | 4048.01 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 4162.90 | 4125.49 | 0.00 | ORB-long ORB[4074.50,4134.00] vol=3.1x ATR=13.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:40:00 | 4182.53 | 4149.04 | 0.00 | T1 1.5R @ 4182.53 |
| Target hit | 2025-06-05 10:20:00 | 4188.40 | 4191.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2025-06-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:45:00 | 4204.00 | 4183.56 | 0.00 | ORB-long ORB[4143.50,4186.70] vol=3.4x ATR=12.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 09:50:00 | 4222.27 | 4189.20 | 0.00 | T1 1.5R @ 4222.27 |
| Stop hit — per-position SL triggered | 2025-06-06 10:00:00 | 4204.00 | 4197.15 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 4195.00 | 4209.62 | 0.00 | ORB-short ORB[4201.10,4227.10] vol=1.9x ATR=9.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:35:00 | 4181.34 | 4205.01 | 0.00 | T1 1.5R @ 4181.34 |
| Stop hit — per-position SL triggered | 2025-06-09 10:25:00 | 4195.00 | 4195.23 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 11:00:00 | 4115.30 | 4104.47 | 0.00 | ORB-long ORB[4080.00,4099.90] vol=2.9x ATR=10.64 |
| Stop hit — per-position SL triggered | 2025-06-11 11:05:00 | 4104.66 | 4104.46 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 10:50:00 | 4299.80 | 4287.60 | 0.00 | ORB-long ORB[4255.40,4297.80] vol=2.6x ATR=11.16 |
| Stop hit — per-position SL triggered | 2025-06-23 11:10:00 | 4288.64 | 4289.51 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 11:10:00 | 4308.90 | 4313.19 | 0.00 | ORB-short ORB[4313.10,4356.70] vol=2.0x ATR=10.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 11:40:00 | 4292.55 | 4312.14 | 0.00 | T1 1.5R @ 4292.55 |
| Stop hit — per-position SL triggered | 2025-06-24 11:50:00 | 4308.90 | 4311.80 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:40:00 | 4426.60 | 4407.34 | 0.00 | ORB-long ORB[4381.10,4419.00] vol=1.5x ATR=13.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:45:00 | 4447.21 | 4418.23 | 0.00 | T1 1.5R @ 4447.21 |
| Target hit | 2025-07-01 10:25:00 | 4447.00 | 4447.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2025-07-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:50:00 | 4239.10 | 4226.38 | 0.00 | ORB-long ORB[4203.10,4232.80] vol=1.6x ATR=7.48 |
| Stop hit — per-position SL triggered | 2025-07-09 10:55:00 | 4231.62 | 4226.72 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:35:00 | 4118.70 | 4152.23 | 0.00 | ORB-short ORB[4135.00,4169.00] vol=2.6x ATR=12.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:45:00 | 4099.79 | 4133.00 | 0.00 | T1 1.5R @ 4099.79 |
| Target hit | 2025-07-11 13:45:00 | 4101.00 | 4091.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — BUY (started 2025-07-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:45:00 | 4062.00 | 4041.54 | 0.00 | ORB-long ORB[4012.50,4049.40] vol=1.7x ATR=10.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 09:55:00 | 4076.99 | 4049.32 | 0.00 | T1 1.5R @ 4076.99 |
| Stop hit — per-position SL triggered | 2025-07-16 10:40:00 | 4062.00 | 4055.07 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:15:00 | 4075.20 | 4082.11 | 0.00 | ORB-short ORB[4075.50,4100.00] vol=2.1x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 12:15:00 | 4065.62 | 4080.56 | 0.00 | T1 1.5R @ 4065.62 |
| Target hit | 2025-07-17 15:20:00 | 4052.50 | 4067.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-07-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:50:00 | 4044.90 | 4046.79 | 0.00 | ORB-short ORB[4045.90,4068.90] vol=2.8x ATR=8.21 |
| Stop hit — per-position SL triggered | 2025-07-18 11:30:00 | 4053.11 | 4046.72 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:00:00 | 4001.20 | 4004.07 | 0.00 | ORB-short ORB[4005.00,4027.10] vol=1.5x ATR=8.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 3988.92 | 4001.47 | 0.00 | T1 1.5R @ 3988.92 |
| Target hit | 2025-07-25 14:30:00 | 3985.50 | 3985.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — BUY (started 2025-07-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:55:00 | 4007.20 | 3980.21 | 0.00 | ORB-long ORB[3955.80,3997.00] vol=1.9x ATR=10.79 |
| Stop hit — per-position SL triggered | 2025-07-28 11:30:00 | 3996.41 | 3985.40 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:35:00 | 3945.70 | 3972.20 | 0.00 | ORB-short ORB[3967.30,3995.00] vol=1.8x ATR=10.23 |
| Stop hit — per-position SL triggered | 2025-07-29 12:00:00 | 3955.93 | 3963.59 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 10:50:00 | 4254.00 | 4229.99 | 0.00 | ORB-long ORB[4188.10,4241.80] vol=1.9x ATR=12.50 |
| Stop hit — per-position SL triggered | 2025-08-05 11:10:00 | 4241.50 | 4232.58 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 09:30:00 | 4247.60 | 4237.77 | 0.00 | ORB-long ORB[4207.10,4244.20] vol=2.6x ATR=10.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:40:00 | 4264.05 | 4243.48 | 0.00 | T1 1.5R @ 4264.05 |
| Target hit | 2025-08-07 11:30:00 | 4262.00 | 4263.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2025-08-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:00:00 | 4218.70 | 4185.62 | 0.00 | ORB-long ORB[4150.10,4193.70] vol=2.4x ATR=10.73 |
| Stop hit — per-position SL triggered | 2025-08-11 11:20:00 | 4207.97 | 4190.18 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:55:00 | 4242.00 | 4217.87 | 0.00 | ORB-long ORB[4184.50,4210.00] vol=2.0x ATR=9.73 |
| Stop hit — per-position SL triggered | 2025-08-12 10:10:00 | 4232.27 | 4224.82 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 09:50:00 | 4390.00 | 4356.64 | 0.00 | ORB-long ORB[4327.10,4359.10] vol=2.4x ATR=11.85 |
| Stop hit — per-position SL triggered | 2025-08-14 09:55:00 | 4378.15 | 4359.27 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 4708.60 | 4680.98 | 0.00 | ORB-long ORB[4630.10,4692.90] vol=1.8x ATR=13.26 |
| Stop hit — per-position SL triggered | 2025-08-20 09:40:00 | 4695.34 | 4684.60 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:55:00 | 4664.90 | 4681.77 | 0.00 | ORB-short ORB[4671.50,4710.00] vol=2.0x ATR=10.50 |
| Stop hit — per-position SL triggered | 2025-08-22 10:25:00 | 4675.40 | 4678.87 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:30:00 | 4747.10 | 4723.77 | 0.00 | ORB-long ORB[4700.10,4732.70] vol=1.8x ATR=11.06 |
| Stop hit — per-position SL triggered | 2025-08-25 10:35:00 | 4736.04 | 4724.40 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:45:00 | 4794.00 | 4773.67 | 0.00 | ORB-long ORB[4732.10,4788.00] vol=2.5x ATR=12.10 |
| Stop hit — per-position SL triggered | 2025-09-01 11:45:00 | 4781.90 | 4781.13 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:50:00 | 4722.30 | 4705.37 | 0.00 | ORB-long ORB[4675.50,4706.70] vol=2.0x ATR=10.60 |
| Stop hit — per-position SL triggered | 2025-09-03 09:55:00 | 4711.70 | 4706.46 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 10:10:00 | 4768.60 | 4729.70 | 0.00 | ORB-long ORB[4705.60,4733.80] vol=1.7x ATR=13.66 |
| Stop hit — per-position SL triggered | 2025-09-09 11:35:00 | 4754.94 | 4743.41 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 09:45:00 | 4757.60 | 4769.96 | 0.00 | ORB-short ORB[4759.90,4789.80] vol=2.4x ATR=10.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 09:55:00 | 4741.58 | 4765.79 | 0.00 | T1 1.5R @ 4741.58 |
| Stop hit — per-position SL triggered | 2025-09-10 10:10:00 | 4757.60 | 4763.63 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 09:40:00 | 4596.00 | 4621.78 | 0.00 | ORB-short ORB[4613.10,4646.00] vol=1.6x ATR=10.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 10:15:00 | 4580.09 | 4608.13 | 0.00 | T1 1.5R @ 4580.09 |
| Target hit | 2025-09-11 12:20:00 | 4591.90 | 4591.68 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — SELL (started 2025-09-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 10:10:00 | 4588.50 | 4604.79 | 0.00 | ORB-short ORB[4601.60,4637.80] vol=1.6x ATR=12.17 |
| Stop hit — per-position SL triggered | 2025-09-15 10:25:00 | 4600.67 | 4603.32 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:25:00 | 4699.30 | 4685.60 | 0.00 | ORB-long ORB[4655.00,4677.30] vol=1.9x ATR=10.12 |
| Stop hit — per-position SL triggered | 2025-09-16 10:30:00 | 4689.18 | 4685.73 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 11:10:00 | 4748.20 | 4732.86 | 0.00 | ORB-long ORB[4701.10,4739.00] vol=1.7x ATR=7.67 |
| Stop hit — per-position SL triggered | 2025-09-17 11:40:00 | 4740.53 | 4735.90 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 11:10:00 | 4639.50 | 4636.54 | 0.00 | ORB-long ORB[4606.40,4630.00] vol=1.7x ATR=9.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 11:35:00 | 4654.37 | 4638.20 | 0.00 | T1 1.5R @ 4654.37 |
| Target hit | 2025-09-24 14:15:00 | 4681.60 | 4684.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2025-09-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:25:00 | 4560.40 | 4581.44 | 0.00 | ORB-short ORB[4571.70,4629.30] vol=1.6x ATR=14.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:40:00 | 4538.74 | 4576.27 | 0.00 | T1 1.5R @ 4538.74 |
| Stop hit — per-position SL triggered | 2025-09-26 10:45:00 | 4560.40 | 4575.03 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:00:00 | 4546.30 | 4528.62 | 0.00 | ORB-long ORB[4483.80,4540.20] vol=2.4x ATR=15.83 |
| Stop hit — per-position SL triggered | 2025-09-29 10:30:00 | 4530.47 | 4531.43 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:40:00 | 4491.40 | 4529.45 | 0.00 | ORB-short ORB[4512.30,4566.90] vol=1.8x ATR=12.42 |
| Stop hit — per-position SL triggered | 2025-09-30 10:55:00 | 4503.82 | 4525.11 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:50:00 | 4300.30 | 4314.56 | 0.00 | ORB-short ORB[4302.00,4338.20] vol=7.2x ATR=9.63 |
| Stop hit — per-position SL triggered | 2025-10-07 12:05:00 | 4309.93 | 4311.73 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:50:00 | 4268.00 | 4249.41 | 0.00 | ORB-long ORB[4215.00,4250.00] vol=1.7x ATR=8.54 |
| Stop hit — per-position SL triggered | 2025-10-15 11:00:00 | 4259.46 | 4249.99 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 11:15:00 | 4296.40 | 4285.53 | 0.00 | ORB-long ORB[4267.40,4295.00] vol=3.0x ATR=7.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 11:50:00 | 4307.64 | 4288.70 | 0.00 | T1 1.5R @ 4307.64 |
| Target hit | 2025-10-16 15:10:00 | 4307.00 | 4308.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 4332.10 | 4305.38 | 0.00 | ORB-long ORB[4268.60,4309.80] vol=2.7x ATR=10.83 |
| Stop hit — per-position SL triggered | 2025-10-23 09:35:00 | 4321.27 | 4307.33 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:10:00 | 4228.00 | 4249.16 | 0.00 | ORB-short ORB[4243.70,4271.70] vol=1.6x ATR=6.80 |
| Stop hit — per-position SL triggered | 2025-10-24 11:35:00 | 4234.80 | 4244.89 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:30:00 | 4194.00 | 4208.68 | 0.00 | ORB-short ORB[4207.60,4248.00] vol=2.7x ATR=7.23 |
| Stop hit — per-position SL triggered | 2025-10-30 10:40:00 | 4201.23 | 4207.89 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:35:00 | 4193.00 | 4174.46 | 0.00 | ORB-long ORB[4145.00,4167.50] vol=1.6x ATR=7.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:35:00 | 4204.15 | 4190.94 | 0.00 | T1 1.5R @ 4204.15 |
| Stop hit — per-position SL triggered | 2025-11-04 12:30:00 | 4193.00 | 4194.60 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 11:00:00 | 4137.90 | 4149.58 | 0.00 | ORB-short ORB[4162.40,4191.40] vol=5.4x ATR=8.63 |
| Stop hit — per-position SL triggered | 2025-11-06 11:10:00 | 4146.53 | 4148.55 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:50:00 | 4051.00 | 4037.32 | 0.00 | ORB-long ORB[4011.00,4041.90] vol=3.9x ATR=9.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:15:00 | 4065.43 | 4044.96 | 0.00 | T1 1.5R @ 4065.43 |
| Stop hit — per-position SL triggered | 2025-11-10 11:40:00 | 4051.00 | 4047.98 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:55:00 | 4049.50 | 4038.17 | 0.00 | ORB-long ORB[4020.00,4047.30] vol=1.8x ATR=7.08 |
| Stop hit — per-position SL triggered | 2025-11-11 11:00:00 | 4042.42 | 4038.50 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:25:00 | 4060.70 | 4070.59 | 0.00 | ORB-short ORB[4070.50,4112.60] vol=1.5x ATR=6.37 |
| Stop hit — per-position SL triggered | 2025-11-12 10:55:00 | 4067.07 | 4069.47 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:45:00 | 4080.00 | 4065.87 | 0.00 | ORB-long ORB[4040.40,4068.60] vol=4.1x ATR=7.68 |
| Stop hit — per-position SL triggered | 2025-11-13 13:20:00 | 4072.32 | 4073.85 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 09:30:00 | 4049.80 | 4060.69 | 0.00 | ORB-short ORB[4050.10,4086.00] vol=1.8x ATR=8.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 10:50:00 | 4036.80 | 4053.58 | 0.00 | T1 1.5R @ 4036.80 |
| Target hit | 2025-11-17 15:20:00 | 4036.90 | 4044.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2025-11-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:40:00 | 4005.00 | 4014.70 | 0.00 | ORB-short ORB[4015.30,4042.20] vol=4.5x ATR=5.59 |
| Stop hit — per-position SL triggered | 2025-11-18 09:45:00 | 4010.59 | 4014.19 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:10:00 | 4058.20 | 4070.06 | 0.00 | ORB-short ORB[4061.90,4090.00] vol=1.6x ATR=9.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:45:00 | 4043.34 | 4061.91 | 0.00 | T1 1.5R @ 4043.34 |
| Target hit | 2025-11-21 15:20:00 | 4035.70 | 4057.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-11-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:55:00 | 4000.10 | 4011.16 | 0.00 | ORB-short ORB[4011.20,4048.50] vol=1.8x ATR=6.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 12:50:00 | 3989.73 | 4000.45 | 0.00 | T1 1.5R @ 3989.73 |
| Target hit | 2025-11-24 14:50:00 | 3997.30 | 3996.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 56 — BUY (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 4042.40 | 4029.19 | 0.00 | ORB-long ORB[4000.00,4034.80] vol=1.6x ATR=7.77 |
| Stop hit — per-position SL triggered | 2025-11-28 09:35:00 | 4034.63 | 4029.22 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:45:00 | 3967.40 | 3984.90 | 0.00 | ORB-short ORB[3985.00,4009.50] vol=2.3x ATR=6.98 |
| Stop hit — per-position SL triggered | 2025-12-01 09:50:00 | 3974.38 | 3982.31 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:10:00 | 3899.60 | 3921.02 | 0.00 | ORB-short ORB[3934.20,3969.20] vol=2.3x ATR=7.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:20:00 | 3887.67 | 3918.28 | 0.00 | T1 1.5R @ 3887.67 |
| Stop hit — per-position SL triggered | 2025-12-03 12:00:00 | 3899.60 | 3912.13 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:35:00 | 3920.30 | 3912.34 | 0.00 | ORB-long ORB[3897.30,3917.80] vol=1.9x ATR=7.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 11:25:00 | 3931.80 | 3916.25 | 0.00 | T1 1.5R @ 3931.80 |
| Stop hit — per-position SL triggered | 2025-12-04 11:35:00 | 3920.30 | 3916.66 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:45:00 | 3901.20 | 3924.24 | 0.00 | ORB-short ORB[3932.10,3955.20] vol=1.5x ATR=7.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 3890.66 | 3912.87 | 0.00 | T1 1.5R @ 3890.66 |
| Stop hit — per-position SL triggered | 2025-12-08 12:25:00 | 3901.20 | 3911.66 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:50:00 | 3850.00 | 3882.92 | 0.00 | ORB-short ORB[3890.00,3923.70] vol=1.6x ATR=8.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:35:00 | 3837.28 | 3868.60 | 0.00 | T1 1.5R @ 3837.28 |
| Target hit | 2025-12-10 15:20:00 | 3815.60 | 3828.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-12-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:00:00 | 3857.00 | 3844.79 | 0.00 | ORB-long ORB[3812.50,3839.30] vol=1.5x ATR=8.92 |
| Stop hit — per-position SL triggered | 2025-12-12 10:20:00 | 3848.08 | 3847.04 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 10:15:00 | 3913.00 | 3890.48 | 0.00 | ORB-long ORB[3867.00,3899.00] vol=1.7x ATR=12.44 |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 3900.56 | 3894.42 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 11:10:00 | 3778.30 | 3795.64 | 0.00 | ORB-short ORB[3790.00,3822.30] vol=3.5x ATR=6.71 |
| Stop hit — per-position SL triggered | 2025-12-18 11:20:00 | 3785.01 | 3794.91 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 09:30:00 | 3851.00 | 3841.02 | 0.00 | ORB-long ORB[3818.00,3848.00] vol=1.6x ATR=8.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 10:05:00 | 3863.45 | 3846.69 | 0.00 | T1 1.5R @ 3863.45 |
| Stop hit — per-position SL triggered | 2025-12-22 10:30:00 | 3851.00 | 3847.82 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 11:10:00 | 3839.70 | 3823.37 | 0.00 | ORB-long ORB[3808.00,3835.00] vol=1.6x ATR=7.32 |
| Stop hit — per-position SL triggered | 2025-12-23 11:50:00 | 3832.38 | 3825.66 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 11:05:00 | 3813.80 | 3826.15 | 0.00 | ORB-short ORB[3816.60,3843.00] vol=1.9x ATR=5.34 |
| Stop hit — per-position SL triggered | 2025-12-24 11:20:00 | 3819.14 | 3821.08 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:05:00 | 3797.10 | 3803.09 | 0.00 | ORB-short ORB[3798.00,3824.00] vol=3.7x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:35:00 | 3787.73 | 3798.07 | 0.00 | T1 1.5R @ 3787.73 |
| Stop hit — per-position SL triggered | 2025-12-26 12:00:00 | 3797.10 | 3796.69 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-12-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:05:00 | 3779.60 | 3785.35 | 0.00 | ORB-short ORB[3781.80,3804.00] vol=11.7x ATR=6.38 |
| Stop hit — per-position SL triggered | 2025-12-30 12:00:00 | 3785.98 | 3784.31 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:30:00 | 3775.50 | 3759.26 | 0.00 | ORB-long ORB[3751.00,3774.00] vol=2.0x ATR=8.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 10:35:00 | 3788.37 | 3764.38 | 0.00 | T1 1.5R @ 3788.37 |
| Stop hit — per-position SL triggered | 2025-12-31 11:35:00 | 3775.50 | 3776.54 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:05:00 | 3746.40 | 3737.92 | 0.00 | ORB-long ORB[3714.20,3739.60] vol=2.6x ATR=9.50 |
| Stop hit — per-position SL triggered | 2026-01-02 11:30:00 | 3736.90 | 3740.19 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:50:00 | 3652.20 | 3626.65 | 0.00 | ORB-long ORB[3611.00,3641.00] vol=3.3x ATR=8.73 |
| Stop hit — per-position SL triggered | 2026-01-06 11:20:00 | 3643.47 | 3630.40 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-01-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:10:00 | 3822.20 | 3831.48 | 0.00 | ORB-short ORB[3835.00,3889.00] vol=4.3x ATR=12.25 |
| Stop hit — per-position SL triggered | 2026-01-13 11:35:00 | 3834.45 | 3831.07 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:50:00 | 3707.80 | 3720.07 | 0.00 | ORB-short ORB[3746.50,3786.50] vol=8.7x ATR=9.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:00:00 | 3693.05 | 3717.31 | 0.00 | T1 1.5R @ 3693.05 |
| Target hit | 2026-01-20 15:20:00 | 3661.40 | 3695.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2026-01-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 11:05:00 | 3707.00 | 3696.78 | 0.00 | ORB-long ORB[3663.00,3700.00] vol=2.6x ATR=7.38 |
| Stop hit — per-position SL triggered | 2026-01-28 11:35:00 | 3699.62 | 3698.18 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-01-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:25:00 | 3679.30 | 3697.99 | 0.00 | ORB-short ORB[3703.70,3727.20] vol=1.6x ATR=9.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:40:00 | 3664.58 | 3693.66 | 0.00 | T1 1.5R @ 3664.58 |
| Stop hit — per-position SL triggered | 2026-01-29 11:20:00 | 3679.30 | 3688.69 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:40:00 | 3704.30 | 3689.84 | 0.00 | ORB-long ORB[3670.00,3695.20] vol=2.2x ATR=8.38 |
| Stop hit — per-position SL triggered | 2026-02-01 12:00:00 | 3695.92 | 3697.36 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-02-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:45:00 | 3862.30 | 3892.54 | 0.00 | ORB-short ORB[3900.00,3949.00] vol=2.2x ATR=9.74 |
| Stop hit — per-position SL triggered | 2026-02-06 11:10:00 | 3872.04 | 3888.04 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 3964.50 | 3940.39 | 0.00 | ORB-long ORB[3920.00,3956.00] vol=1.6x ATR=9.49 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 3955.01 | 3944.13 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 3998.20 | 3999.79 | 0.00 | ORB-short ORB[4005.70,4032.20] vol=2.8x ATR=8.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:10:00 | 3985.66 | 3999.42 | 0.00 | T1 1.5R @ 3985.66 |
| Stop hit — per-position SL triggered | 2026-02-11 11:35:00 | 3998.20 | 3999.02 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 3871.00 | 3915.53 | 0.00 | ORB-short ORB[3908.00,3944.00] vol=2.0x ATR=8.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:50:00 | 3857.70 | 3909.17 | 0.00 | T1 1.5R @ 3857.70 |
| Stop hit — per-position SL triggered | 2026-02-19 13:20:00 | 3871.00 | 3900.49 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 3851.10 | 3860.86 | 0.00 | ORB-short ORB[3871.20,3900.00] vol=2.6x ATR=10.08 |
| Stop hit — per-position SL triggered | 2026-02-23 11:30:00 | 3861.18 | 3858.15 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 3808.30 | 3823.69 | 0.00 | ORB-short ORB[3816.00,3847.10] vol=2.4x ATR=8.57 |
| Stop hit — per-position SL triggered | 2026-02-24 11:55:00 | 3816.87 | 3820.73 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-02-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:10:00 | 3940.90 | 3904.91 | 0.00 | ORB-long ORB[3869.50,3906.40] vol=3.7x ATR=10.08 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 3930.82 | 3906.92 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:05:00 | 3814.90 | 3824.12 | 0.00 | ORB-short ORB[3840.80,3880.20] vol=3.4x ATR=9.37 |
| Stop hit — per-position SL triggered | 2026-02-27 10:30:00 | 3824.27 | 3822.86 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 3749.00 | 3750.50 | 0.00 | ORB-short ORB[3754.60,3790.50] vol=2.8x ATR=8.36 |
| Stop hit — per-position SL triggered | 2026-03-05 11:05:00 | 3757.36 | 3751.43 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-03-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 09:30:00 | 3897.00 | 3861.30 | 0.00 | ORB-long ORB[3814.50,3868.00] vol=3.2x ATR=16.52 |
| Stop hit — per-position SL triggered | 2026-03-09 09:45:00 | 3880.48 | 3869.80 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 3846.90 | 3826.43 | 0.00 | ORB-long ORB[3810.00,3842.00] vol=1.8x ATR=11.40 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 3835.50 | 3830.38 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:45:00 | 3839.10 | 3816.96 | 0.00 | ORB-long ORB[3802.20,3829.00] vol=1.6x ATR=11.82 |
| Stop hit — per-position SL triggered | 2026-03-20 11:50:00 | 3827.28 | 3821.93 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-03-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 10:00:00 | 3925.80 | 3918.28 | 0.00 | ORB-long ORB[3865.50,3915.90] vol=4.5x ATR=13.51 |
| Stop hit — per-position SL triggered | 2026-03-27 10:15:00 | 3912.29 | 3918.01 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2026-03-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 09:35:00 | 3922.80 | 3912.71 | 0.00 | ORB-long ORB[3877.00,3922.50] vol=2.6x ATR=15.85 |
| Stop hit — per-position SL triggered | 2026-03-30 09:40:00 | 3906.95 | 3912.89 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:35:00 | 4094.10 | 2025-05-13 09:40:00 | 4081.64 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-16 10:35:00 | 4130.00 | 2025-05-16 11:15:00 | 4148.50 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-16 10:35:00 | 4130.00 | 2025-05-16 15:20:00 | 4187.50 | TARGET_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2025-05-27 09:30:00 | 4098.00 | 2025-05-27 09:40:00 | 4082.77 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-05-27 09:30:00 | 4098.00 | 2025-05-27 10:10:00 | 4098.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-29 11:05:00 | 4040.20 | 2025-05-29 11:15:00 | 4031.52 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2025-05-29 11:05:00 | 4040.20 | 2025-05-29 11:30:00 | 4040.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 09:30:00 | 4162.90 | 2025-06-05 09:40:00 | 4182.53 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-05 09:30:00 | 4162.90 | 2025-06-05 10:20:00 | 4188.40 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2025-06-06 09:45:00 | 4204.00 | 2025-06-06 09:50:00 | 4222.27 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-06-06 09:45:00 | 4204.00 | 2025-06-06 10:00:00 | 4204.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-09 09:30:00 | 4195.00 | 2025-06-09 09:35:00 | 4181.34 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-06-09 09:30:00 | 4195.00 | 2025-06-09 10:25:00 | 4195.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-11 11:00:00 | 4115.30 | 2025-06-11 11:05:00 | 4104.66 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-23 10:50:00 | 4299.80 | 2025-06-23 11:10:00 | 4288.64 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-24 11:10:00 | 4308.90 | 2025-06-24 11:40:00 | 4292.55 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-06-24 11:10:00 | 4308.90 | 2025-06-24 11:50:00 | 4308.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-01 09:40:00 | 4426.60 | 2025-07-01 09:45:00 | 4447.21 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-07-01 09:40:00 | 4426.60 | 2025-07-01 10:25:00 | 4447.00 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2025-07-09 10:50:00 | 4239.10 | 2025-07-09 10:55:00 | 4231.62 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-11 10:35:00 | 4118.70 | 2025-07-11 11:45:00 | 4099.79 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-07-11 10:35:00 | 4118.70 | 2025-07-11 13:45:00 | 4101.00 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-07-16 09:45:00 | 4062.00 | 2025-07-16 09:55:00 | 4076.99 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-16 09:45:00 | 4062.00 | 2025-07-16 10:40:00 | 4062.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-17 11:15:00 | 4075.20 | 2025-07-17 12:15:00 | 4065.62 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-07-17 11:15:00 | 4075.20 | 2025-07-17 15:20:00 | 4052.50 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-07-18 10:50:00 | 4044.90 | 2025-07-18 11:30:00 | 4053.11 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-25 10:00:00 | 4001.20 | 2025-07-25 10:15:00 | 3988.92 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-25 10:00:00 | 4001.20 | 2025-07-25 14:30:00 | 3985.50 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-28 10:55:00 | 4007.20 | 2025-07-28 11:30:00 | 3996.41 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-29 10:35:00 | 3945.70 | 2025-07-29 12:00:00 | 3955.93 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-05 10:50:00 | 4254.00 | 2025-08-05 11:10:00 | 4241.50 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-07 09:30:00 | 4247.60 | 2025-08-07 09:40:00 | 4264.05 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-08-07 09:30:00 | 4247.60 | 2025-08-07 11:30:00 | 4262.00 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-08-11 11:00:00 | 4218.70 | 2025-08-11 11:20:00 | 4207.97 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-12 09:55:00 | 4242.00 | 2025-08-12 10:10:00 | 4232.27 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-14 09:50:00 | 4390.00 | 2025-08-14 09:55:00 | 4378.15 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-20 09:30:00 | 4708.60 | 2025-08-20 09:40:00 | 4695.34 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-22 09:55:00 | 4664.90 | 2025-08-22 10:25:00 | 4675.40 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-25 10:30:00 | 4747.10 | 2025-08-25 10:35:00 | 4736.04 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-01 10:45:00 | 4794.00 | 2025-09-01 11:45:00 | 4781.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-03 09:50:00 | 4722.30 | 2025-09-03 09:55:00 | 4711.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-09 10:10:00 | 4768.60 | 2025-09-09 11:35:00 | 4754.94 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-10 09:45:00 | 4757.60 | 2025-09-10 09:55:00 | 4741.58 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-10 09:45:00 | 4757.60 | 2025-09-10 10:10:00 | 4757.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-11 09:40:00 | 4596.00 | 2025-09-11 10:15:00 | 4580.09 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-09-11 09:40:00 | 4596.00 | 2025-09-11 12:20:00 | 4591.90 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2025-09-15 10:10:00 | 4588.50 | 2025-09-15 10:25:00 | 4600.67 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-16 10:25:00 | 4699.30 | 2025-09-16 10:30:00 | 4689.18 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-17 11:10:00 | 4748.20 | 2025-09-17 11:40:00 | 4740.53 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-24 11:10:00 | 4639.50 | 2025-09-24 11:35:00 | 4654.37 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-24 11:10:00 | 4639.50 | 2025-09-24 14:15:00 | 4681.60 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2025-09-26 10:25:00 | 4560.40 | 2025-09-26 10:40:00 | 4538.74 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-09-26 10:25:00 | 4560.40 | 2025-09-26 10:45:00 | 4560.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-29 10:00:00 | 4546.30 | 2025-09-29 10:30:00 | 4530.47 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-09-30 10:40:00 | 4491.40 | 2025-09-30 10:55:00 | 4503.82 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-07 10:50:00 | 4300.30 | 2025-10-07 12:05:00 | 4309.93 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-15 10:50:00 | 4268.00 | 2025-10-15 11:00:00 | 4259.46 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-16 11:15:00 | 4296.40 | 2025-10-16 11:50:00 | 4307.64 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-10-16 11:15:00 | 4296.40 | 2025-10-16 15:10:00 | 4307.00 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-10-23 09:30:00 | 4332.10 | 2025-10-23 09:35:00 | 4321.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-24 11:10:00 | 4228.00 | 2025-10-24 11:35:00 | 4234.80 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-10-30 10:30:00 | 4194.00 | 2025-10-30 10:40:00 | 4201.23 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-04 10:35:00 | 4193.00 | 2025-11-04 11:35:00 | 4204.15 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-11-04 10:35:00 | 4193.00 | 2025-11-04 12:30:00 | 4193.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 11:00:00 | 4137.90 | 2025-11-06 11:10:00 | 4146.53 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-10 10:50:00 | 4051.00 | 2025-11-10 11:15:00 | 4065.43 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-11-10 10:50:00 | 4051.00 | 2025-11-10 11:40:00 | 4051.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-11 10:55:00 | 4049.50 | 2025-11-11 11:00:00 | 4042.42 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-11-12 10:25:00 | 4060.70 | 2025-11-12 10:55:00 | 4067.07 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-11-13 10:45:00 | 4080.00 | 2025-11-13 13:20:00 | 4072.32 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-17 09:30:00 | 4049.80 | 2025-11-17 10:50:00 | 4036.80 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-11-17 09:30:00 | 4049.80 | 2025-11-17 15:20:00 | 4036.90 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2025-11-18 09:40:00 | 4005.00 | 2025-11-18 09:45:00 | 4010.59 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-11-21 11:10:00 | 4058.20 | 2025-11-21 14:45:00 | 4043.34 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-11-21 11:10:00 | 4058.20 | 2025-11-21 15:20:00 | 4035.70 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2025-11-24 10:55:00 | 4000.10 | 2025-11-24 12:50:00 | 3989.73 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-11-24 10:55:00 | 4000.10 | 2025-11-24 14:50:00 | 3997.30 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2025-11-28 09:30:00 | 4042.40 | 2025-11-28 09:35:00 | 4034.63 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-01 09:45:00 | 3967.40 | 2025-12-01 09:50:00 | 3974.38 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-03 11:10:00 | 3899.60 | 2025-12-03 11:20:00 | 3887.67 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-03 11:10:00 | 3899.60 | 2025-12-03 12:00:00 | 3899.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-04 10:35:00 | 3920.30 | 2025-12-04 11:25:00 | 3931.80 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-12-04 10:35:00 | 3920.30 | 2025-12-04 11:35:00 | 3920.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 10:45:00 | 3901.20 | 2025-12-08 12:15:00 | 3890.66 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-08 10:45:00 | 3901.20 | 2025-12-08 12:25:00 | 3901.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-10 10:50:00 | 3850.00 | 2025-12-10 11:35:00 | 3837.28 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-10 10:50:00 | 3850.00 | 2025-12-10 15:20:00 | 3815.60 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2025-12-12 10:00:00 | 3857.00 | 2025-12-12 10:20:00 | 3848.08 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-16 10:15:00 | 3913.00 | 2025-12-16 11:15:00 | 3900.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-18 11:10:00 | 3778.30 | 2025-12-18 11:20:00 | 3785.01 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-12-22 09:30:00 | 3851.00 | 2025-12-22 10:05:00 | 3863.45 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-12-22 09:30:00 | 3851.00 | 2025-12-22 10:30:00 | 3851.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 11:10:00 | 3839.70 | 2025-12-23 11:50:00 | 3832.38 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-24 11:05:00 | 3813.80 | 2025-12-24 11:20:00 | 3819.14 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-12-26 11:05:00 | 3797.10 | 2025-12-26 11:35:00 | 3787.73 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-26 11:05:00 | 3797.10 | 2025-12-26 12:00:00 | 3797.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 11:05:00 | 3779.60 | 2025-12-30 12:00:00 | 3785.98 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-31 10:30:00 | 3775.50 | 2025-12-31 10:35:00 | 3788.37 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-12-31 10:30:00 | 3775.50 | 2025-12-31 11:35:00 | 3775.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 10:05:00 | 3746.40 | 2026-01-02 11:30:00 | 3736.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-06 10:50:00 | 3652.20 | 2026-01-06 11:20:00 | 3643.47 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-13 11:10:00 | 3822.20 | 2026-01-13 11:35:00 | 3834.45 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-20 10:50:00 | 3707.80 | 2026-01-20 12:00:00 | 3693.05 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-20 10:50:00 | 3707.80 | 2026-01-20 15:20:00 | 3661.40 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2026-01-28 11:05:00 | 3707.00 | 2026-01-28 11:35:00 | 3699.62 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-29 10:25:00 | 3679.30 | 2026-01-29 10:40:00 | 3664.58 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-29 10:25:00 | 3679.30 | 2026-01-29 11:20:00 | 3679.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 10:40:00 | 3704.30 | 2026-02-01 12:00:00 | 3695.92 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-06 10:45:00 | 3862.30 | 2026-02-06 11:10:00 | 3872.04 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-10 09:40:00 | 3964.50 | 2026-02-10 09:55:00 | 3955.01 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-11 11:00:00 | 3998.20 | 2026-02-11 11:10:00 | 3985.66 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-11 11:00:00 | 3998.20 | 2026-02-11 11:35:00 | 3998.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:15:00 | 3871.00 | 2026-02-19 11:50:00 | 3857.70 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-19 11:15:00 | 3871.00 | 2026-02-19 13:20:00 | 3871.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 10:40:00 | 3851.10 | 2026-02-23 11:30:00 | 3861.18 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-24 11:10:00 | 3808.30 | 2026-02-24 11:55:00 | 3816.87 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-25 11:10:00 | 3940.90 | 2026-02-25 11:20:00 | 3930.82 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-27 10:05:00 | 3814.90 | 2026-02-27 10:30:00 | 3824.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 10:55:00 | 3749.00 | 2026-03-05 11:05:00 | 3757.36 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-03-09 09:30:00 | 3897.00 | 2026-03-09 09:45:00 | 3880.48 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-17 10:35:00 | 3846.90 | 2026-03-17 11:05:00 | 3835.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-20 10:45:00 | 3839.10 | 2026-03-20 11:50:00 | 3827.28 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-27 10:00:00 | 3925.80 | 2026-03-27 10:15:00 | 3912.29 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-30 09:35:00 | 3922.80 | 2026-03-30 09:40:00 | 3906.95 | STOP_HIT | 1.00 | -0.40% |
