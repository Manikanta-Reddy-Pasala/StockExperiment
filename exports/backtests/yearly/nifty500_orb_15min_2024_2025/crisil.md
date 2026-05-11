# CRISIL Ltd. (CRISIL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 4160.70
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 11 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 87 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 52
- **Target hits / Stop hits / Partials:** 11 / 52 / 24
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 14.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 17 | 37.8% | 4 | 28 | 13 | 0.23% | 10.1% |
| BUY @ 2nd Alert (retest1) | 45 | 17 | 37.8% | 4 | 28 | 13 | 0.23% | 10.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 18 | 42.9% | 7 | 24 | 11 | 0.10% | 4.3% |
| SELL @ 2nd Alert (retest1) | 42 | 18 | 42.9% | 7 | 24 | 11 | 0.10% | 4.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 87 | 35 | 40.2% | 11 | 52 | 24 | 0.17% | 14.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:35:00 | 4345.90 | 4320.22 | 0.00 | ORB-long ORB[4290.95,4326.55] vol=1.9x ATR=11.96 |
| Stop hit — per-position SL triggered | 2024-05-14 10:45:00 | 4333.94 | 4322.09 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:55:00 | 4382.45 | 4363.90 | 0.00 | ORB-long ORB[4335.00,4359.80] vol=3.3x ATR=12.20 |
| Stop hit — per-position SL triggered | 2024-05-15 10:00:00 | 4370.25 | 4364.39 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:45:00 | 4347.30 | 4369.60 | 0.00 | ORB-short ORB[4364.15,4398.45] vol=2.4x ATR=13.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:50:00 | 4327.65 | 4368.20 | 0.00 | T1 1.5R @ 4327.65 |
| Stop hit — per-position SL triggered | 2024-05-16 10:00:00 | 4347.30 | 4366.83 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:45:00 | 4411.00 | 4434.44 | 0.00 | ORB-short ORB[4423.80,4462.55] vol=1.6x ATR=10.90 |
| Stop hit — per-position SL triggered | 2024-05-23 12:30:00 | 4421.90 | 4427.51 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 10:40:00 | 4102.85 | 4125.50 | 0.00 | ORB-short ORB[4115.35,4145.00] vol=5.5x ATR=11.88 |
| Stop hit — per-position SL triggered | 2024-06-11 10:55:00 | 4114.73 | 4124.25 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 09:45:00 | 4094.05 | 4109.19 | 0.00 | ORB-short ORB[4125.10,4143.80] vol=2.3x ATR=12.78 |
| Stop hit — per-position SL triggered | 2024-06-12 10:45:00 | 4106.83 | 4100.59 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 10:45:00 | 4130.00 | 4142.90 | 0.00 | ORB-short ORB[4140.00,4165.00] vol=1.6x ATR=10.37 |
| Stop hit — per-position SL triggered | 2024-06-14 10:55:00 | 4140.37 | 4142.15 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:55:00 | 4133.75 | 4132.55 | 0.00 | ORB-long ORB[4110.20,4130.00] vol=2.3x ATR=10.40 |
| Stop hit — per-position SL triggered | 2024-06-20 11:35:00 | 4123.35 | 4131.86 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 4226.65 | 4198.79 | 0.00 | ORB-long ORB[4159.60,4214.50] vol=2.1x ATR=15.12 |
| Stop hit — per-position SL triggered | 2024-06-21 10:15:00 | 4211.53 | 4207.62 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:30:00 | 4151.00 | 4172.36 | 0.00 | ORB-short ORB[4156.90,4182.15] vol=3.0x ATR=12.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:50:00 | 4131.88 | 4166.03 | 0.00 | T1 1.5R @ 4131.88 |
| Stop hit — per-position SL triggered | 2024-06-27 11:50:00 | 4151.00 | 4158.56 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:25:00 | 4311.70 | 4276.53 | 0.00 | ORB-long ORB[4250.00,4278.35] vol=2.0x ATR=12.83 |
| Stop hit — per-position SL triggered | 2024-07-09 10:35:00 | 4298.87 | 4280.88 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:30:00 | 4314.95 | 4283.07 | 0.00 | ORB-long ORB[4262.20,4301.40] vol=1.5x ATR=13.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:35:00 | 4334.71 | 4295.02 | 0.00 | T1 1.5R @ 4334.71 |
| Stop hit — per-position SL triggered | 2024-07-12 10:45:00 | 4314.95 | 4310.71 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:00:00 | 4310.85 | 4301.19 | 0.00 | ORB-long ORB[4247.90,4299.70] vol=1.7x ATR=22.27 |
| Stop hit — per-position SL triggered | 2024-07-22 10:15:00 | 4288.58 | 4300.84 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:40:00 | 4394.10 | 4409.52 | 0.00 | ORB-short ORB[4400.00,4424.00] vol=2.2x ATR=12.97 |
| Stop hit — per-position SL triggered | 2024-07-31 10:45:00 | 4407.07 | 4409.23 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:10:00 | 4390.00 | 4399.18 | 0.00 | ORB-short ORB[4393.80,4420.00] vol=1.6x ATR=13.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:55:00 | 4369.82 | 4396.37 | 0.00 | T1 1.5R @ 4369.82 |
| Target hit | 2024-08-01 15:20:00 | 4338.55 | 4364.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2024-08-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:40:00 | 4349.05 | 4324.57 | 0.00 | ORB-long ORB[4236.30,4300.00] vol=6.6x ATR=16.61 |
| Stop hit — per-position SL triggered | 2024-08-08 10:55:00 | 4332.44 | 4325.75 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:10:00 | 4421.75 | 4440.65 | 0.00 | ORB-short ORB[4426.25,4491.55] vol=2.4x ATR=22.78 |
| Stop hit — per-position SL triggered | 2024-08-09 13:40:00 | 4444.53 | 4432.74 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:05:00 | 4428.75 | 4404.54 | 0.00 | ORB-long ORB[4391.00,4428.50] vol=2.3x ATR=18.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 10:50:00 | 4456.67 | 4421.95 | 0.00 | T1 1.5R @ 4456.67 |
| Stop hit — per-position SL triggered | 2024-08-12 11:50:00 | 4428.75 | 4429.51 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 4511.00 | 4489.88 | 0.00 | ORB-long ORB[4440.00,4500.00] vol=1.9x ATR=14.59 |
| Stop hit — per-position SL triggered | 2024-08-16 10:05:00 | 4496.41 | 4493.47 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:35:00 | 4532.10 | 4518.77 | 0.00 | ORB-long ORB[4502.00,4531.20] vol=2.4x ATR=13.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 11:15:00 | 4551.85 | 4529.51 | 0.00 | T1 1.5R @ 4551.85 |
| Target hit | 2024-08-19 15:20:00 | 4648.90 | 4571.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-08-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:55:00 | 4536.85 | 4563.74 | 0.00 | ORB-short ORB[4570.00,4608.45] vol=2.3x ATR=14.06 |
| Stop hit — per-position SL triggered | 2024-08-22 10:30:00 | 4550.91 | 4553.12 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:35:00 | 4578.00 | 4554.39 | 0.00 | ORB-long ORB[4525.10,4575.00] vol=1.9x ATR=18.38 |
| Stop hit — per-position SL triggered | 2024-08-27 11:35:00 | 4559.62 | 4575.33 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:40:00 | 4500.95 | 4513.79 | 0.00 | ORB-short ORB[4506.60,4542.00] vol=2.4x ATR=8.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 12:00:00 | 4488.57 | 4506.29 | 0.00 | T1 1.5R @ 4488.57 |
| Target hit | 2024-08-29 14:50:00 | 4495.00 | 4493.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2024-08-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:35:00 | 4513.55 | 4503.32 | 0.00 | ORB-long ORB[4480.75,4509.35] vol=1.7x ATR=10.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 10:50:00 | 4529.71 | 4507.14 | 0.00 | T1 1.5R @ 4529.71 |
| Stop hit — per-position SL triggered | 2024-08-30 11:05:00 | 4513.55 | 4507.91 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:40:00 | 4545.00 | 4517.59 | 0.00 | ORB-long ORB[4456.35,4500.00] vol=1.9x ATR=18.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 10:45:00 | 4572.45 | 4522.76 | 0.00 | T1 1.5R @ 4572.45 |
| Stop hit — per-position SL triggered | 2024-09-02 11:45:00 | 4545.00 | 4532.00 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:50:00 | 4617.00 | 4586.32 | 0.00 | ORB-long ORB[4536.60,4567.10] vol=6.2x ATR=14.78 |
| Stop hit — per-position SL triggered | 2024-09-03 09:55:00 | 4602.22 | 4593.06 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:50:00 | 4607.40 | 4634.29 | 0.00 | ORB-short ORB[4617.00,4659.40] vol=2.2x ATR=12.66 |
| Stop hit — per-position SL triggered | 2024-09-05 11:05:00 | 4620.06 | 4631.34 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:35:00 | 4665.60 | 4636.68 | 0.00 | ORB-long ORB[4596.70,4632.50] vol=2.8x ATR=14.50 |
| Stop hit — per-position SL triggered | 2024-09-10 09:45:00 | 4651.10 | 4637.95 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 4674.00 | 4646.92 | 0.00 | ORB-long ORB[4595.00,4659.95] vol=2.2x ATR=17.50 |
| Stop hit — per-position SL triggered | 2024-09-11 09:35:00 | 4656.50 | 4648.37 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 10:10:00 | 4639.20 | 4649.93 | 0.00 | ORB-short ORB[4648.80,4685.00] vol=1.8x ATR=11.76 |
| Stop hit — per-position SL triggered | 2024-09-13 10:30:00 | 4650.96 | 4648.04 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:30:00 | 4651.00 | 4667.94 | 0.00 | ORB-short ORB[4660.00,4688.85] vol=2.3x ATR=12.57 |
| Stop hit — per-position SL triggered | 2024-09-16 10:45:00 | 4663.57 | 4667.47 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:35:00 | 4696.30 | 4719.53 | 0.00 | ORB-short ORB[4698.20,4766.55] vol=1.5x ATR=13.47 |
| Stop hit — per-position SL triggered | 2024-09-17 10:40:00 | 4709.77 | 4719.32 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 4606.00 | 4624.42 | 0.00 | ORB-short ORB[4615.05,4670.35] vol=2.4x ATR=11.63 |
| Stop hit — per-position SL triggered | 2024-09-24 10:05:00 | 4617.63 | 4626.95 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:30:00 | 4605.20 | 4616.53 | 0.00 | ORB-short ORB[4610.05,4650.00] vol=1.7x ATR=10.00 |
| Stop hit — per-position SL triggered | 2024-09-25 11:25:00 | 4615.20 | 4614.33 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:00:00 | 4629.10 | 4606.78 | 0.00 | ORB-long ORB[4567.50,4615.00] vol=3.1x ATR=19.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 10:10:00 | 4658.00 | 4618.01 | 0.00 | T1 1.5R @ 4658.00 |
| Stop hit — per-position SL triggered | 2024-09-30 13:30:00 | 4629.10 | 4636.83 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 10:45:00 | 4543.35 | 4586.14 | 0.00 | ORB-short ORB[4570.00,4628.85] vol=2.9x ATR=13.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 10:50:00 | 4523.72 | 4546.51 | 0.00 | T1 1.5R @ 4523.72 |
| Target hit | 2024-10-03 15:20:00 | 4494.95 | 4504.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-10-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:45:00 | 4352.80 | 4387.76 | 0.00 | ORB-short ORB[4383.25,4444.70] vol=1.7x ATR=17.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:05:00 | 4327.13 | 4372.95 | 0.00 | T1 1.5R @ 4327.13 |
| Target hit | 2024-10-07 15:20:00 | 4300.00 | 4305.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-10-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:50:00 | 4543.60 | 4519.21 | 0.00 | ORB-long ORB[4474.90,4537.55] vol=1.5x ATR=16.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 11:45:00 | 4568.83 | 4537.18 | 0.00 | T1 1.5R @ 4568.83 |
| Target hit | 2024-10-11 15:20:00 | 4691.90 | 4656.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2024-10-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 11:10:00 | 4755.00 | 4739.02 | 0.00 | ORB-long ORB[4692.30,4745.80] vol=2.5x ATR=12.55 |
| Stop hit — per-position SL triggered | 2024-10-15 11:25:00 | 4742.45 | 4741.36 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 10:00:00 | 4877.85 | 4895.06 | 0.00 | ORB-short ORB[4881.75,4917.95] vol=2.4x ATR=16.90 |
| Stop hit — per-position SL triggered | 2024-10-24 10:05:00 | 4894.75 | 4895.44 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-11-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:50:00 | 5281.25 | 5319.91 | 0.00 | ORB-short ORB[5320.00,5362.85] vol=1.9x ATR=18.59 |
| Stop hit — per-position SL triggered | 2024-11-27 11:05:00 | 5299.84 | 5317.08 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 10:50:00 | 5389.20 | 5340.20 | 0.00 | ORB-long ORB[5330.95,5372.00] vol=3.0x ATR=17.33 |
| Stop hit — per-position SL triggered | 2024-12-05 10:55:00 | 5371.87 | 5341.38 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:55:00 | 5490.00 | 5442.13 | 0.00 | ORB-long ORB[5397.00,5460.00] vol=1.5x ATR=22.91 |
| Stop hit — per-position SL triggered | 2024-12-11 10:00:00 | 5467.09 | 5445.66 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 09:30:00 | 5589.40 | 5556.33 | 0.00 | ORB-long ORB[5504.40,5570.00] vol=2.7x ATR=20.99 |
| Stop hit — per-position SL triggered | 2024-12-13 09:35:00 | 5568.41 | 5584.71 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 11:05:00 | 5834.20 | 5794.29 | 0.00 | ORB-long ORB[5740.00,5822.90] vol=1.9x ATR=23.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 11:20:00 | 5869.98 | 5804.77 | 0.00 | T1 1.5R @ 5869.98 |
| Stop hit — per-position SL triggered | 2024-12-18 12:45:00 | 5834.20 | 5826.75 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 5798.00 | 5830.87 | 0.00 | ORB-short ORB[5807.50,5859.85] vol=1.6x ATR=20.10 |
| Stop hit — per-position SL triggered | 2024-12-20 09:35:00 | 5818.10 | 5830.39 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:05:00 | 5586.15 | 5619.82 | 0.00 | ORB-short ORB[5634.40,5694.95] vol=1.5x ATR=19.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 12:30:00 | 5557.07 | 5607.15 | 0.00 | T1 1.5R @ 5557.07 |
| Stop hit — per-position SL triggered | 2024-12-26 12:45:00 | 5586.15 | 5604.96 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:40:00 | 5624.75 | 5573.01 | 0.00 | ORB-long ORB[5523.00,5586.05] vol=1.8x ATR=22.53 |
| Stop hit — per-position SL triggered | 2024-12-30 09:45:00 | 5602.22 | 5579.78 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 09:35:00 | 6187.45 | 6137.72 | 0.00 | ORB-long ORB[6103.05,6145.65] vol=1.6x ATR=25.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 09:55:00 | 6226.13 | 6170.47 | 0.00 | T1 1.5R @ 6226.13 |
| Stop hit — per-position SL triggered | 2025-01-06 10:20:00 | 6187.45 | 6199.90 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 10:05:00 | 5949.95 | 5866.74 | 0.00 | ORB-long ORB[5762.00,5847.85] vol=1.8x ATR=40.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 10:25:00 | 6011.08 | 5897.23 | 0.00 | T1 1.5R @ 6011.08 |
| Stop hit — per-position SL triggered | 2025-01-14 11:20:00 | 5949.95 | 5924.57 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 5879.95 | 5923.76 | 0.00 | ORB-short ORB[5890.05,5973.90] vol=1.8x ATR=29.68 |
| Stop hit — per-position SL triggered | 2025-01-15 10:25:00 | 5909.63 | 5902.88 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 09:30:00 | 5790.00 | 5806.14 | 0.00 | ORB-short ORB[5790.30,5839.95] vol=1.7x ATR=17.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 09:35:00 | 5763.98 | 5789.43 | 0.00 | T1 1.5R @ 5763.98 |
| Target hit | 2025-01-16 10:15:00 | 5776.35 | 5762.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — SELL (started 2025-01-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:55:00 | 5413.25 | 5477.82 | 0.00 | ORB-short ORB[5460.75,5529.60] vol=2.6x ATR=22.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:25:00 | 5379.01 | 5455.32 | 0.00 | T1 1.5R @ 5379.01 |
| Stop hit — per-position SL triggered | 2025-01-21 10:45:00 | 5413.25 | 5437.68 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:10:00 | 5410.00 | 5416.04 | 0.00 | ORB-short ORB[5412.20,5476.15] vol=6.1x ATR=15.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:35:00 | 5387.48 | 5414.49 | 0.00 | T1 1.5R @ 5387.48 |
| Target hit | 2025-02-01 13:15:00 | 5385.15 | 5377.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 55 — SELL (started 2025-02-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:45:00 | 5147.00 | 5184.02 | 0.00 | ORB-short ORB[5169.95,5219.95] vol=1.6x ATR=18.13 |
| Stop hit — per-position SL triggered | 2025-02-07 11:05:00 | 5165.13 | 5182.23 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-19 10:50:00 | 4749.95 | 4785.63 | 0.00 | ORB-short ORB[4765.55,4834.55] vol=2.1x ATR=19.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 13:05:00 | 4720.86 | 4766.09 | 0.00 | T1 1.5R @ 4720.86 |
| Target hit | 2025-02-19 15:20:00 | 4694.95 | 4752.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 09:30:00 | 4552.20 | 4576.05 | 0.00 | ORB-short ORB[4560.20,4609.95] vol=1.8x ATR=22.98 |
| Stop hit — per-position SL triggered | 2025-02-24 09:35:00 | 4575.18 | 4573.26 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 10:40:00 | 4429.00 | 4395.27 | 0.00 | ORB-long ORB[4352.45,4403.25] vol=2.3x ATR=12.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 10:45:00 | 4447.02 | 4404.73 | 0.00 | T1 1.5R @ 4447.02 |
| Stop hit — per-position SL triggered | 2025-03-17 10:50:00 | 4429.00 | 4407.00 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 10:30:00 | 4220.00 | 4202.37 | 0.00 | ORB-long ORB[4180.05,4218.00] vol=3.7x ATR=12.27 |
| Stop hit — per-position SL triggered | 2025-03-28 10:40:00 | 4207.73 | 4205.49 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-04-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:40:00 | 4271.60 | 4269.63 | 0.00 | ORB-long ORB[4218.10,4259.00] vol=1.6x ATR=13.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 11:20:00 | 4292.50 | 4270.70 | 0.00 | T1 1.5R @ 4292.50 |
| Target hit | 2025-04-15 15:20:00 | 4364.60 | 4310.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2025-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:05:00 | 4409.80 | 4374.19 | 0.00 | ORB-long ORB[4349.90,4402.50] vol=4.6x ATR=14.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 13:25:00 | 4431.77 | 4396.27 | 0.00 | T1 1.5R @ 4431.77 |
| Target hit | 2025-04-16 15:20:00 | 4465.30 | 4427.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:55:00 | 4611.00 | 4567.57 | 0.00 | ORB-long ORB[4535.70,4600.00] vol=2.3x ATR=14.12 |
| Stop hit — per-position SL triggered | 2025-04-21 11:15:00 | 4596.88 | 4576.44 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:40:00 | 4670.30 | 4694.45 | 0.00 | ORB-short ORB[4690.00,4753.90] vol=2.1x ATR=12.78 |
| Stop hit — per-position SL triggered | 2025-04-23 10:50:00 | 4683.08 | 4693.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 10:35:00 | 4345.90 | 2024-05-14 10:45:00 | 4333.94 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-15 09:55:00 | 4382.45 | 2024-05-15 10:00:00 | 4370.25 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-16 09:45:00 | 4347.30 | 2024-05-16 09:50:00 | 4327.65 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-16 09:45:00 | 4347.30 | 2024-05-16 10:00:00 | 4347.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-23 10:45:00 | 4411.00 | 2024-05-23 12:30:00 | 4421.90 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-06-11 10:40:00 | 4102.85 | 2024-06-11 10:55:00 | 4114.73 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-06-12 09:45:00 | 4094.05 | 2024-06-12 10:45:00 | 4106.83 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-14 10:45:00 | 4130.00 | 2024-06-14 10:55:00 | 4140.37 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-20 10:55:00 | 4133.75 | 2024-06-20 11:35:00 | 4123.35 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-21 09:35:00 | 4226.65 | 2024-06-21 10:15:00 | 4211.53 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-27 10:30:00 | 4151.00 | 2024-06-27 10:50:00 | 4131.88 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-06-27 10:30:00 | 4151.00 | 2024-06-27 11:50:00 | 4151.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-09 10:25:00 | 4311.70 | 2024-07-09 10:35:00 | 4298.87 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-12 10:30:00 | 4314.95 | 2024-07-12 10:35:00 | 4334.71 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-12 10:30:00 | 4314.95 | 2024-07-12 10:45:00 | 4314.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-22 10:00:00 | 4310.85 | 2024-07-22 10:15:00 | 4288.58 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-07-31 10:40:00 | 4394.10 | 2024-07-31 10:45:00 | 4407.07 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-01 11:10:00 | 4390.00 | 2024-08-01 11:55:00 | 4369.82 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-01 11:10:00 | 4390.00 | 2024-08-01 15:20:00 | 4338.55 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2024-08-08 10:40:00 | 4349.05 | 2024-08-08 10:55:00 | 4332.44 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-09 10:10:00 | 4421.75 | 2024-08-09 13:40:00 | 4444.53 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-08-12 10:05:00 | 4428.75 | 2024-08-12 10:50:00 | 4456.67 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-08-12 10:05:00 | 4428.75 | 2024-08-12 11:50:00 | 4428.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-16 09:30:00 | 4511.00 | 2024-08-16 10:05:00 | 4496.41 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-19 10:35:00 | 4532.10 | 2024-08-19 11:15:00 | 4551.85 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-08-19 10:35:00 | 4532.10 | 2024-08-19 15:20:00 | 4648.90 | TARGET_HIT | 0.50 | 2.58% |
| SELL | retest1 | 2024-08-22 09:55:00 | 4536.85 | 2024-08-22 10:30:00 | 4550.91 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-27 09:35:00 | 4578.00 | 2024-08-27 11:35:00 | 4559.62 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-29 10:40:00 | 4500.95 | 2024-08-29 12:00:00 | 4488.57 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-08-29 10:40:00 | 4500.95 | 2024-08-29 14:50:00 | 4495.00 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-08-30 10:35:00 | 4513.55 | 2024-08-30 10:50:00 | 4529.71 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-08-30 10:35:00 | 4513.55 | 2024-08-30 11:05:00 | 4513.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-02 10:40:00 | 4545.00 | 2024-09-02 10:45:00 | 4572.45 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-09-02 10:40:00 | 4545.00 | 2024-09-02 11:45:00 | 4545.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:50:00 | 4617.00 | 2024-09-03 09:55:00 | 4602.22 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-05 10:50:00 | 4607.40 | 2024-09-05 11:05:00 | 4620.06 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-10 09:35:00 | 4665.60 | 2024-09-10 09:45:00 | 4651.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-11 09:30:00 | 4674.00 | 2024-09-11 09:35:00 | 4656.50 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-09-13 10:10:00 | 4639.20 | 2024-09-13 10:30:00 | 4650.96 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-16 10:30:00 | 4651.00 | 2024-09-16 10:45:00 | 4663.57 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-17 10:35:00 | 4696.30 | 2024-09-17 10:40:00 | 4709.77 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-24 09:55:00 | 4606.00 | 2024-09-24 10:05:00 | 4617.63 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-25 10:30:00 | 4605.20 | 2024-09-25 11:25:00 | 4615.20 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-30 10:00:00 | 4629.10 | 2024-09-30 10:10:00 | 4658.00 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-09-30 10:00:00 | 4629.10 | 2024-09-30 13:30:00 | 4629.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-03 10:45:00 | 4543.35 | 2024-10-03 10:50:00 | 4523.72 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-03 10:45:00 | 4543.35 | 2024-10-03 15:20:00 | 4494.95 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2024-10-07 09:45:00 | 4352.80 | 2024-10-07 10:05:00 | 4327.13 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-10-07 09:45:00 | 4352.80 | 2024-10-07 15:20:00 | 4300.00 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2024-10-11 09:50:00 | 4543.60 | 2024-10-11 11:45:00 | 4568.83 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-10-11 09:50:00 | 4543.60 | 2024-10-11 15:20:00 | 4691.90 | TARGET_HIT | 0.50 | 3.26% |
| BUY | retest1 | 2024-10-15 11:10:00 | 4755.00 | 2024-10-15 11:25:00 | 4742.45 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-24 10:00:00 | 4877.85 | 2024-10-24 10:05:00 | 4894.75 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-27 10:50:00 | 5281.25 | 2024-11-27 11:05:00 | 5299.84 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-05 10:50:00 | 5389.20 | 2024-12-05 10:55:00 | 5371.87 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-11 09:55:00 | 5490.00 | 2024-12-11 10:00:00 | 5467.09 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-13 09:30:00 | 5589.40 | 2024-12-13 09:35:00 | 5568.41 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-18 11:05:00 | 5834.20 | 2024-12-18 11:20:00 | 5869.98 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-12-18 11:05:00 | 5834.20 | 2024-12-18 12:45:00 | 5834.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 09:30:00 | 5798.00 | 2024-12-20 09:35:00 | 5818.10 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-26 11:05:00 | 5586.15 | 2024-12-26 12:30:00 | 5557.07 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-12-26 11:05:00 | 5586.15 | 2024-12-26 12:45:00 | 5586.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 09:40:00 | 5624.75 | 2024-12-30 09:45:00 | 5602.22 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-01-06 09:35:00 | 6187.45 | 2025-01-06 09:55:00 | 6226.13 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-01-06 09:35:00 | 6187.45 | 2025-01-06 10:20:00 | 6187.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-14 10:05:00 | 5949.95 | 2025-01-14 10:25:00 | 6011.08 | PARTIAL | 0.50 | 1.03% |
| BUY | retest1 | 2025-01-14 10:05:00 | 5949.95 | 2025-01-14 11:20:00 | 5949.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-15 09:30:00 | 5879.95 | 2025-01-15 10:25:00 | 5909.63 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-01-16 09:30:00 | 5790.00 | 2025-01-16 09:35:00 | 5763.98 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-16 09:30:00 | 5790.00 | 2025-01-16 10:15:00 | 5776.35 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2025-01-21 09:55:00 | 5413.25 | 2025-01-21 10:25:00 | 5379.01 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-01-21 09:55:00 | 5413.25 | 2025-01-21 10:45:00 | 5413.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-01 11:10:00 | 5410.00 | 2025-02-01 11:35:00 | 5387.48 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-02-01 11:10:00 | 5410.00 | 2025-02-01 13:15:00 | 5385.15 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2025-02-07 10:45:00 | 5147.00 | 2025-02-07 11:05:00 | 5165.13 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-02-19 10:50:00 | 4749.95 | 2025-02-19 13:05:00 | 4720.86 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-02-19 10:50:00 | 4749.95 | 2025-02-19 15:20:00 | 4694.95 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2025-02-24 09:30:00 | 4552.20 | 2025-02-24 09:35:00 | 4575.18 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-03-17 10:40:00 | 4429.00 | 2025-03-17 10:45:00 | 4447.02 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-03-17 10:40:00 | 4429.00 | 2025-03-17 10:50:00 | 4429.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-28 10:30:00 | 4220.00 | 2025-03-28 10:40:00 | 4207.73 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-15 10:40:00 | 4271.60 | 2025-04-15 11:20:00 | 4292.50 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-15 10:40:00 | 4271.60 | 2025-04-15 15:20:00 | 4364.60 | TARGET_HIT | 0.50 | 2.18% |
| BUY | retest1 | 2025-04-16 11:05:00 | 4409.80 | 2025-04-16 13:25:00 | 4431.77 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-16 11:05:00 | 4409.80 | 2025-04-16 15:20:00 | 4465.30 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2025-04-21 10:55:00 | 4611.00 | 2025-04-21 11:15:00 | 4596.88 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-23 10:40:00 | 4670.30 | 2025-04-23 10:50:00 | 4683.08 | STOP_HIT | 1.00 | -0.27% |
