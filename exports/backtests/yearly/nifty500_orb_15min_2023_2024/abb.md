# ABB India Ltd. (ABB)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2023-09-08 15:25:00 (6301 bars)
- **Last close:** 4462.40
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
| ENTRY1 | 37 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 7 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 30
- **Target hits / Stop hits / Partials:** 7 / 30 / 18
- **Avg / median % per leg:** 0.26% / 0.00%
- **Sum % (uncompounded):** 14.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 17 | 48.6% | 5 | 18 | 12 | 0.24% | 8.4% |
| BUY @ 2nd Alert (retest1) | 35 | 17 | 48.6% | 5 | 18 | 12 | 0.24% | 8.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 20 | 8 | 40.0% | 2 | 12 | 6 | 0.29% | 5.9% |
| SELL @ 2nd Alert (retest1) | 20 | 8 | 40.0% | 2 | 12 | 6 | 0.29% | 5.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 55 | 25 | 45.5% | 7 | 30 | 18 | 0.26% | 14.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 10:25:00 | 3906.50 | 3881.28 | 0.00 | ORB-long ORB[3850.00,3883.00] vol=1.7x ATR=17.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-12 12:20:00 | 3932.21 | 3899.53 | 0.00 | T1 1.5R @ 3932.21 |
| Target hit | 2023-05-12 15:20:00 | 3941.85 | 3921.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2023-05-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 10:00:00 | 3907.95 | 3882.27 | 0.00 | ORB-long ORB[3845.60,3878.00] vol=1.8x ATR=10.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-16 10:25:00 | 3923.29 | 3899.79 | 0.00 | T1 1.5R @ 3923.29 |
| Stop hit — per-position SL triggered | 2023-05-16 11:00:00 | 3907.95 | 3903.15 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 09:40:00 | 3940.95 | 3925.34 | 0.00 | ORB-long ORB[3895.00,3925.00] vol=2.1x ATR=9.57 |
| Stop hit — per-position SL triggered | 2023-05-17 09:55:00 | 3931.38 | 3929.68 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 10:55:00 | 3953.95 | 3922.90 | 0.00 | ORB-long ORB[3885.10,3919.85] vol=1.6x ATR=8.30 |
| Stop hit — per-position SL triggered | 2023-05-24 11:00:00 | 3945.65 | 3924.04 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:45:00 | 3922.50 | 3915.44 | 0.00 | ORB-long ORB[3881.95,3919.55] vol=1.5x ATR=9.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 10:05:00 | 3936.96 | 3919.36 | 0.00 | T1 1.5R @ 3936.96 |
| Stop hit — per-position SL triggered | 2023-05-25 10:20:00 | 3922.50 | 3920.07 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:30:00 | 3987.50 | 3969.79 | 0.00 | ORB-long ORB[3938.50,3984.75] vol=1.8x ATR=12.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 09:40:00 | 4006.62 | 3985.41 | 0.00 | T1 1.5R @ 4006.62 |
| Stop hit — per-position SL triggered | 2023-06-02 09:55:00 | 3987.50 | 3988.13 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 10:20:00 | 3999.05 | 4038.29 | 0.00 | ORB-short ORB[4020.60,4069.00] vol=1.7x ATR=11.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 13:10:00 | 3981.59 | 4020.63 | 0.00 | T1 1.5R @ 3981.59 |
| Stop hit — per-position SL triggered | 2023-06-05 14:05:00 | 3999.05 | 4009.83 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:30:00 | 4065.90 | 4052.44 | 0.00 | ORB-long ORB[4032.95,4061.40] vol=1.6x ATR=9.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 10:05:00 | 4080.05 | 4061.04 | 0.00 | T1 1.5R @ 4080.05 |
| Stop hit — per-position SL triggered | 2023-06-06 10:40:00 | 4065.90 | 4066.55 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:20:00 | 4088.85 | 4078.21 | 0.00 | ORB-long ORB[4058.90,4088.30] vol=1.6x ATR=9.60 |
| Stop hit — per-position SL triggered | 2023-06-07 11:10:00 | 4079.25 | 4080.44 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 09:35:00 | 4113.45 | 4132.30 | 0.00 | ORB-short ORB[4122.40,4171.95] vol=1.7x ATR=12.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 10:00:00 | 4094.80 | 4125.89 | 0.00 | T1 1.5R @ 4094.80 |
| Stop hit — per-position SL triggered | 2023-06-08 10:25:00 | 4113.45 | 4122.80 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 11:10:00 | 4120.00 | 4139.25 | 0.00 | ORB-short ORB[4125.00,4155.00] vol=1.6x ATR=7.62 |
| Stop hit — per-position SL triggered | 2023-06-09 12:15:00 | 4127.62 | 4136.62 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:05:00 | 4209.90 | 4184.68 | 0.00 | ORB-long ORB[4140.80,4195.45] vol=2.1x ATR=12.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:20:00 | 4228.08 | 4195.33 | 0.00 | T1 1.5R @ 4228.08 |
| Target hit | 2023-06-13 15:20:00 | 4322.15 | 4267.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2023-06-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 10:50:00 | 4302.95 | 4320.60 | 0.00 | ORB-short ORB[4316.40,4348.25] vol=1.8x ATR=8.81 |
| Stop hit — per-position SL triggered | 2023-06-15 10:55:00 | 4311.76 | 4320.31 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 11:00:00 | 4335.05 | 4317.52 | 0.00 | ORB-long ORB[4288.00,4322.00] vol=2.2x ATR=9.38 |
| Stop hit — per-position SL triggered | 2023-06-16 11:05:00 | 4325.67 | 4317.74 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:15:00 | 4368.05 | 4412.04 | 0.00 | ORB-short ORB[4380.05,4443.90] vol=1.9x ATR=9.86 |
| Stop hit — per-position SL triggered | 2023-06-21 11:35:00 | 4377.91 | 4407.30 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:30:00 | 4420.00 | 4398.12 | 0.00 | ORB-long ORB[4360.70,4410.00] vol=1.7x ATR=13.37 |
| Stop hit — per-position SL triggered | 2023-07-05 09:40:00 | 4406.63 | 4402.98 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:40:00 | 4375.35 | 4408.18 | 0.00 | ORB-short ORB[4386.20,4430.00] vol=3.7x ATR=9.44 |
| Stop hit — per-position SL triggered | 2023-07-07 10:45:00 | 4384.79 | 4406.94 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:40:00 | 4528.00 | 4503.07 | 0.00 | ORB-long ORB[4477.50,4517.00] vol=1.8x ATR=12.51 |
| Stop hit — per-position SL triggered | 2023-07-12 09:45:00 | 4515.49 | 4504.67 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 11:00:00 | 4388.10 | 4422.92 | 0.00 | ORB-short ORB[4422.00,4463.30] vol=1.7x ATR=10.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 12:20:00 | 4372.40 | 4409.04 | 0.00 | T1 1.5R @ 4372.40 |
| Stop hit — per-position SL triggered | 2023-07-14 12:50:00 | 4388.10 | 4402.85 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:30:00 | 4485.05 | 4461.93 | 0.00 | ORB-long ORB[4421.00,4480.50] vol=1.9x ATR=10.47 |
| Stop hit — per-position SL triggered | 2023-07-18 09:40:00 | 4474.58 | 4467.05 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 10:35:00 | 4415.95 | 4459.48 | 0.00 | ORB-short ORB[4451.00,4505.10] vol=1.8x ATR=15.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 10:40:00 | 4392.82 | 4428.91 | 0.00 | T1 1.5R @ 4392.82 |
| Target hit | 2023-07-20 15:20:00 | 4206.20 | 4220.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2023-07-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 09:50:00 | 4297.85 | 4255.36 | 0.00 | ORB-long ORB[4199.20,4250.00] vol=2.0x ATR=16.45 |
| Stop hit — per-position SL triggered | 2023-07-21 09:55:00 | 4281.40 | 4257.54 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 09:40:00 | 4306.80 | 4288.88 | 0.00 | ORB-long ORB[4250.00,4305.00] vol=1.7x ATR=15.65 |
| Stop hit — per-position SL triggered | 2023-07-24 09:55:00 | 4291.15 | 4292.84 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:45:00 | 4363.05 | 4355.89 | 0.00 | ORB-long ORB[4331.45,4360.65] vol=2.0x ATR=12.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 09:55:00 | 4381.05 | 4357.54 | 0.00 | T1 1.5R @ 4381.05 |
| Stop hit — per-position SL triggered | 2023-07-26 10:15:00 | 4363.05 | 4362.98 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 10:05:00 | 4450.00 | 4432.97 | 0.00 | ORB-long ORB[4400.00,4445.95] vol=1.8x ATR=10.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 10:20:00 | 4466.25 | 4440.38 | 0.00 | T1 1.5R @ 4466.25 |
| Target hit | 2023-07-27 11:25:00 | 4453.00 | 4456.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — BUY (started 2023-07-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:35:00 | 4473.80 | 4450.54 | 0.00 | ORB-long ORB[4410.25,4461.80] vol=1.6x ATR=15.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 10:15:00 | 4497.22 | 4466.95 | 0.00 | T1 1.5R @ 4497.22 |
| Target hit | 2023-07-28 15:20:00 | 4513.25 | 4502.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2023-08-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 10:05:00 | 4492.10 | 4453.65 | 0.00 | ORB-long ORB[4417.60,4470.00] vol=2.1x ATR=14.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 10:15:00 | 4514.34 | 4463.61 | 0.00 | T1 1.5R @ 4514.34 |
| Stop hit — per-position SL triggered | 2023-08-07 10:35:00 | 4492.10 | 4474.03 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 11:10:00 | 4484.45 | 4499.65 | 0.00 | ORB-short ORB[4494.65,4547.95] vol=2.8x ATR=9.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 11:15:00 | 4470.22 | 4497.41 | 0.00 | T1 1.5R @ 4470.22 |
| Stop hit — per-position SL triggered | 2023-08-08 11:20:00 | 4484.45 | 4496.98 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-08-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 09:35:00 | 4552.00 | 4526.99 | 0.00 | ORB-long ORB[4505.05,4537.00] vol=2.7x ATR=16.54 |
| Stop hit — per-position SL triggered | 2023-08-11 09:50:00 | 4535.46 | 4532.13 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 10:10:00 | 4290.45 | 4318.03 | 0.00 | ORB-short ORB[4291.15,4335.65] vol=1.6x ATR=12.56 |
| Stop hit — per-position SL triggered | 2023-08-18 10:20:00 | 4303.01 | 4316.47 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-21 09:55:00 | 4255.00 | 4275.62 | 0.00 | ORB-short ORB[4261.00,4298.80] vol=1.7x ATR=11.33 |
| Stop hit — per-position SL triggered | 2023-08-21 10:05:00 | 4266.33 | 4273.35 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 11:05:00 | 4294.40 | 4306.88 | 0.00 | ORB-short ORB[4301.85,4339.75] vol=1.7x ATR=8.45 |
| Stop hit — per-position SL triggered | 2023-08-22 12:50:00 | 4302.85 | 4302.22 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:25:00 | 4293.75 | 4316.05 | 0.00 | ORB-short ORB[4305.05,4345.00] vol=2.3x ATR=11.99 |
| Stop hit — per-position SL triggered | 2023-08-25 10:45:00 | 4305.74 | 4312.42 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 09:50:00 | 4305.10 | 4283.87 | 0.00 | ORB-long ORB[4246.25,4292.10] vol=1.9x ATR=12.06 |
| Stop hit — per-position SL triggered | 2023-08-28 10:00:00 | 4293.04 | 4287.92 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:15:00 | 4372.75 | 4349.52 | 0.00 | ORB-long ORB[4315.10,4365.00] vol=2.1x ATR=13.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 10:25:00 | 4392.92 | 4359.46 | 0.00 | T1 1.5R @ 4392.92 |
| Target hit | 2023-09-05 15:20:00 | 4442.00 | 4405.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2023-09-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:35:00 | 4509.65 | 4473.23 | 0.00 | ORB-long ORB[4443.05,4488.35] vol=2.8x ATR=14.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 10:10:00 | 4531.09 | 4492.72 | 0.00 | T1 1.5R @ 4531.09 |
| Stop hit — per-position SL triggered | 2023-09-06 10:40:00 | 4509.65 | 4499.77 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 10:15:00 | 4468.95 | 4479.72 | 0.00 | ORB-short ORB[4474.00,4507.00] vol=1.8x ATR=12.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 11:35:00 | 4450.62 | 4472.92 | 0.00 | T1 1.5R @ 4450.62 |
| Target hit | 2023-09-07 15:20:00 | 4447.10 | 4463.80 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 10:25:00 | 3906.50 | 2023-05-12 12:20:00 | 3932.21 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2023-05-12 10:25:00 | 3906.50 | 2023-05-12 15:20:00 | 3941.85 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2023-05-16 10:00:00 | 3907.95 | 2023-05-16 10:25:00 | 3923.29 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-05-16 10:00:00 | 3907.95 | 2023-05-16 11:00:00 | 3907.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-17 09:40:00 | 3940.95 | 2023-05-17 09:55:00 | 3931.38 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-24 10:55:00 | 3953.95 | 2023-05-24 11:00:00 | 3945.65 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-05-25 09:45:00 | 3922.50 | 2023-05-25 10:05:00 | 3936.96 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-05-25 09:45:00 | 3922.50 | 2023-05-25 10:20:00 | 3922.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-02 09:30:00 | 3987.50 | 2023-06-02 09:40:00 | 4006.62 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-06-02 09:30:00 | 3987.50 | 2023-06-02 09:55:00 | 3987.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-05 10:20:00 | 3999.05 | 2023-06-05 13:10:00 | 3981.59 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-06-05 10:20:00 | 3999.05 | 2023-06-05 14:05:00 | 3999.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-06 09:30:00 | 4065.90 | 2023-06-06 10:05:00 | 4080.05 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-06-06 09:30:00 | 4065.90 | 2023-06-06 10:40:00 | 4065.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-07 10:20:00 | 4088.85 | 2023-06-07 11:10:00 | 4079.25 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-06-08 09:35:00 | 4113.45 | 2023-06-08 10:00:00 | 4094.80 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-06-08 09:35:00 | 4113.45 | 2023-06-08 10:25:00 | 4113.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-09 11:10:00 | 4120.00 | 2023-06-09 12:15:00 | 4127.62 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-06-13 10:05:00 | 4209.90 | 2023-06-13 10:20:00 | 4228.08 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-06-13 10:05:00 | 4209.90 | 2023-06-13 15:20:00 | 4322.15 | TARGET_HIT | 0.50 | 2.67% |
| SELL | retest1 | 2023-06-15 10:50:00 | 4302.95 | 2023-06-15 10:55:00 | 4311.76 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-16 11:00:00 | 4335.05 | 2023-06-16 11:05:00 | 4325.67 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-06-21 11:15:00 | 4368.05 | 2023-06-21 11:35:00 | 4377.91 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-05 09:30:00 | 4420.00 | 2023-07-05 09:40:00 | 4406.63 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-07-07 10:40:00 | 4375.35 | 2023-07-07 10:45:00 | 4384.79 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-12 09:40:00 | 4528.00 | 2023-07-12 09:45:00 | 4515.49 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-07-14 11:00:00 | 4388.10 | 2023-07-14 12:20:00 | 4372.40 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-07-14 11:00:00 | 4388.10 | 2023-07-14 12:50:00 | 4388.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-18 09:30:00 | 4485.05 | 2023-07-18 09:40:00 | 4474.58 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-07-20 10:35:00 | 4415.95 | 2023-07-20 10:40:00 | 4392.82 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-07-20 10:35:00 | 4415.95 | 2023-07-20 15:20:00 | 4206.20 | TARGET_HIT | 0.50 | 4.75% |
| BUY | retest1 | 2023-07-21 09:50:00 | 4297.85 | 2023-07-21 09:55:00 | 4281.40 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-07-24 09:40:00 | 4306.80 | 2023-07-24 09:55:00 | 4291.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-07-26 09:45:00 | 4363.05 | 2023-07-26 09:55:00 | 4381.05 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-07-26 09:45:00 | 4363.05 | 2023-07-26 10:15:00 | 4363.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-27 10:05:00 | 4450.00 | 2023-07-27 10:20:00 | 4466.25 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-07-27 10:05:00 | 4450.00 | 2023-07-27 11:25:00 | 4453.00 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2023-07-28 09:35:00 | 4473.80 | 2023-07-28 10:15:00 | 4497.22 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-28 09:35:00 | 4473.80 | 2023-07-28 15:20:00 | 4513.25 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2023-08-07 10:05:00 | 4492.10 | 2023-08-07 10:15:00 | 4514.34 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-08-07 10:05:00 | 4492.10 | 2023-08-07 10:35:00 | 4492.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-08 11:10:00 | 4484.45 | 2023-08-08 11:15:00 | 4470.22 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-08-08 11:10:00 | 4484.45 | 2023-08-08 11:20:00 | 4484.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-11 09:35:00 | 4552.00 | 2023-08-11 09:50:00 | 4535.46 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-08-18 10:10:00 | 4290.45 | 2023-08-18 10:20:00 | 4303.01 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-08-21 09:55:00 | 4255.00 | 2023-08-21 10:05:00 | 4266.33 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-22 11:05:00 | 4294.40 | 2023-08-22 12:50:00 | 4302.85 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-25 10:25:00 | 4293.75 | 2023-08-25 10:45:00 | 4305.74 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-28 09:50:00 | 4305.10 | 2023-08-28 10:00:00 | 4293.04 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-09-05 10:15:00 | 4372.75 | 2023-09-05 10:25:00 | 4392.92 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-09-05 10:15:00 | 4372.75 | 2023-09-05 15:20:00 | 4442.00 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2023-09-06 09:35:00 | 4509.65 | 2023-09-06 10:10:00 | 4531.09 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-09-06 09:35:00 | 4509.65 | 2023-09-06 10:40:00 | 4509.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-07 10:15:00 | 4468.95 | 2023-09-07 11:35:00 | 4450.62 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-09-07 10:15:00 | 4468.95 | 2023-09-07 15:20:00 | 4447.10 | TARGET_HIT | 0.50 | 0.49% |
