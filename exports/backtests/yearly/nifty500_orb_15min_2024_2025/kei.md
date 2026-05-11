# KEI Industries Ltd. (KEI)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 5117.00
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
| PARTIAL | 19 |
| TARGET_HIT | 8 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 29
- **Target hits / Stop hits / Partials:** 8 / 29 / 19
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 12.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 12 | 48.0% | 3 | 13 | 9 | 0.18% | 4.4% |
| BUY @ 2nd Alert (retest1) | 25 | 12 | 48.0% | 3 | 13 | 9 | 0.18% | 4.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 31 | 15 | 48.4% | 5 | 16 | 10 | 0.25% | 7.9% |
| SELL @ 2nd Alert (retest1) | 31 | 15 | 48.4% | 5 | 16 | 10 | 0.25% | 7.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 56 | 27 | 48.2% | 8 | 29 | 19 | 0.22% | 12.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 10:55:00 | 4447.05 | 4483.85 | 0.00 | ORB-short ORB[4480.30,4530.00] vol=3.2x ATR=14.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 11:00:00 | 4425.31 | 4475.33 | 0.00 | T1 1.5R @ 4425.31 |
| Stop hit — per-position SL triggered | 2024-06-28 13:15:00 | 4447.05 | 4447.55 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 4598.95 | 4581.84 | 0.00 | ORB-long ORB[4530.00,4594.50] vol=3.2x ATR=16.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:15:00 | 4623.78 | 4592.96 | 0.00 | T1 1.5R @ 4623.78 |
| Stop hit — per-position SL triggered | 2024-07-04 10:35:00 | 4598.95 | 4594.86 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-07-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:05:00 | 4600.00 | 4637.38 | 0.00 | ORB-short ORB[4622.95,4690.00] vol=1.6x ATR=14.01 |
| Stop hit — per-position SL triggered | 2024-07-09 10:10:00 | 4614.01 | 4634.30 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:25:00 | 4503.20 | 4587.40 | 0.00 | ORB-short ORB[4587.40,4651.00] vol=1.7x ATR=23.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 4467.24 | 4576.09 | 0.00 | T1 1.5R @ 4467.24 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 4503.20 | 4569.37 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 11:00:00 | 4366.15 | 4333.29 | 0.00 | ORB-long ORB[4307.05,4339.00] vol=2.2x ATR=10.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 11:20:00 | 4381.22 | 4342.95 | 0.00 | T1 1.5R @ 4381.22 |
| Stop hit — per-position SL triggered | 2024-07-31 12:25:00 | 4366.15 | 4379.07 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 4315.95 | 4293.08 | 0.00 | ORB-long ORB[4253.55,4302.00] vol=2.3x ATR=17.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:45:00 | 4341.94 | 4311.57 | 0.00 | T1 1.5R @ 4341.94 |
| Target hit | 2024-08-13 12:30:00 | 4331.45 | 4339.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-08-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:25:00 | 4517.05 | 4551.56 | 0.00 | ORB-short ORB[4528.20,4575.00] vol=1.6x ATR=12.95 |
| Stop hit — per-position SL triggered | 2024-08-29 10:30:00 | 4530.00 | 4537.21 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-09-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 11:00:00 | 4539.95 | 4583.07 | 0.00 | ORB-short ORB[4587.60,4647.50] vol=6.0x ATR=12.25 |
| Stop hit — per-position SL triggered | 2024-09-02 11:45:00 | 4552.20 | 4576.95 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 4481.25 | 4509.27 | 0.00 | ORB-short ORB[4501.10,4547.85] vol=1.7x ATR=13.66 |
| Stop hit — per-position SL triggered | 2024-09-03 09:50:00 | 4494.91 | 4501.11 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-09-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 09:55:00 | 4473.00 | 4489.84 | 0.00 | ORB-short ORB[4477.55,4518.80] vol=3.0x ATR=12.59 |
| Stop hit — per-position SL triggered | 2024-09-05 11:50:00 | 4485.59 | 4478.73 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-09-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:40:00 | 4453.10 | 4508.78 | 0.00 | ORB-short ORB[4515.40,4560.00] vol=2.2x ATR=18.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:05:00 | 4424.80 | 4498.30 | 0.00 | T1 1.5R @ 4424.80 |
| Target hit | 2024-09-19 15:20:00 | 4386.45 | 4440.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:30:00 | 4101.50 | 4070.27 | 0.00 | ORB-long ORB[4022.55,4081.00] vol=2.5x ATR=24.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:45:00 | 4138.32 | 4083.63 | 0.00 | T1 1.5R @ 4138.32 |
| Target hit | 2024-10-08 12:05:00 | 4128.55 | 4134.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2024-10-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 11:10:00 | 4100.85 | 4067.15 | 0.00 | ORB-long ORB[4041.05,4095.15] vol=1.7x ATR=13.56 |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 4087.29 | 4067.56 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-10-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:35:00 | 3904.10 | 3949.78 | 0.00 | ORB-short ORB[3954.00,4004.95] vol=2.3x ATR=13.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:45:00 | 3884.23 | 3918.46 | 0.00 | T1 1.5R @ 3884.23 |
| Target hit | 2024-10-25 12:15:00 | 3899.00 | 3867.45 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2024-10-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 11:00:00 | 3818.00 | 3781.57 | 0.00 | ORB-long ORB[3763.40,3806.35] vol=2.0x ATR=14.00 |
| Stop hit — per-position SL triggered | 2024-10-30 11:05:00 | 3804.00 | 3783.10 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-11-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:55:00 | 4056.10 | 4022.72 | 0.00 | ORB-long ORB[3989.15,4048.95] vol=1.6x ATR=18.66 |
| Stop hit — per-position SL triggered | 2024-11-07 11:00:00 | 4037.44 | 4033.92 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-11-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 09:45:00 | 4268.20 | 4290.84 | 0.00 | ORB-short ORB[4270.05,4329.50] vol=2.7x ATR=14.29 |
| Stop hit — per-position SL triggered | 2024-11-29 10:00:00 | 4282.49 | 4290.14 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-12-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:45:00 | 4515.95 | 4487.88 | 0.00 | ORB-long ORB[4450.00,4499.00] vol=1.8x ATR=16.38 |
| Stop hit — per-position SL triggered | 2024-12-06 09:50:00 | 4499.57 | 4489.57 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-12-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:35:00 | 4418.20 | 4437.40 | 0.00 | ORB-short ORB[4474.45,4519.00] vol=6.8x ATR=10.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 11:00:00 | 4402.31 | 4435.96 | 0.00 | T1 1.5R @ 4402.31 |
| Stop hit — per-position SL triggered | 2024-12-10 11:05:00 | 4418.20 | 4434.71 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-01-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:55:00 | 4465.00 | 4474.51 | 0.00 | ORB-short ORB[4480.70,4519.20] vol=2.0x ATR=12.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 12:05:00 | 4445.85 | 4470.39 | 0.00 | T1 1.5R @ 4445.85 |
| Target hit | 2025-01-03 15:20:00 | 4354.15 | 4427.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2025-01-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:30:00 | 4319.85 | 4337.47 | 0.00 | ORB-short ORB[4322.50,4369.65] vol=1.7x ATR=15.73 |
| Stop hit — per-position SL triggered | 2025-01-06 09:45:00 | 4335.58 | 4333.30 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-01-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 11:00:00 | 4291.80 | 4258.11 | 0.00 | ORB-long ORB[4212.05,4260.65] vol=1.7x ATR=14.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 13:05:00 | 4314.10 | 4272.21 | 0.00 | T1 1.5R @ 4314.10 |
| Target hit | 2025-01-07 15:20:00 | 4310.35 | 4289.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-01-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 11:10:00 | 4091.65 | 4118.26 | 0.00 | ORB-short ORB[4100.00,4143.45] vol=3.9x ATR=13.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 11:55:00 | 4071.04 | 4109.57 | 0.00 | T1 1.5R @ 4071.04 |
| Stop hit — per-position SL triggered | 2025-01-16 12:45:00 | 4091.65 | 4101.07 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-01-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:55:00 | 4053.65 | 4076.46 | 0.00 | ORB-short ORB[4056.00,4099.00] vol=1.8x ATR=11.22 |
| Stop hit — per-position SL triggered | 2025-01-17 11:00:00 | 4064.87 | 4074.79 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-01-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 11:10:00 | 4143.30 | 4119.42 | 0.00 | ORB-long ORB[4094.40,4138.00] vol=3.1x ATR=11.81 |
| Stop hit — per-position SL triggered | 2025-01-20 11:30:00 | 4131.49 | 4120.71 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-01-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:10:00 | 4072.90 | 4128.67 | 0.00 | ORB-short ORB[4130.15,4184.65] vol=1.7x ATR=25.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:25:00 | 4034.99 | 4108.75 | 0.00 | T1 1.5R @ 4034.99 |
| Target hit | 2025-01-27 15:20:00 | 3994.20 | 4039.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-01-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:30:00 | 3998.05 | 4022.24 | 0.00 | ORB-short ORB[4000.00,4049.00] vol=1.8x ATR=14.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:50:00 | 3976.54 | 4015.76 | 0.00 | T1 1.5R @ 3976.54 |
| Stop hit — per-position SL triggered | 2025-01-30 11:10:00 | 3998.05 | 4011.30 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-01-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:40:00 | 4055.60 | 4025.90 | 0.00 | ORB-long ORB[3963.00,4019.95] vol=2.1x ATR=21.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:10:00 | 4087.28 | 4031.56 | 0.00 | T1 1.5R @ 4087.28 |
| Stop hit — per-position SL triggered | 2025-01-31 13:40:00 | 4055.60 | 4046.15 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-02-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 09:30:00 | 3861.00 | 3894.60 | 0.00 | ORB-short ORB[3868.05,3920.00] vol=2.0x ATR=15.82 |
| Stop hit — per-position SL triggered | 2025-02-04 09:40:00 | 3876.82 | 3890.14 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-02-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:45:00 | 3940.00 | 3906.29 | 0.00 | ORB-long ORB[3878.05,3925.10] vol=1.7x ATR=13.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 11:00:00 | 3960.84 | 3910.24 | 0.00 | T1 1.5R @ 3960.84 |
| Stop hit — per-position SL triggered | 2025-02-05 11:20:00 | 3940.00 | 3914.53 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-02-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 09:35:00 | 3468.30 | 3445.36 | 0.00 | ORB-long ORB[3410.00,3451.95] vol=1.7x ATR=17.28 |
| Stop hit — per-position SL triggered | 2025-02-14 09:40:00 | 3451.02 | 3447.68 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:40:00 | 3228.20 | 3412.81 | 0.00 | ORB-short ORB[3418.10,3450.00] vol=1.5x ATR=37.19 |
| Stop hit — per-position SL triggered | 2025-02-27 09:45:00 | 3265.39 | 3308.14 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-03-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 09:45:00 | 3233.45 | 3251.63 | 0.00 | ORB-short ORB[3240.00,3285.00] vol=1.8x ATR=15.44 |
| Stop hit — per-position SL triggered | 2025-03-07 10:45:00 | 3248.89 | 3242.58 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:55:00 | 3012.35 | 2986.03 | 0.00 | ORB-long ORB[2963.05,3006.30] vol=2.4x ATR=16.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 11:00:00 | 3037.11 | 2987.20 | 0.00 | T1 1.5R @ 3037.11 |
| Stop hit — per-position SL triggered | 2025-03-11 11:20:00 | 3012.35 | 2990.04 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-03-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:25:00 | 2859.65 | 2836.00 | 0.00 | ORB-long ORB[2805.30,2845.00] vol=1.7x ATR=11.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 10:40:00 | 2876.32 | 2846.68 | 0.00 | T1 1.5R @ 2876.32 |
| Stop hit — per-position SL triggered | 2025-03-27 14:30:00 | 2859.65 | 2860.57 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-03-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 10:50:00 | 2939.65 | 2918.73 | 0.00 | ORB-long ORB[2901.15,2926.85] vol=1.7x ATR=10.60 |
| Stop hit — per-position SL triggered | 2025-03-28 11:10:00 | 2929.05 | 2921.01 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 2954.90 | 2978.16 | 0.00 | ORB-short ORB[2967.00,3009.50] vol=2.4x ATR=12.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:55:00 | 2936.11 | 2963.90 | 0.00 | T1 1.5R @ 2936.11 |
| Target hit | 2025-04-25 12:15:00 | 2941.80 | 2937.86 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-06-28 10:55:00 | 4447.05 | 2024-06-28 11:00:00 | 4425.31 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-06-28 10:55:00 | 4447.05 | 2024-06-28 13:15:00 | 4447.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 09:30:00 | 4598.95 | 2024-07-04 10:15:00 | 4623.78 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-07-04 09:30:00 | 4598.95 | 2024-07-04 10:35:00 | 4598.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-09 10:05:00 | 4600.00 | 2024-07-09 10:10:00 | 4614.01 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-10 10:25:00 | 4503.20 | 2024-07-10 10:35:00 | 4467.24 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2024-07-10 10:25:00 | 4503.20 | 2024-07-10 10:40:00 | 4503.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 11:00:00 | 4366.15 | 2024-07-31 11:20:00 | 4381.22 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-07-31 11:00:00 | 4366.15 | 2024-07-31 12:25:00 | 4366.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-13 09:35:00 | 4315.95 | 2024-08-13 09:45:00 | 4341.94 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-08-13 09:35:00 | 4315.95 | 2024-08-13 12:30:00 | 4331.45 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-29 10:25:00 | 4517.05 | 2024-08-29 10:30:00 | 4530.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-02 11:00:00 | 4539.95 | 2024-09-02 11:45:00 | 4552.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-03 09:35:00 | 4481.25 | 2024-09-03 09:50:00 | 4494.91 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-05 09:55:00 | 4473.00 | 2024-09-05 11:50:00 | 4485.59 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-19 10:40:00 | 4453.10 | 2024-09-19 11:05:00 | 4424.80 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-09-19 10:40:00 | 4453.10 | 2024-09-19 15:20:00 | 4386.45 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2024-10-08 09:30:00 | 4101.50 | 2024-10-08 09:45:00 | 4138.32 | PARTIAL | 0.50 | 0.90% |
| BUY | retest1 | 2024-10-08 09:30:00 | 4101.50 | 2024-10-08 12:05:00 | 4128.55 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2024-10-21 11:10:00 | 4100.85 | 2024-10-21 11:15:00 | 4087.29 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-25 09:35:00 | 3904.10 | 2024-10-25 09:45:00 | 3884.23 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-25 09:35:00 | 3904.10 | 2024-10-25 12:15:00 | 3899.00 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-10-30 11:00:00 | 3818.00 | 2024-10-30 11:05:00 | 3804.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-07 09:55:00 | 4056.10 | 2024-11-07 11:00:00 | 4037.44 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-11-29 09:45:00 | 4268.20 | 2024-11-29 10:00:00 | 4282.49 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-06 09:45:00 | 4515.95 | 2024-12-06 09:50:00 | 4499.57 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-10 10:35:00 | 4418.20 | 2024-12-10 11:00:00 | 4402.31 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-10 10:35:00 | 4418.20 | 2024-12-10 11:05:00 | 4418.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 10:55:00 | 4465.00 | 2025-01-03 12:05:00 | 4445.85 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-03 10:55:00 | 4465.00 | 2025-01-03 15:20:00 | 4354.15 | TARGET_HIT | 0.50 | 2.48% |
| SELL | retest1 | 2025-01-06 09:30:00 | 4319.85 | 2025-01-06 09:45:00 | 4335.58 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-07 11:00:00 | 4291.80 | 2025-01-07 13:05:00 | 4314.10 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-01-07 11:00:00 | 4291.80 | 2025-01-07 15:20:00 | 4310.35 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-16 11:10:00 | 4091.65 | 2025-01-16 11:55:00 | 4071.04 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-01-16 11:10:00 | 4091.65 | 2025-01-16 12:45:00 | 4091.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-17 10:55:00 | 4053.65 | 2025-01-17 11:00:00 | 4064.87 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-20 11:10:00 | 4143.30 | 2025-01-20 11:30:00 | 4131.49 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-27 10:10:00 | 4072.90 | 2025-01-27 10:25:00 | 4034.99 | PARTIAL | 0.50 | 0.93% |
| SELL | retest1 | 2025-01-27 10:10:00 | 4072.90 | 2025-01-27 15:20:00 | 3994.20 | TARGET_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2025-01-30 10:30:00 | 3998.05 | 2025-01-30 10:50:00 | 3976.54 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-01-30 10:30:00 | 3998.05 | 2025-01-30 11:10:00 | 3998.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 10:40:00 | 4055.60 | 2025-01-31 11:10:00 | 4087.28 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2025-01-31 10:40:00 | 4055.60 | 2025-01-31 13:40:00 | 4055.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-04 09:30:00 | 3861.00 | 2025-02-04 09:40:00 | 3876.82 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-02-05 10:45:00 | 3940.00 | 2025-02-05 11:00:00 | 3960.84 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-02-05 10:45:00 | 3940.00 | 2025-02-05 11:20:00 | 3940.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-14 09:35:00 | 3468.30 | 2025-02-14 09:40:00 | 3451.02 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-02-27 09:40:00 | 3228.20 | 2025-02-27 09:45:00 | 3265.39 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest1 | 2025-03-07 09:45:00 | 3233.45 | 2025-03-07 10:45:00 | 3248.89 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-03-11 10:55:00 | 3012.35 | 2025-03-11 11:00:00 | 3037.11 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2025-03-11 10:55:00 | 3012.35 | 2025-03-11 11:20:00 | 3012.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-27 10:25:00 | 2859.65 | 2025-03-27 10:40:00 | 2876.32 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-03-27 10:25:00 | 2859.65 | 2025-03-27 14:30:00 | 2859.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-28 10:50:00 | 2939.65 | 2025-03-28 11:10:00 | 2929.05 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-04-25 09:35:00 | 2954.90 | 2025-04-25 09:55:00 | 2936.11 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-04-25 09:35:00 | 2954.90 | 2025-04-25 12:15:00 | 2941.80 | TARGET_HIT | 0.50 | 0.44% |
