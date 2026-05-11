# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-01-05 15:25:00 (12237 bars)
- **Last close:** 4997.00
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
| ENTRY1 | 54 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 16 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 38
- **Target hits / Stop hits / Partials:** 16 / 38 / 23
- **Avg / median % per leg:** 0.23% / 0.05%
- **Sum % (uncompounded):** 17.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 23 | 54.8% | 9 | 19 | 14 | 0.33% | 14.1% |
| BUY @ 2nd Alert (retest1) | 42 | 23 | 54.8% | 9 | 19 | 14 | 0.33% | 14.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 16 | 45.7% | 7 | 19 | 9 | 0.11% | 3.9% |
| SELL @ 2nd Alert (retest1) | 35 | 16 | 45.7% | 7 | 19 | 9 | 0.11% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 77 | 39 | 50.6% | 16 | 38 | 23 | 0.23% | 18.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 10:40:00 | 3472.45 | 3419.11 | 0.00 | ORB-long ORB[3351.00,3400.15] vol=1.7x ATR=15.70 |
| Stop hit — per-position SL triggered | 2023-05-12 11:20:00 | 3456.75 | 3426.66 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:35:00 | 3510.10 | 3492.17 | 0.00 | ORB-long ORB[3451.70,3500.00] vol=1.8x ATR=11.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 09:45:00 | 3527.86 | 3501.89 | 0.00 | T1 1.5R @ 3527.86 |
| Stop hit — per-position SL triggered | 2023-05-15 10:05:00 | 3510.10 | 3505.60 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-16 11:15:00 | 3489.90 | 3494.99 | 0.00 | ORB-short ORB[3494.05,3536.55] vol=1.7x ATR=9.39 |
| Stop hit — per-position SL triggered | 2023-05-16 11:50:00 | 3499.29 | 3494.99 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:20:00 | 3530.40 | 3513.88 | 0.00 | ORB-long ORB[3466.05,3507.95] vol=1.7x ATR=13.20 |
| Stop hit — per-position SL triggered | 2023-05-22 11:20:00 | 3517.20 | 3516.34 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:30:00 | 3942.05 | 3927.61 | 0.00 | ORB-long ORB[3892.00,3942.00] vol=3.0x ATR=18.84 |
| Stop hit — per-position SL triggered | 2023-06-06 10:10:00 | 3923.21 | 3924.26 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 09:45:00 | 3925.00 | 3890.32 | 0.00 | ORB-long ORB[3860.05,3898.65] vol=1.8x ATR=16.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 10:15:00 | 3949.25 | 3916.42 | 0.00 | T1 1.5R @ 3949.25 |
| Target hit | 2023-06-12 11:30:00 | 3934.95 | 3937.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2023-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 09:30:00 | 3949.50 | 3930.20 | 0.00 | ORB-long ORB[3892.55,3945.00] vol=3.4x ATR=13.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 09:40:00 | 3969.42 | 3939.59 | 0.00 | T1 1.5R @ 3969.42 |
| Target hit | 2023-06-14 10:40:00 | 4006.35 | 4008.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2023-06-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 11:10:00 | 4048.10 | 4064.66 | 0.00 | ORB-short ORB[4054.05,4100.00] vol=3.7x ATR=11.90 |
| Target hit | 2023-06-28 15:20:00 | 4045.00 | 4053.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2023-07-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 10:10:00 | 4023.50 | 4034.02 | 0.00 | ORB-short ORB[4030.00,4068.35] vol=1.9x ATR=14.81 |
| Stop hit — per-position SL triggered | 2023-07-03 10:35:00 | 4038.31 | 4032.46 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-07-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:45:00 | 4008.15 | 3980.83 | 0.00 | ORB-long ORB[3955.05,4004.90] vol=2.1x ATR=16.27 |
| Stop hit — per-position SL triggered | 2023-07-06 12:00:00 | 3991.88 | 3985.56 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-07-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 11:00:00 | 3945.95 | 3966.50 | 0.00 | ORB-short ORB[3960.00,3989.00] vol=2.0x ATR=12.02 |
| Stop hit — per-position SL triggered | 2023-07-07 11:05:00 | 3957.97 | 3966.20 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-07-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 10:05:00 | 4067.85 | 4048.01 | 0.00 | ORB-long ORB[4027.70,4059.00] vol=3.9x ATR=14.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 10:10:00 | 4088.89 | 4061.22 | 0.00 | T1 1.5R @ 4088.89 |
| Target hit | 2023-07-12 14:05:00 | 4098.80 | 4113.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2023-07-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 10:25:00 | 4585.00 | 4564.48 | 0.00 | ORB-long ORB[4540.00,4584.35] vol=1.7x ATR=14.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 10:35:00 | 4607.09 | 4570.53 | 0.00 | T1 1.5R @ 4607.09 |
| Target hit | 2023-07-21 11:10:00 | 4591.30 | 4596.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2023-07-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 09:55:00 | 4613.00 | 4569.87 | 0.00 | ORB-long ORB[4525.30,4593.30] vol=3.7x ATR=22.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 10:05:00 | 4646.96 | 4609.51 | 0.00 | T1 1.5R @ 4646.96 |
| Target hit | 2023-07-24 11:40:00 | 4645.50 | 4649.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2023-08-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 10:50:00 | 4684.05 | 4666.61 | 0.00 | ORB-long ORB[4627.05,4678.50] vol=3.9x ATR=14.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 10:55:00 | 4706.23 | 4689.01 | 0.00 | T1 1.5R @ 4706.23 |
| Target hit | 2023-08-02 11:30:00 | 4696.05 | 4724.12 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2023-08-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 10:50:00 | 4577.10 | 4603.62 | 0.00 | ORB-short ORB[4600.15,4659.00] vol=2.8x ATR=16.13 |
| Stop hit — per-position SL triggered | 2023-08-03 11:30:00 | 4593.23 | 4601.47 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-08-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 11:05:00 | 4617.10 | 4609.13 | 0.00 | ORB-long ORB[4568.00,4615.00] vol=3.3x ATR=14.65 |
| Stop hit — per-position SL triggered | 2023-08-04 11:15:00 | 4602.45 | 4610.60 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-08-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 10:00:00 | 4710.00 | 4680.27 | 0.00 | ORB-long ORB[4647.05,4694.05] vol=1.9x ATR=13.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 10:10:00 | 4730.98 | 4691.25 | 0.00 | T1 1.5R @ 4730.98 |
| Stop hit — per-position SL triggered | 2023-08-07 10:15:00 | 4710.00 | 4692.87 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-08-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 10:00:00 | 4771.50 | 4733.73 | 0.00 | ORB-long ORB[4689.10,4721.25] vol=5.5x ATR=16.03 |
| Stop hit — per-position SL triggered | 2023-08-08 10:05:00 | 4755.47 | 4737.06 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-08-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 09:40:00 | 4776.05 | 4741.44 | 0.00 | ORB-long ORB[4710.00,4774.95] vol=2.8x ATR=19.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-14 09:50:00 | 4805.01 | 4751.05 | 0.00 | T1 1.5R @ 4805.01 |
| Stop hit — per-position SL triggered | 2023-08-14 10:40:00 | 4776.05 | 4778.42 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 09:35:00 | 4845.10 | 4825.04 | 0.00 | ORB-long ORB[4775.70,4830.00] vol=1.5x ATR=16.75 |
| Stop hit — per-position SL triggered | 2023-08-16 09:40:00 | 4828.35 | 4829.77 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 10:40:00 | 4920.40 | 4886.22 | 0.00 | ORB-long ORB[4859.95,4915.00] vol=2.1x ATR=14.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 10:55:00 | 4942.29 | 4895.92 | 0.00 | T1 1.5R @ 4942.29 |
| Stop hit — per-position SL triggered | 2023-08-18 11:45:00 | 4920.40 | 4912.59 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 10:50:00 | 4968.75 | 4922.16 | 0.00 | ORB-long ORB[4888.10,4943.95] vol=4.5x ATR=17.75 |
| Stop hit — per-position SL triggered | 2023-08-23 10:55:00 | 4951.00 | 4924.86 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 09:35:00 | 4923.60 | 4940.87 | 0.00 | ORB-short ORB[4925.20,4985.10] vol=4.0x ATR=18.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 09:50:00 | 4896.00 | 4935.00 | 0.00 | T1 1.5R @ 4896.00 |
| Target hit | 2023-08-25 15:05:00 | 4905.80 | 4900.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — SELL (started 2023-08-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 10:10:00 | 4852.75 | 4861.43 | 0.00 | ORB-short ORB[4856.00,4886.50] vol=2.2x ATR=11.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 13:40:00 | 4835.47 | 4851.15 | 0.00 | T1 1.5R @ 4835.47 |
| Stop hit — per-position SL triggered | 2023-08-29 14:50:00 | 4852.75 | 4849.90 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 10:40:00 | 4824.65 | 4847.57 | 0.00 | ORB-short ORB[4844.20,4891.35] vol=8.5x ATR=14.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 11:15:00 | 4802.95 | 4846.33 | 0.00 | T1 1.5R @ 4802.95 |
| Stop hit — per-position SL triggered | 2023-08-30 13:20:00 | 4824.65 | 4838.89 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 09:40:00 | 4827.40 | 4836.81 | 0.00 | ORB-short ORB[4840.60,4873.05] vol=5.4x ATR=16.20 |
| Stop hit — per-position SL triggered | 2023-09-05 09:50:00 | 4843.60 | 4836.58 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-09-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 09:35:00 | 4966.50 | 4953.43 | 0.00 | ORB-long ORB[4899.15,4964.85] vol=4.4x ATR=18.22 |
| Stop hit — per-position SL triggered | 2023-09-07 09:45:00 | 4948.28 | 4954.10 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:30:00 | 4909.70 | 4901.02 | 0.00 | ORB-long ORB[4850.05,4908.95] vol=3.8x ATR=16.04 |
| Stop hit — per-position SL triggered | 2023-09-08 09:35:00 | 4893.66 | 4900.18 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 09:55:00 | 4743.85 | 4777.25 | 0.00 | ORB-short ORB[4768.70,4815.15] vol=1.9x ATR=15.30 |
| Stop hit — per-position SL triggered | 2023-09-20 10:20:00 | 4759.15 | 4765.28 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 10:50:00 | 4723.00 | 4764.66 | 0.00 | ORB-short ORB[4758.25,4818.30] vol=1.5x ATR=16.36 |
| Stop hit — per-position SL triggered | 2023-09-21 11:50:00 | 4739.36 | 4754.30 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 10:30:00 | 4669.95 | 4678.58 | 0.00 | ORB-short ORB[4670.00,4725.00] vol=3.4x ATR=13.19 |
| Stop hit — per-position SL triggered | 2023-09-27 11:05:00 | 4683.14 | 4677.59 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:10:00 | 4809.10 | 4782.81 | 0.00 | ORB-long ORB[4730.15,4783.00] vol=5.3x ATR=16.34 |
| Stop hit — per-position SL triggered | 2023-09-29 10:20:00 | 4792.76 | 4783.59 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 10:15:00 | 4625.50 | 4631.19 | 0.00 | ORB-short ORB[4632.00,4692.15] vol=8.5x ATR=13.78 |
| Target hit | 2023-10-06 15:20:00 | 4623.25 | 4627.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2023-10-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 09:30:00 | 4565.00 | 4543.59 | 0.00 | ORB-long ORB[4520.65,4564.95] vol=1.7x ATR=17.28 |
| Stop hit — per-position SL triggered | 2023-10-12 10:05:00 | 4547.72 | 4552.37 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-10-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:35:00 | 4546.00 | 4562.05 | 0.00 | ORB-short ORB[4560.00,4597.75] vol=2.1x ATR=12.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 11:10:00 | 4526.88 | 4557.21 | 0.00 | T1 1.5R @ 4526.88 |
| Stop hit — per-position SL triggered | 2023-10-13 13:35:00 | 4546.00 | 4541.77 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:30:00 | 4533.00 | 4553.47 | 0.00 | ORB-short ORB[4552.55,4593.00] vol=1.6x ATR=12.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:10:00 | 4514.36 | 4545.21 | 0.00 | T1 1.5R @ 4514.36 |
| Target hit | 2023-10-18 15:15:00 | 4510.00 | 4507.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — BUY (started 2023-10-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 09:50:00 | 4542.70 | 4531.95 | 0.00 | ORB-long ORB[4460.00,4520.00] vol=15.3x ATR=13.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 09:55:00 | 4563.09 | 4532.35 | 0.00 | T1 1.5R @ 4563.09 |
| Stop hit — per-position SL triggered | 2023-10-19 11:35:00 | 4542.70 | 4537.03 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 09:30:00 | 4515.65 | 4528.11 | 0.00 | ORB-short ORB[4518.15,4555.00] vol=1.8x ATR=16.11 |
| Stop hit — per-position SL triggered | 2023-10-23 09:45:00 | 4531.76 | 4528.96 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-25 10:00:00 | 4439.95 | 4505.56 | 0.00 | ORB-short ORB[4510.00,4550.00] vol=2.0x ATR=20.65 |
| Stop hit — per-position SL triggered | 2023-10-25 10:10:00 | 4460.60 | 4503.16 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-10-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 11:00:00 | 4466.95 | 4427.13 | 0.00 | ORB-long ORB[4402.25,4444.70] vol=3.8x ATR=11.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-27 11:15:00 | 4484.91 | 4439.33 | 0.00 | T1 1.5R @ 4484.91 |
| Target hit | 2023-10-27 15:20:00 | 4644.45 | 4556.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2023-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 09:35:00 | 4926.90 | 4885.50 | 0.00 | ORB-long ORB[4825.00,4898.00] vol=2.0x ATR=47.85 |
| Stop hit — per-position SL triggered | 2023-10-31 09:40:00 | 4879.05 | 4887.18 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:15:00 | 4884.10 | 4893.52 | 0.00 | ORB-short ORB[4888.05,4920.20] vol=1.5x ATR=12.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 10:30:00 | 4865.13 | 4886.77 | 0.00 | T1 1.5R @ 4865.13 |
| Target hit | 2023-11-13 15:20:00 | 4787.70 | 4825.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2023-11-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 09:45:00 | 4891.20 | 4871.25 | 0.00 | ORB-long ORB[4826.00,4876.35] vol=2.0x ATR=18.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 10:40:00 | 4918.50 | 4886.50 | 0.00 | T1 1.5R @ 4918.50 |
| Target hit | 2023-11-16 15:20:00 | 5081.50 | 5064.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2023-11-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:35:00 | 5164.20 | 5187.39 | 0.00 | ORB-short ORB[5175.00,5230.00] vol=1.6x ATR=13.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 09:55:00 | 5143.75 | 5170.18 | 0.00 | T1 1.5R @ 5143.75 |
| Target hit | 2023-11-24 15:00:00 | 5130.05 | 5127.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — SELL (started 2023-12-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-05 10:00:00 | 5102.60 | 5143.94 | 0.00 | ORB-short ORB[5115.10,5168.45] vol=3.1x ATR=19.20 |
| Stop hit — per-position SL triggered | 2023-12-05 10:10:00 | 5121.80 | 5141.56 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:15:00 | 5137.80 | 5183.22 | 0.00 | ORB-short ORB[5184.75,5241.85] vol=1.8x ATR=14.98 |
| Stop hit — per-position SL triggered | 2023-12-08 11:20:00 | 5152.78 | 5180.85 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:20:00 | 5139.90 | 5171.06 | 0.00 | ORB-short ORB[5157.05,5200.00] vol=2.0x ATR=12.44 |
| Stop hit — per-position SL triggered | 2023-12-12 10:25:00 | 5152.34 | 5167.81 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-12-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 11:00:00 | 5357.85 | 5326.75 | 0.00 | ORB-long ORB[5280.20,5350.00] vol=6.2x ATR=15.25 |
| Stop hit — per-position SL triggered | 2023-12-18 13:05:00 | 5342.60 | 5339.84 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-12-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 10:00:00 | 5370.00 | 5382.26 | 0.00 | ORB-short ORB[5379.15,5429.15] vol=9.7x ATR=18.03 |
| Stop hit — per-position SL triggered | 2023-12-26 11:40:00 | 5388.03 | 5389.86 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-12-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 10:40:00 | 5371.75 | 5376.00 | 0.00 | ORB-short ORB[5393.65,5432.00] vol=8.5x ATR=15.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 12:30:00 | 5348.05 | 5373.33 | 0.00 | T1 1.5R @ 5348.05 |
| Target hit | 2023-12-27 15:20:00 | 5307.15 | 5356.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2023-12-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 11:10:00 | 5337.25 | 5314.64 | 0.00 | ORB-long ORB[5253.20,5330.15] vol=1.9x ATR=14.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 11:35:00 | 5359.37 | 5318.47 | 0.00 | T1 1.5R @ 5359.37 |
| Target hit | 2023-12-29 15:20:00 | 5401.95 | 5373.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2024-01-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 10:35:00 | 5365.25 | 5392.23 | 0.00 | ORB-short ORB[5378.15,5441.45] vol=1.8x ATR=11.59 |
| Stop hit — per-position SL triggered | 2024-01-01 10:40:00 | 5376.84 | 5391.49 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 5285.75 | 5318.27 | 0.00 | ORB-short ORB[5321.00,5366.05] vol=2.7x ATR=19.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 11:45:00 | 5256.93 | 5286.23 | 0.00 | T1 1.5R @ 5256.93 |
| Stop hit — per-position SL triggered | 2024-01-02 11:55:00 | 5285.75 | 5284.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 10:40:00 | 3472.45 | 2023-05-12 11:20:00 | 3456.75 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-05-15 09:35:00 | 3510.10 | 2023-05-15 09:45:00 | 3527.86 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-05-15 09:35:00 | 3510.10 | 2023-05-15 10:05:00 | 3510.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-16 11:15:00 | 3489.90 | 2023-05-16 11:50:00 | 3499.29 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-05-22 10:20:00 | 3530.40 | 2023-05-22 11:20:00 | 3517.20 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-06-06 09:30:00 | 3942.05 | 2023-06-06 10:10:00 | 3923.21 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2023-06-12 09:45:00 | 3925.00 | 2023-06-12 10:15:00 | 3949.25 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2023-06-12 09:45:00 | 3925.00 | 2023-06-12 11:30:00 | 3934.95 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2023-06-14 09:30:00 | 3949.50 | 2023-06-14 09:40:00 | 3969.42 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-06-14 09:30:00 | 3949.50 | 2023-06-14 10:40:00 | 4006.35 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2023-06-28 11:10:00 | 4048.10 | 2023-06-28 15:20:00 | 4045.00 | TARGET_HIT | 1.00 | 0.08% |
| SELL | retest1 | 2023-07-03 10:10:00 | 4023.50 | 2023-07-03 10:35:00 | 4038.31 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-07-06 09:45:00 | 4008.15 | 2023-07-06 12:00:00 | 3991.88 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-07-07 11:00:00 | 3945.95 | 2023-07-07 11:05:00 | 3957.97 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-07-12 10:05:00 | 4067.85 | 2023-07-12 10:10:00 | 4088.89 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-12 10:05:00 | 4067.85 | 2023-07-12 14:05:00 | 4098.80 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2023-07-21 10:25:00 | 4585.00 | 2023-07-21 10:35:00 | 4607.09 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-07-21 10:25:00 | 4585.00 | 2023-07-21 11:10:00 | 4591.30 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2023-07-24 09:55:00 | 4613.00 | 2023-07-24 10:05:00 | 4646.96 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2023-07-24 09:55:00 | 4613.00 | 2023-07-24 11:40:00 | 4645.50 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2023-08-02 10:50:00 | 4684.05 | 2023-08-02 10:55:00 | 4706.23 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-08-02 10:50:00 | 4684.05 | 2023-08-02 11:30:00 | 4696.05 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2023-08-03 10:50:00 | 4577.10 | 2023-08-03 11:30:00 | 4593.23 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-08-04 11:05:00 | 4617.10 | 2023-08-04 11:15:00 | 4602.45 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-08-07 10:00:00 | 4710.00 | 2023-08-07 10:10:00 | 4730.98 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-08-07 10:00:00 | 4710.00 | 2023-08-07 10:15:00 | 4710.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-08 10:00:00 | 4771.50 | 2023-08-08 10:05:00 | 4755.47 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-08-14 09:40:00 | 4776.05 | 2023-08-14 09:50:00 | 4805.01 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2023-08-14 09:40:00 | 4776.05 | 2023-08-14 10:40:00 | 4776.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-16 09:35:00 | 4845.10 | 2023-08-16 09:40:00 | 4828.35 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-08-18 10:40:00 | 4920.40 | 2023-08-18 10:55:00 | 4942.29 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-08-18 10:40:00 | 4920.40 | 2023-08-18 11:45:00 | 4920.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-23 10:50:00 | 4968.75 | 2023-08-23 10:55:00 | 4951.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-08-25 09:35:00 | 4923.60 | 2023-08-25 09:50:00 | 4896.00 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-08-25 09:35:00 | 4923.60 | 2023-08-25 15:05:00 | 4905.80 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2023-08-29 10:10:00 | 4852.75 | 2023-08-29 13:40:00 | 4835.47 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-08-29 10:10:00 | 4852.75 | 2023-08-29 14:50:00 | 4852.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-30 10:40:00 | 4824.65 | 2023-08-30 11:15:00 | 4802.95 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-08-30 10:40:00 | 4824.65 | 2023-08-30 13:20:00 | 4824.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-05 09:40:00 | 4827.40 | 2023-09-05 09:50:00 | 4843.60 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-09-07 09:35:00 | 4966.50 | 2023-09-07 09:45:00 | 4948.28 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-09-08 09:30:00 | 4909.70 | 2023-09-08 09:35:00 | 4893.66 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-09-20 09:55:00 | 4743.85 | 2023-09-20 10:20:00 | 4759.15 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-09-21 10:50:00 | 4723.00 | 2023-09-21 11:50:00 | 4739.36 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-09-27 10:30:00 | 4669.95 | 2023-09-27 11:05:00 | 4683.14 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-09-29 10:10:00 | 4809.10 | 2023-09-29 10:20:00 | 4792.76 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-10-06 10:15:00 | 4625.50 | 2023-10-06 15:20:00 | 4623.25 | TARGET_HIT | 1.00 | 0.05% |
| BUY | retest1 | 2023-10-12 09:30:00 | 4565.00 | 2023-10-12 10:05:00 | 4547.72 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-10-13 10:35:00 | 4546.00 | 2023-10-13 11:10:00 | 4526.88 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-10-13 10:35:00 | 4546.00 | 2023-10-13 13:35:00 | 4546.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-18 10:30:00 | 4533.00 | 2023-10-18 11:10:00 | 4514.36 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-10-18 10:30:00 | 4533.00 | 2023-10-18 15:15:00 | 4510.00 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2023-10-19 09:50:00 | 4542.70 | 2023-10-19 09:55:00 | 4563.09 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-10-19 09:50:00 | 4542.70 | 2023-10-19 11:35:00 | 4542.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-23 09:30:00 | 4515.65 | 2023-10-23 09:45:00 | 4531.76 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-10-25 10:00:00 | 4439.95 | 2023-10-25 10:10:00 | 4460.60 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-10-27 11:00:00 | 4466.95 | 2023-10-27 11:15:00 | 4484.91 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-10-27 11:00:00 | 4466.95 | 2023-10-27 15:20:00 | 4644.45 | TARGET_HIT | 0.50 | 3.97% |
| BUY | retest1 | 2023-10-31 09:35:00 | 4926.90 | 2023-10-31 09:40:00 | 4879.05 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest1 | 2023-11-13 10:15:00 | 4884.10 | 2023-11-13 10:30:00 | 4865.13 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-11-13 10:15:00 | 4884.10 | 2023-11-13 15:20:00 | 4787.70 | TARGET_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2023-11-16 09:45:00 | 4891.20 | 2023-11-16 10:40:00 | 4918.50 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-11-16 09:45:00 | 4891.20 | 2023-11-16 15:20:00 | 5081.50 | TARGET_HIT | 0.50 | 3.89% |
| SELL | retest1 | 2023-11-24 09:35:00 | 5164.20 | 2023-11-24 09:55:00 | 5143.75 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-11-24 09:35:00 | 5164.20 | 2023-11-24 15:00:00 | 5130.05 | TARGET_HIT | 0.50 | 0.66% |
| SELL | retest1 | 2023-12-05 10:00:00 | 5102.60 | 2023-12-05 10:10:00 | 5121.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-12-08 11:15:00 | 5137.80 | 2023-12-08 11:20:00 | 5152.78 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-12-12 10:20:00 | 5139.90 | 2023-12-12 10:25:00 | 5152.34 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-18 11:00:00 | 5357.85 | 2023-12-18 13:05:00 | 5342.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-26 10:00:00 | 5370.00 | 2023-12-26 11:40:00 | 5388.03 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-12-27 10:40:00 | 5371.75 | 2023-12-27 12:30:00 | 5348.05 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-12-27 10:40:00 | 5371.75 | 2023-12-27 15:20:00 | 5307.15 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2023-12-29 11:10:00 | 5337.25 | 2023-12-29 11:35:00 | 5359.37 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-12-29 11:10:00 | 5337.25 | 2023-12-29 15:20:00 | 5401.95 | TARGET_HIT | 0.50 | 1.21% |
| SELL | retest1 | 2024-01-01 10:35:00 | 5365.25 | 2024-01-01 10:40:00 | 5376.84 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-01-02 09:55:00 | 5285.75 | 2024-01-02 11:45:00 | 5256.93 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-01-02 09:55:00 | 5285.75 | 2024-01-02 11:55:00 | 5285.75 | STOP_HIT | 0.50 | 0.00% |
