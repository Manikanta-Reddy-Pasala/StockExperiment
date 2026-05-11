# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
- **Last close:** 5516.00
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
| ENTRY1 | 90 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 13 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 77
- **Target hits / Stop hits / Partials:** 13 / 77 / 38
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 12.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 27 | 43.5% | 6 | 35 | 21 | 0.13% | 8.4% |
| BUY @ 2nd Alert (retest1) | 62 | 27 | 43.5% | 6 | 35 | 21 | 0.13% | 8.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 24 | 36.4% | 7 | 42 | 17 | 0.07% | 4.4% |
| SELL @ 2nd Alert (retest1) | 66 | 24 | 36.4% | 7 | 42 | 17 | 0.07% | 4.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 128 | 51 | 39.8% | 13 | 77 | 38 | 0.10% | 12.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 10:10:00 | 4503.00 | 4533.71 | 0.00 | ORB-short ORB[4530.70,4593.10] vol=1.7x ATR=10.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:40:00 | 4487.85 | 4517.45 | 0.00 | T1 1.5R @ 4487.85 |
| Stop hit — per-position SL triggered | 2023-05-19 11:05:00 | 4503.00 | 4514.86 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 11:10:00 | 4613.65 | 4598.68 | 0.00 | ORB-long ORB[4563.00,4602.40] vol=1.6x ATR=7.51 |
| Stop hit — per-position SL triggered | 2023-05-30 11:45:00 | 4606.14 | 4600.53 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-06-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:55:00 | 4935.00 | 4927.05 | 0.00 | ORB-long ORB[4892.30,4934.95] vol=2.2x ATR=8.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:15:00 | 4947.93 | 4928.94 | 0.00 | T1 1.5R @ 4947.93 |
| Stop hit — per-position SL triggered | 2023-06-13 10:25:00 | 4935.00 | 4930.77 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-06-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:30:00 | 4968.30 | 4955.48 | 0.00 | ORB-long ORB[4939.00,4960.00] vol=2.4x ATR=8.14 |
| Stop hit — per-position SL triggered | 2023-06-14 10:40:00 | 4960.16 | 4956.01 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 5019.00 | 5029.08 | 0.00 | ORB-short ORB[5021.05,5045.05] vol=5.1x ATR=9.15 |
| Stop hit — per-position SL triggered | 2023-06-20 09:35:00 | 5028.15 | 5028.39 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 11:15:00 | 4954.00 | 4962.14 | 0.00 | ORB-short ORB[4960.55,4985.85] vol=1.6x ATR=7.09 |
| Stop hit — per-position SL triggered | 2023-06-23 11:40:00 | 4961.09 | 4961.50 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 11:15:00 | 4997.15 | 4979.68 | 0.00 | ORB-long ORB[4962.10,4996.95] vol=3.4x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 11:35:00 | 5006.97 | 4982.63 | 0.00 | T1 1.5R @ 5006.97 |
| Stop hit — per-position SL triggered | 2023-06-28 13:10:00 | 4997.15 | 4994.91 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-07-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:35:00 | 5038.00 | 5017.70 | 0.00 | ORB-long ORB[4960.50,5024.80] vol=2.9x ATR=9.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 11:10:00 | 5051.94 | 5025.72 | 0.00 | T1 1.5R @ 5051.94 |
| Target hit | 2023-07-05 15:20:00 | 5107.45 | 5055.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2023-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:30:00 | 5136.80 | 5091.50 | 0.00 | ORB-long ORB[5050.00,5097.45] vol=2.5x ATR=15.56 |
| Stop hit — per-position SL triggered | 2023-07-11 10:05:00 | 5121.24 | 5119.61 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-07-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 09:40:00 | 5140.80 | 5123.61 | 0.00 | ORB-long ORB[5086.60,5135.00] vol=1.5x ATR=12.19 |
| Stop hit — per-position SL triggered | 2023-07-13 10:50:00 | 5128.61 | 5129.56 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-07-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 10:50:00 | 5127.55 | 5139.76 | 0.00 | ORB-short ORB[5130.00,5160.55] vol=1.7x ATR=9.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:05:00 | 5112.86 | 5137.16 | 0.00 | T1 1.5R @ 5112.86 |
| Target hit | 2023-07-18 15:20:00 | 5065.00 | 5090.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2023-07-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:40:00 | 5122.40 | 5099.41 | 0.00 | ORB-long ORB[5064.15,5104.35] vol=2.0x ATR=12.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 09:50:00 | 5141.50 | 5112.45 | 0.00 | T1 1.5R @ 5141.50 |
| Stop hit — per-position SL triggered | 2023-07-19 09:55:00 | 5122.40 | 5113.81 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 11:15:00 | 5000.00 | 5022.45 | 0.00 | ORB-short ORB[5003.15,5048.00] vol=3.2x ATR=9.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 11:30:00 | 4985.54 | 5014.79 | 0.00 | T1 1.5R @ 4985.54 |
| Target hit | 2023-07-24 15:20:00 | 4966.50 | 4989.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2023-07-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 11:05:00 | 4881.00 | 4894.58 | 0.00 | ORB-short ORB[4887.60,4959.40] vol=1.6x ATR=12.69 |
| Stop hit — per-position SL triggered | 2023-07-28 11:30:00 | 4893.69 | 4893.82 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 11:00:00 | 4769.20 | 4801.23 | 0.00 | ORB-short ORB[4800.00,4825.10] vol=3.5x ATR=9.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 11:25:00 | 4754.32 | 4794.11 | 0.00 | T1 1.5R @ 4754.32 |
| Stop hit — per-position SL triggered | 2023-08-01 11:40:00 | 4769.20 | 4788.43 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-08-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 10:25:00 | 4773.80 | 4803.76 | 0.00 | ORB-short ORB[4785.00,4832.00] vol=1.8x ATR=14.10 |
| Stop hit — per-position SL triggered | 2023-08-04 10:40:00 | 4787.90 | 4799.12 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-08-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 11:05:00 | 4612.00 | 4631.27 | 0.00 | ORB-short ORB[4621.45,4677.45] vol=2.0x ATR=7.24 |
| Stop hit — per-position SL triggered | 2023-08-08 11:20:00 | 4619.24 | 4629.53 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-08-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 09:40:00 | 4551.30 | 4570.26 | 0.00 | ORB-short ORB[4564.00,4618.65] vol=1.8x ATR=10.45 |
| Stop hit — per-position SL triggered | 2023-08-10 09:45:00 | 4561.75 | 4568.36 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-08-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 11:05:00 | 4508.15 | 4487.58 | 0.00 | ORB-long ORB[4466.65,4499.00] vol=1.6x ATR=6.92 |
| Stop hit — per-position SL triggered | 2023-08-18 11:15:00 | 4501.23 | 4488.38 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 11:15:00 | 4495.45 | 4505.06 | 0.00 | ORB-short ORB[4508.00,4555.00] vol=1.6x ATR=5.26 |
| Stop hit — per-position SL triggered | 2023-08-23 11:30:00 | 4500.71 | 4504.74 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-08-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:10:00 | 4513.80 | 4536.55 | 0.00 | ORB-short ORB[4536.05,4575.00] vol=2.2x ATR=4.70 |
| Stop hit — per-position SL triggered | 2023-08-31 11:30:00 | 4518.50 | 4534.27 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 11:15:00 | 4440.20 | 4451.22 | 0.00 | ORB-short ORB[4455.00,4475.00] vol=1.7x ATR=7.23 |
| Stop hit — per-position SL triggered | 2023-09-01 11:55:00 | 4447.43 | 4449.30 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-09-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 11:10:00 | 4530.55 | 4505.38 | 0.00 | ORB-long ORB[4486.05,4514.30] vol=1.7x ATR=7.98 |
| Stop hit — per-position SL triggered | 2023-09-05 11:55:00 | 4522.57 | 4512.16 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-09-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:50:00 | 4538.05 | 4527.12 | 0.00 | ORB-long ORB[4503.05,4532.85] vol=1.9x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 11:25:00 | 4548.52 | 4532.51 | 0.00 | T1 1.5R @ 4548.52 |
| Target hit | 2023-09-06 15:20:00 | 4577.65 | 4550.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2023-09-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 10:45:00 | 4523.05 | 4551.16 | 0.00 | ORB-short ORB[4540.90,4590.40] vol=2.0x ATR=8.69 |
| Stop hit — per-position SL triggered | 2023-09-07 10:55:00 | 4531.74 | 4548.96 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 09:30:00 | 4602.90 | 4580.03 | 0.00 | ORB-long ORB[4549.10,4586.00] vol=4.0x ATR=9.30 |
| Stop hit — per-position SL triggered | 2023-09-13 09:40:00 | 4593.60 | 4587.36 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 11:00:00 | 4535.00 | 4544.90 | 0.00 | ORB-short ORB[4537.35,4589.25] vol=1.7x ATR=6.45 |
| Stop hit — per-position SL triggered | 2023-09-15 11:05:00 | 4541.45 | 4544.66 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-09-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:35:00 | 4625.95 | 4601.05 | 0.00 | ORB-long ORB[4575.30,4605.00] vol=2.5x ATR=8.69 |
| Stop hit — per-position SL triggered | 2023-09-27 09:50:00 | 4617.26 | 4613.60 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 10:30:00 | 4571.20 | 4603.67 | 0.00 | ORB-short ORB[4601.15,4634.30] vol=1.5x ATR=11.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 10:40:00 | 4554.62 | 4598.45 | 0.00 | T1 1.5R @ 4554.62 |
| Stop hit — per-position SL triggered | 2023-09-28 10:45:00 | 4571.20 | 4597.41 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-10-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 10:10:00 | 4552.15 | 4532.63 | 0.00 | ORB-long ORB[4520.65,4534.60] vol=1.6x ATR=7.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 10:25:00 | 4563.36 | 4536.73 | 0.00 | T1 1.5R @ 4563.36 |
| Stop hit — per-position SL triggered | 2023-10-11 10:30:00 | 4552.15 | 4537.23 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-10-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 11:15:00 | 4545.30 | 4552.45 | 0.00 | ORB-short ORB[4557.15,4591.60] vol=8.0x ATR=6.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 11:20:00 | 4535.52 | 4552.10 | 0.00 | T1 1.5R @ 4535.52 |
| Stop hit — per-position SL triggered | 2023-10-12 11:45:00 | 4545.30 | 4551.03 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 11:15:00 | 4598.30 | 4585.48 | 0.00 | ORB-long ORB[4562.20,4595.00] vol=1.9x ATR=6.65 |
| Stop hit — per-position SL triggered | 2023-10-17 14:05:00 | 4591.65 | 4592.99 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-10-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-23 11:10:00 | 4593.10 | 4585.05 | 0.00 | ORB-long ORB[4556.00,4592.10] vol=9.2x ATR=7.31 |
| Stop hit — per-position SL triggered | 2023-10-23 11:45:00 | 4585.79 | 4586.79 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-10-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 11:00:00 | 4485.00 | 4493.98 | 0.00 | ORB-short ORB[4486.60,4524.80] vol=2.4x ATR=8.76 |
| Stop hit — per-position SL triggered | 2023-10-26 11:40:00 | 4493.76 | 4492.97 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-10-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 11:05:00 | 4506.40 | 4468.27 | 0.00 | ORB-long ORB[4463.00,4500.00] vol=1.8x ATR=10.74 |
| Stop hit — per-position SL triggered | 2023-10-30 11:10:00 | 4495.66 | 4468.96 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-11-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 11:05:00 | 4392.60 | 4419.59 | 0.00 | ORB-short ORB[4401.90,4442.60] vol=2.2x ATR=9.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 11:35:00 | 4378.19 | 4414.07 | 0.00 | T1 1.5R @ 4378.19 |
| Stop hit — per-position SL triggered | 2023-11-01 14:05:00 | 4392.60 | 4389.77 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-11-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:50:00 | 4578.30 | 4574.09 | 0.00 | ORB-long ORB[4546.05,4567.30] vol=12.0x ATR=6.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 11:15:00 | 4588.33 | 4574.94 | 0.00 | T1 1.5R @ 4588.33 |
| Target hit | 2023-11-06 15:20:00 | 4619.80 | 4601.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 11:15:00 | 4647.20 | 4649.96 | 0.00 | ORB-short ORB[4652.00,4682.90] vol=4.4x ATR=7.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 12:25:00 | 4636.51 | 4648.25 | 0.00 | T1 1.5R @ 4636.51 |
| Stop hit — per-position SL triggered | 2023-11-09 12:30:00 | 4647.20 | 4648.22 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-11-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:00:00 | 4714.95 | 4704.98 | 0.00 | ORB-long ORB[4679.95,4710.00] vol=3.0x ATR=7.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:15:00 | 4726.48 | 4708.34 | 0.00 | T1 1.5R @ 4726.48 |
| Stop hit — per-position SL triggered | 2023-11-16 11:45:00 | 4714.95 | 4710.25 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-11-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 10:30:00 | 4672.00 | 4688.60 | 0.00 | ORB-short ORB[4690.05,4730.00] vol=1.9x ATR=7.49 |
| Stop hit — per-position SL triggered | 2023-11-21 11:25:00 | 4679.49 | 4684.65 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 09:35:00 | 4660.00 | 4646.63 | 0.00 | ORB-long ORB[4626.00,4653.85] vol=1.5x ATR=7.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 09:40:00 | 4670.64 | 4650.43 | 0.00 | T1 1.5R @ 4670.64 |
| Stop hit — per-position SL triggered | 2023-11-28 09:50:00 | 4660.00 | 4654.95 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-11-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:50:00 | 4760.00 | 4740.44 | 0.00 | ORB-long ORB[4701.40,4734.80] vol=2.5x ATR=8.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 10:10:00 | 4772.39 | 4749.36 | 0.00 | T1 1.5R @ 4772.39 |
| Stop hit — per-position SL triggered | 2023-11-29 10:40:00 | 4760.00 | 4755.24 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-11-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 09:30:00 | 4831.35 | 4822.66 | 0.00 | ORB-long ORB[4805.20,4830.20] vol=2.5x ATR=8.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 09:35:00 | 4843.77 | 4826.38 | 0.00 | T1 1.5R @ 4843.77 |
| Stop hit — per-position SL triggered | 2023-11-30 09:45:00 | 4831.35 | 4828.24 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-12-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 09:55:00 | 5010.00 | 4995.82 | 0.00 | ORB-long ORB[4963.15,5003.15] vol=1.6x ATR=11.21 |
| Stop hit — per-position SL triggered | 2023-12-07 10:45:00 | 4998.79 | 4999.61 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-12-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:30:00 | 5015.95 | 5022.79 | 0.00 | ORB-short ORB[5018.10,5060.00] vol=2.6x ATR=10.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 10:40:00 | 5000.01 | 5022.41 | 0.00 | T1 1.5R @ 5000.01 |
| Stop hit — per-position SL triggered | 2023-12-08 11:50:00 | 5015.95 | 5017.01 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:15:00 | 4872.70 | 4891.33 | 0.00 | ORB-short ORB[4882.75,4927.45] vol=4.5x ATR=10.01 |
| Stop hit — per-position SL triggered | 2023-12-19 10:50:00 | 4882.71 | 4885.86 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:35:00 | 4963.20 | 4941.18 | 0.00 | ORB-long ORB[4916.00,4946.90] vol=3.4x ATR=9.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 09:45:00 | 4977.74 | 4951.22 | 0.00 | T1 1.5R @ 4977.74 |
| Stop hit — per-position SL triggered | 2023-12-20 10:05:00 | 4963.20 | 4960.20 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-12-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 09:30:00 | 4952.50 | 4910.23 | 0.00 | ORB-long ORB[4877.55,4935.00] vol=2.2x ATR=21.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-21 09:40:00 | 4984.36 | 4926.85 | 0.00 | T1 1.5R @ 4984.36 |
| Target hit | 2023-12-21 15:20:00 | 5049.50 | 5040.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2023-12-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:40:00 | 5286.00 | 5267.62 | 0.00 | ORB-long ORB[5238.65,5275.40] vol=1.7x ATR=17.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 11:20:00 | 5311.83 | 5280.13 | 0.00 | T1 1.5R @ 5311.83 |
| Target hit | 2023-12-29 12:05:00 | 5303.45 | 5320.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2024-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:45:00 | 5261.75 | 5284.32 | 0.00 | ORB-short ORB[5280.45,5334.95] vol=2.0x ATR=10.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 11:05:00 | 5246.36 | 5279.45 | 0.00 | T1 1.5R @ 5246.36 |
| Stop hit — per-position SL triggered | 2024-01-02 12:35:00 | 5261.75 | 5262.40 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-01-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 09:55:00 | 5232.20 | 5250.98 | 0.00 | ORB-short ORB[5234.05,5271.00] vol=1.7x ATR=12.66 |
| Stop hit — per-position SL triggered | 2024-01-04 10:30:00 | 5244.86 | 5245.70 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-01-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:55:00 | 5286.15 | 5306.01 | 0.00 | ORB-short ORB[5323.90,5363.35] vol=2.0x ATR=11.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 11:50:00 | 5268.65 | 5299.15 | 0.00 | T1 1.5R @ 5268.65 |
| Stop hit — per-position SL triggered | 2024-01-05 13:40:00 | 5286.15 | 5289.79 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 10:55:00 | 5197.95 | 5224.74 | 0.00 | ORB-short ORB[5240.10,5305.00] vol=1.6x ATR=9.69 |
| Stop hit — per-position SL triggered | 2024-01-08 11:20:00 | 5207.64 | 5222.40 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-01-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:45:00 | 5126.50 | 5157.10 | 0.00 | ORB-short ORB[5158.70,5210.20] vol=2.3x ATR=11.18 |
| Stop hit — per-position SL triggered | 2024-01-09 10:05:00 | 5137.68 | 5148.62 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-01-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:50:00 | 5089.30 | 5095.30 | 0.00 | ORB-short ORB[5095.40,5129.90] vol=7.0x ATR=8.47 |
| Stop hit — per-position SL triggered | 2024-01-11 11:15:00 | 5097.77 | 5095.63 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-01-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 09:40:00 | 5175.00 | 5159.75 | 0.00 | ORB-long ORB[5100.05,5171.90] vol=2.4x ATR=15.18 |
| Stop hit — per-position SL triggered | 2024-01-12 10:05:00 | 5159.82 | 5164.05 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:50:00 | 5086.90 | 5108.11 | 0.00 | ORB-short ORB[5095.50,5135.25] vol=5.3x ATR=12.25 |
| Stop hit — per-position SL triggered | 2024-01-17 10:55:00 | 5099.15 | 5107.72 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-01-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:45:00 | 4993.25 | 5038.15 | 0.00 | ORB-short ORB[5018.30,5077.95] vol=2.2x ATR=16.48 |
| Stop hit — per-position SL triggered | 2024-01-18 10:05:00 | 5009.73 | 5025.88 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-01-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 10:40:00 | 5137.00 | 5090.23 | 0.00 | ORB-long ORB[5030.00,5074.45] vol=3.3x ATR=19.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-24 10:50:00 | 5165.54 | 5095.33 | 0.00 | T1 1.5R @ 5165.54 |
| Stop hit — per-position SL triggered | 2024-01-24 11:25:00 | 5137.00 | 5115.55 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-02-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:40:00 | 4963.35 | 5012.20 | 0.00 | ORB-short ORB[5048.60,5122.40] vol=1.5x ATR=17.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:00:00 | 4937.58 | 5003.36 | 0.00 | T1 1.5R @ 4937.58 |
| Target hit | 2024-02-08 15:20:00 | 4888.00 | 4923.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2024-02-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 10:05:00 | 5005.00 | 4986.12 | 0.00 | ORB-long ORB[4940.20,4992.40] vol=1.8x ATR=14.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 10:10:00 | 5026.07 | 4993.91 | 0.00 | T1 1.5R @ 5026.07 |
| Stop hit — per-position SL triggered | 2024-02-13 10:55:00 | 5005.00 | 5000.94 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-02-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 09:40:00 | 5003.55 | 4982.96 | 0.00 | ORB-long ORB[4956.05,4992.60] vol=1.7x ATR=12.71 |
| Stop hit — per-position SL triggered | 2024-02-14 09:45:00 | 4990.84 | 4983.53 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-02-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 10:25:00 | 4981.80 | 4997.08 | 0.00 | ORB-short ORB[4990.00,5043.35] vol=2.9x ATR=12.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 11:15:00 | 4963.37 | 4991.07 | 0.00 | T1 1.5R @ 4963.37 |
| Target hit | 2024-02-15 15:20:00 | 4950.80 | 4961.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2024-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 10:45:00 | 4911.70 | 4935.53 | 0.00 | ORB-short ORB[4942.90,4970.55] vol=1.5x ATR=8.40 |
| Stop hit — per-position SL triggered | 2024-02-16 10:55:00 | 4920.10 | 4932.61 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:50:00 | 4900.00 | 4915.66 | 0.00 | ORB-short ORB[4910.15,4931.25] vol=1.5x ATR=10.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 10:10:00 | 4884.99 | 4906.64 | 0.00 | T1 1.5R @ 4884.99 |
| Stop hit — per-position SL triggered | 2024-02-20 10:15:00 | 4900.00 | 4905.69 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-02-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 09:35:00 | 4890.90 | 4904.29 | 0.00 | ORB-short ORB[4893.80,4924.95] vol=1.6x ATR=12.05 |
| Stop hit — per-position SL triggered | 2024-02-22 09:40:00 | 4902.95 | 4903.69 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 09:45:00 | 4919.00 | 4904.64 | 0.00 | ORB-long ORB[4879.00,4909.75] vol=2.1x ATR=7.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 10:35:00 | 4930.52 | 4910.43 | 0.00 | T1 1.5R @ 4930.52 |
| Stop hit — per-position SL triggered | 2024-02-27 11:05:00 | 4919.00 | 4916.25 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:55:00 | 4890.60 | 4903.67 | 0.00 | ORB-short ORB[4890.90,4916.10] vol=1.8x ATR=8.55 |
| Stop hit — per-position SL triggered | 2024-02-28 12:15:00 | 4899.15 | 4899.81 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-02-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 11:05:00 | 4900.00 | 4870.33 | 0.00 | ORB-long ORB[4828.85,4889.00] vol=2.4x ATR=13.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 11:15:00 | 4920.62 | 4874.59 | 0.00 | T1 1.5R @ 4920.62 |
| Stop hit — per-position SL triggered | 2024-02-29 11:30:00 | 4900.00 | 4877.11 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-03-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 11:10:00 | 4951.00 | 4957.43 | 0.00 | ORB-short ORB[4951.45,4986.70] vol=1.5x ATR=10.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-01 11:40:00 | 4934.66 | 4955.04 | 0.00 | T1 1.5R @ 4934.66 |
| Target hit | 2024-03-01 13:20:00 | 4940.80 | 4940.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — SELL (started 2024-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:35:00 | 4881.00 | 4896.56 | 0.00 | ORB-short ORB[4882.10,4942.45] vol=2.5x ATR=12.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 13:00:00 | 4862.85 | 4882.65 | 0.00 | T1 1.5R @ 4862.85 |
| Target hit | 2024-03-04 15:20:00 | 4840.05 | 4867.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2024-03-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 11:10:00 | 4880.35 | 4857.84 | 0.00 | ORB-long ORB[4851.50,4880.00] vol=2.5x ATR=10.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 12:35:00 | 4896.42 | 4869.25 | 0.00 | T1 1.5R @ 4896.42 |
| Stop hit — per-position SL triggered | 2024-03-07 13:25:00 | 4880.35 | 4873.53 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 09:45:00 | 4920.00 | 4916.33 | 0.00 | ORB-long ORB[4871.10,4916.60] vol=1.9x ATR=12.24 |
| Stop hit — per-position SL triggered | 2024-03-11 09:55:00 | 4907.76 | 4915.78 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-13 09:45:00 | 4914.10 | 4892.44 | 0.00 | ORB-long ORB[4854.10,4906.50] vol=1.8x ATR=14.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 10:10:00 | 4935.38 | 4912.01 | 0.00 | T1 1.5R @ 4935.38 |
| Target hit | 2024-03-13 11:00:00 | 4936.05 | 4938.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 75 — BUY (started 2024-03-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 10:45:00 | 4998.05 | 4966.79 | 0.00 | ORB-long ORB[4912.00,4978.80] vol=3.3x ATR=16.61 |
| Stop hit — per-position SL triggered | 2024-03-15 12:45:00 | 4981.44 | 4980.24 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-03-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 11:00:00 | 4844.75 | 4821.79 | 0.00 | ORB-long ORB[4792.40,4844.40] vol=1.6x ATR=13.86 |
| Stop hit — per-position SL triggered | 2024-03-22 11:25:00 | 4830.89 | 4824.61 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-04-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-03 09:30:00 | 4819.00 | 4840.95 | 0.00 | ORB-short ORB[4831.00,4896.95] vol=3.7x ATR=12.22 |
| Stop hit — per-position SL triggered | 2024-04-03 09:40:00 | 4831.22 | 4833.91 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 4794.90 | 4820.45 | 0.00 | ORB-short ORB[4819.00,4855.00] vol=2.0x ATR=11.38 |
| Stop hit — per-position SL triggered | 2024-04-04 10:25:00 | 4806.28 | 4810.56 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-04-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 10:20:00 | 4804.50 | 4815.37 | 0.00 | ORB-short ORB[4805.10,4831.60] vol=2.1x ATR=8.19 |
| Stop hit — per-position SL triggered | 2024-04-08 10:40:00 | 4812.69 | 4809.33 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-04-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-09 11:05:00 | 4815.00 | 4826.52 | 0.00 | ORB-short ORB[4826.00,4850.00] vol=1.7x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 12:05:00 | 4806.70 | 4822.64 | 0.00 | T1 1.5R @ 4806.70 |
| Target hit | 2024-04-09 15:20:00 | 4791.85 | 4806.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — SELL (started 2024-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 09:30:00 | 4768.20 | 4782.08 | 0.00 | ORB-short ORB[4775.10,4818.55] vol=2.2x ATR=9.44 |
| Stop hit — per-position SL triggered | 2024-04-10 09:40:00 | 4777.64 | 4780.82 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-04-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 10:25:00 | 4786.75 | 4790.85 | 0.00 | ORB-short ORB[4793.00,4820.85] vol=3.4x ATR=9.37 |
| Stop hit — per-position SL triggered | 2024-04-12 10:30:00 | 4796.12 | 4790.16 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-15 11:05:00 | 4715.00 | 4741.09 | 0.00 | ORB-short ORB[4715.95,4751.70] vol=2.3x ATR=11.78 |
| Stop hit — per-position SL triggered | 2024-04-15 12:05:00 | 4726.78 | 4732.27 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-16 11:05:00 | 4697.00 | 4726.35 | 0.00 | ORB-short ORB[4717.05,4758.45] vol=2.7x ATR=12.69 |
| Stop hit — per-position SL triggered | 2024-04-16 11:20:00 | 4709.69 | 4722.72 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-04-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:35:00 | 4720.00 | 4733.93 | 0.00 | ORB-short ORB[4740.90,4758.70] vol=2.3x ATR=9.82 |
| Stop hit — per-position SL triggered | 2024-04-18 09:40:00 | 4729.82 | 4732.99 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 11:05:00 | 4718.00 | 4689.22 | 0.00 | ORB-long ORB[4662.30,4688.60] vol=1.6x ATR=9.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 11:40:00 | 4731.97 | 4696.68 | 0.00 | T1 1.5R @ 4731.97 |
| Stop hit — per-position SL triggered | 2024-04-22 12:20:00 | 4718.00 | 4704.34 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-04-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 11:00:00 | 4758.00 | 4783.79 | 0.00 | ORB-short ORB[4780.65,4816.15] vol=5.1x ATR=11.85 |
| Stop hit — per-position SL triggered | 2024-04-25 11:10:00 | 4769.85 | 4781.55 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-04-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 10:40:00 | 4816.80 | 4799.82 | 0.00 | ORB-long ORB[4776.30,4802.30] vol=1.5x ATR=9.41 |
| Stop hit — per-position SL triggered | 2024-04-30 10:45:00 | 4807.39 | 4800.27 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-05-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 11:00:00 | 4807.00 | 4803.39 | 0.00 | ORB-long ORB[4780.00,4803.20] vol=2.4x ATR=9.21 |
| Stop hit — per-position SL triggered | 2024-05-02 11:05:00 | 4797.79 | 4803.13 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 11:15:00 | 5243.15 | 5181.74 | 0.00 | ORB-long ORB[5123.85,5179.35] vol=5.4x ATR=14.81 |
| Stop hit — per-position SL triggered | 2024-05-08 11:20:00 | 5228.34 | 5185.67 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-19 10:10:00 | 4503.00 | 2023-05-19 10:40:00 | 4487.85 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-05-19 10:10:00 | 4503.00 | 2023-05-19 11:05:00 | 4503.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-30 11:10:00 | 4613.65 | 2023-05-30 11:45:00 | 4606.14 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-06-13 09:55:00 | 4935.00 | 2023-06-13 10:15:00 | 4947.93 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-06-13 09:55:00 | 4935.00 | 2023-06-13 10:25:00 | 4935.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-14 10:30:00 | 4968.30 | 2023-06-14 10:40:00 | 4960.16 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-06-20 09:30:00 | 5019.00 | 2023-06-20 09:35:00 | 5028.15 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-23 11:15:00 | 4954.00 | 2023-06-23 11:40:00 | 4961.09 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-06-28 11:15:00 | 4997.15 | 2023-06-28 11:35:00 | 5006.97 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2023-06-28 11:15:00 | 4997.15 | 2023-06-28 13:10:00 | 4997.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-05 10:35:00 | 5038.00 | 2023-07-05 11:10:00 | 5051.94 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-07-05 10:35:00 | 5038.00 | 2023-07-05 15:20:00 | 5107.45 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2023-07-11 09:30:00 | 5136.80 | 2023-07-11 10:05:00 | 5121.24 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-07-13 09:40:00 | 5140.80 | 2023-07-13 10:50:00 | 5128.61 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-18 10:50:00 | 5127.55 | 2023-07-18 11:05:00 | 5112.86 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-07-18 10:50:00 | 5127.55 | 2023-07-18 15:20:00 | 5065.00 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2023-07-19 09:40:00 | 5122.40 | 2023-07-19 09:50:00 | 5141.50 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-07-19 09:40:00 | 5122.40 | 2023-07-19 09:55:00 | 5122.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-24 11:15:00 | 5000.00 | 2023-07-24 11:30:00 | 4985.54 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-07-24 11:15:00 | 5000.00 | 2023-07-24 15:20:00 | 4966.50 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2023-07-28 11:05:00 | 4881.00 | 2023-07-28 11:30:00 | 4893.69 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-08-01 11:00:00 | 4769.20 | 2023-08-01 11:25:00 | 4754.32 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-08-01 11:00:00 | 4769.20 | 2023-08-01 11:40:00 | 4769.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-04 10:25:00 | 4773.80 | 2023-08-04 10:40:00 | 4787.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-08-08 11:05:00 | 4612.00 | 2023-08-08 11:20:00 | 4619.24 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-10 09:40:00 | 4551.30 | 2023-08-10 09:45:00 | 4561.75 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-08-18 11:05:00 | 4508.15 | 2023-08-18 11:15:00 | 4501.23 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-08-23 11:15:00 | 4495.45 | 2023-08-23 11:30:00 | 4500.71 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2023-08-31 11:10:00 | 4513.80 | 2023-08-31 11:30:00 | 4518.50 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest1 | 2023-09-01 11:15:00 | 4440.20 | 2023-09-01 11:55:00 | 4447.43 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-05 11:10:00 | 4530.55 | 2023-09-05 11:55:00 | 4522.57 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-09-06 10:50:00 | 4538.05 | 2023-09-06 11:25:00 | 4548.52 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-09-06 10:50:00 | 4538.05 | 2023-09-06 15:20:00 | 4577.65 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2023-09-07 10:45:00 | 4523.05 | 2023-09-07 10:55:00 | 4531.74 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-09-13 09:30:00 | 4602.90 | 2023-09-13 09:40:00 | 4593.60 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-15 11:00:00 | 4535.00 | 2023-09-15 11:05:00 | 4541.45 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-09-27 09:35:00 | 4625.95 | 2023-09-27 09:50:00 | 4617.26 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-09-28 10:30:00 | 4571.20 | 2023-09-28 10:40:00 | 4554.62 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-09-28 10:30:00 | 4571.20 | 2023-09-28 10:45:00 | 4571.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-11 10:10:00 | 4552.15 | 2023-10-11 10:25:00 | 4563.36 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-10-11 10:10:00 | 4552.15 | 2023-10-11 10:30:00 | 4552.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-12 11:15:00 | 4545.30 | 2023-10-12 11:20:00 | 4535.52 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-10-12 11:15:00 | 4545.30 | 2023-10-12 11:45:00 | 4545.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-17 11:15:00 | 4598.30 | 2023-10-17 14:05:00 | 4591.65 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-10-23 11:10:00 | 4593.10 | 2023-10-23 11:45:00 | 4585.79 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-10-26 11:00:00 | 4485.00 | 2023-10-26 11:40:00 | 4493.76 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-30 11:05:00 | 4506.40 | 2023-10-30 11:10:00 | 4495.66 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-11-01 11:05:00 | 4392.60 | 2023-11-01 11:35:00 | 4378.19 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-11-01 11:05:00 | 4392.60 | 2023-11-01 14:05:00 | 4392.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-06 10:50:00 | 4578.30 | 2023-11-06 11:15:00 | 4588.33 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-11-06 10:50:00 | 4578.30 | 2023-11-06 15:20:00 | 4619.80 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2023-11-09 11:15:00 | 4647.20 | 2023-11-09 12:25:00 | 4636.51 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-11-09 11:15:00 | 4647.20 | 2023-11-09 12:30:00 | 4647.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-16 11:00:00 | 4714.95 | 2023-11-16 11:15:00 | 4726.48 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-11-16 11:00:00 | 4714.95 | 2023-11-16 11:45:00 | 4714.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-21 10:30:00 | 4672.00 | 2023-11-21 11:25:00 | 4679.49 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-28 09:35:00 | 4660.00 | 2023-11-28 09:40:00 | 4670.64 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-11-28 09:35:00 | 4660.00 | 2023-11-28 09:50:00 | 4660.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-29 09:50:00 | 4760.00 | 2023-11-29 10:10:00 | 4772.39 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-11-29 09:50:00 | 4760.00 | 2023-11-29 10:40:00 | 4760.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 09:30:00 | 4831.35 | 2023-11-30 09:35:00 | 4843.77 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-11-30 09:30:00 | 4831.35 | 2023-11-30 09:45:00 | 4831.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-07 09:55:00 | 5010.00 | 2023-12-07 10:45:00 | 4998.79 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-12-08 10:30:00 | 5015.95 | 2023-12-08 10:40:00 | 5000.01 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-12-08 10:30:00 | 5015.95 | 2023-12-08 11:50:00 | 5015.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-19 10:15:00 | 4872.70 | 2023-12-19 10:50:00 | 4882.71 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-20 09:35:00 | 4963.20 | 2023-12-20 09:45:00 | 4977.74 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-12-20 09:35:00 | 4963.20 | 2023-12-20 10:05:00 | 4963.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-21 09:30:00 | 4952.50 | 2023-12-21 09:40:00 | 4984.36 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2023-12-21 09:30:00 | 4952.50 | 2023-12-21 15:20:00 | 5049.50 | TARGET_HIT | 0.50 | 1.96% |
| BUY | retest1 | 2023-12-29 10:40:00 | 5286.00 | 2023-12-29 11:20:00 | 5311.83 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-12-29 10:40:00 | 5286.00 | 2023-12-29 12:05:00 | 5303.45 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2024-01-02 10:45:00 | 5261.75 | 2024-01-02 11:05:00 | 5246.36 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-01-02 10:45:00 | 5261.75 | 2024-01-02 12:35:00 | 5261.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-04 09:55:00 | 5232.20 | 2024-01-04 10:30:00 | 5244.86 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-05 10:55:00 | 5286.15 | 2024-01-05 11:50:00 | 5268.65 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-01-05 10:55:00 | 5286.15 | 2024-01-05 13:40:00 | 5286.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-08 10:55:00 | 5197.95 | 2024-01-08 11:20:00 | 5207.64 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-01-09 09:45:00 | 5126.50 | 2024-01-09 10:05:00 | 5137.68 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-01-11 10:50:00 | 5089.30 | 2024-01-11 11:15:00 | 5097.77 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-01-12 09:40:00 | 5175.00 | 2024-01-12 10:05:00 | 5159.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-01-17 10:50:00 | 5086.90 | 2024-01-17 10:55:00 | 5099.15 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-18 09:45:00 | 4993.25 | 2024-01-18 10:05:00 | 5009.73 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-01-24 10:40:00 | 5137.00 | 2024-01-24 10:50:00 | 5165.54 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-01-24 10:40:00 | 5137.00 | 2024-01-24 11:25:00 | 5137.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 10:40:00 | 4963.35 | 2024-02-08 11:00:00 | 4937.58 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-02-08 10:40:00 | 4963.35 | 2024-02-08 15:20:00 | 4888.00 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2024-02-13 10:05:00 | 5005.00 | 2024-02-13 10:10:00 | 5026.07 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-02-13 10:05:00 | 5005.00 | 2024-02-13 10:55:00 | 5005.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-14 09:40:00 | 5003.55 | 2024-02-14 09:45:00 | 4990.84 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-15 10:25:00 | 4981.80 | 2024-02-15 11:15:00 | 4963.37 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-02-15 10:25:00 | 4981.80 | 2024-02-15 15:20:00 | 4950.80 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2024-02-16 10:45:00 | 4911.70 | 2024-02-16 10:55:00 | 4920.10 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-02-20 09:50:00 | 4900.00 | 2024-02-20 10:10:00 | 4884.99 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-02-20 09:50:00 | 4900.00 | 2024-02-20 10:15:00 | 4900.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-22 09:35:00 | 4890.90 | 2024-02-22 09:40:00 | 4902.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-02-27 09:45:00 | 4919.00 | 2024-02-27 10:35:00 | 4930.52 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2024-02-27 09:45:00 | 4919.00 | 2024-02-27 11:05:00 | 4919.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-28 10:55:00 | 4890.60 | 2024-02-28 12:15:00 | 4899.15 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-02-29 11:05:00 | 4900.00 | 2024-02-29 11:15:00 | 4920.62 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-02-29 11:05:00 | 4900.00 | 2024-02-29 11:30:00 | 4900.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-01 11:10:00 | 4951.00 | 2024-03-01 11:40:00 | 4934.66 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-03-01 11:10:00 | 4951.00 | 2024-03-01 13:20:00 | 4940.80 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2024-03-04 09:35:00 | 4881.00 | 2024-03-04 13:00:00 | 4862.85 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-03-04 09:35:00 | 4881.00 | 2024-03-04 15:20:00 | 4840.05 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2024-03-07 11:10:00 | 4880.35 | 2024-03-07 12:35:00 | 4896.42 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-03-07 11:10:00 | 4880.35 | 2024-03-07 13:25:00 | 4880.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-11 09:45:00 | 4920.00 | 2024-03-11 09:55:00 | 4907.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-03-13 09:45:00 | 4914.10 | 2024-03-13 10:10:00 | 4935.38 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-03-13 09:45:00 | 4914.10 | 2024-03-13 11:00:00 | 4936.05 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-03-15 10:45:00 | 4998.05 | 2024-03-15 12:45:00 | 4981.44 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-22 11:00:00 | 4844.75 | 2024-03-22 11:25:00 | 4830.89 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-04-03 09:30:00 | 4819.00 | 2024-04-03 09:40:00 | 4831.22 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-04-04 09:50:00 | 4794.90 | 2024-04-04 10:25:00 | 4806.28 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-08 10:20:00 | 4804.50 | 2024-04-08 10:40:00 | 4812.69 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-04-09 11:05:00 | 4815.00 | 2024-04-09 12:05:00 | 4806.70 | PARTIAL | 0.50 | 0.17% |
| SELL | retest1 | 2024-04-09 11:05:00 | 4815.00 | 2024-04-09 15:20:00 | 4791.85 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2024-04-10 09:30:00 | 4768.20 | 2024-04-10 09:40:00 | 4777.64 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-04-12 10:25:00 | 4786.75 | 2024-04-12 10:30:00 | 4796.12 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-04-15 11:05:00 | 4715.00 | 2024-04-15 12:05:00 | 4726.78 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-04-16 11:05:00 | 4697.00 | 2024-04-16 11:20:00 | 4709.69 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-04-18 09:35:00 | 4720.00 | 2024-04-18 09:40:00 | 4729.82 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-04-22 11:05:00 | 4718.00 | 2024-04-22 11:40:00 | 4731.97 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-04-22 11:05:00 | 4718.00 | 2024-04-22 12:20:00 | 4718.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-25 11:00:00 | 4758.00 | 2024-04-25 11:10:00 | 4769.85 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-04-30 10:40:00 | 4816.80 | 2024-04-30 10:45:00 | 4807.39 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-05-02 11:00:00 | 4807.00 | 2024-05-02 11:05:00 | 4797.79 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-05-08 11:15:00 | 5243.15 | 2024-05-08 11:20:00 | 5228.34 | STOP_HIT | 1.00 | -0.28% |
