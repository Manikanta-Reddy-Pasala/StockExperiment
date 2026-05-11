# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Window:** 2024-08-09 09:15:00 → 2026-05-08 15:25:00 (32275 bars)
- **Last close:** 7309.00
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
| ENTRY1 | 71 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 8 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 63
- **Target hits / Stop hits / Partials:** 8 / 63 / 24
- **Avg / median % per leg:** 0.02% / 0.00%
- **Sum % (uncompounded):** 2.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 18 | 36.0% | 4 | 32 | 14 | 0.08% | 4.1% |
| BUY @ 2nd Alert (retest1) | 50 | 18 | 36.0% | 4 | 32 | 14 | 0.08% | 4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 45 | 14 | 31.1% | 4 | 31 | 10 | -0.05% | -2.1% |
| SELL @ 2nd Alert (retest1) | 45 | 14 | 31.1% | 4 | 31 | 10 | -0.05% | -2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 95 | 32 | 33.7% | 8 | 63 | 24 | 0.02% | 2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:35:00 | 4760.00 | 4771.32 | 0.00 | ORB-short ORB[4790.05,4820.00] vol=2.0x ATR=11.66 |
| Stop hit — per-position SL triggered | 2024-08-14 10:40:00 | 4771.66 | 4771.00 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-08-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:45:00 | 4805.00 | 4787.80 | 0.00 | ORB-long ORB[4735.10,4801.00] vol=1.7x ATR=13.93 |
| Stop hit — per-position SL triggered | 2024-08-16 10:00:00 | 4791.07 | 4789.01 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:15:00 | 4788.20 | 4805.71 | 0.00 | ORB-short ORB[4795.40,4848.00] vol=1.6x ATR=10.52 |
| Stop hit — per-position SL triggered | 2024-08-19 10:45:00 | 4798.72 | 4800.79 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-08-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:10:00 | 4903.00 | 4897.62 | 0.00 | ORB-long ORB[4870.40,4890.00] vol=1.5x ATR=8.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 12:40:00 | 4916.39 | 4899.71 | 0.00 | T1 1.5R @ 4916.39 |
| Stop hit — per-position SL triggered | 2024-08-21 14:40:00 | 4903.00 | 4903.32 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:15:00 | 4852.00 | 4855.83 | 0.00 | ORB-short ORB[4860.00,4887.25] vol=1.8x ATR=7.68 |
| Stop hit — per-position SL triggered | 2024-08-27 10:25:00 | 4859.68 | 4855.69 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-08-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:20:00 | 4887.00 | 4868.17 | 0.00 | ORB-long ORB[4840.00,4884.00] vol=1.7x ATR=11.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:45:00 | 4904.51 | 4874.94 | 0.00 | T1 1.5R @ 4904.51 |
| Target hit | 2024-08-28 15:20:00 | 4947.45 | 4939.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-08-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:40:00 | 4977.55 | 4956.27 | 0.00 | ORB-long ORB[4924.05,4953.15] vol=1.6x ATR=12.13 |
| Stop hit — per-position SL triggered | 2024-08-29 09:55:00 | 4965.42 | 4967.28 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-09-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:25:00 | 4867.00 | 4852.74 | 0.00 | ORB-long ORB[4821.20,4856.70] vol=4.0x ATR=10.53 |
| Stop hit — per-position SL triggered | 2024-09-04 13:05:00 | 4856.47 | 4857.31 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-09-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:55:00 | 4772.50 | 4786.19 | 0.00 | ORB-short ORB[4779.65,4824.00] vol=2.8x ATR=12.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 11:10:00 | 4754.36 | 4780.34 | 0.00 | T1 1.5R @ 4754.36 |
| Target hit | 2024-09-06 15:20:00 | 4759.40 | 4767.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-09-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:25:00 | 4729.95 | 4741.45 | 0.00 | ORB-short ORB[4743.10,4765.45] vol=2.1x ATR=11.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:30:00 | 4712.32 | 4732.41 | 0.00 | T1 1.5R @ 4712.32 |
| Stop hit — per-position SL triggered | 2024-09-10 10:35:00 | 4729.95 | 4729.62 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-09-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:40:00 | 4750.00 | 4726.55 | 0.00 | ORB-long ORB[4705.35,4730.00] vol=1.8x ATR=11.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 10:50:00 | 4766.61 | 4731.69 | 0.00 | T1 1.5R @ 4766.61 |
| Stop hit — per-position SL triggered | 2024-09-11 10:55:00 | 4750.00 | 4732.53 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-09-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:05:00 | 4929.00 | 4896.18 | 0.00 | ORB-long ORB[4860.00,4903.15] vol=1.8x ATR=15.77 |
| Stop hit — per-position SL triggered | 2024-09-16 10:15:00 | 4913.23 | 4898.40 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 4857.45 | 4866.01 | 0.00 | ORB-short ORB[4861.15,4898.50] vol=3.0x ATR=8.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:45:00 | 4843.97 | 4859.57 | 0.00 | T1 1.5R @ 4843.97 |
| Stop hit — per-position SL triggered | 2024-09-17 10:20:00 | 4857.45 | 4849.22 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:50:00 | 4840.45 | 4856.38 | 0.00 | ORB-short ORB[4851.65,4876.75] vol=1.7x ATR=8.99 |
| Stop hit — per-position SL triggered | 2024-09-18 11:25:00 | 4849.44 | 4852.91 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:45:00 | 4916.00 | 4899.72 | 0.00 | ORB-long ORB[4863.40,4910.00] vol=2.5x ATR=14.66 |
| Stop hit — per-position SL triggered | 2024-09-19 09:55:00 | 4901.34 | 4900.38 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:25:00 | 4906.55 | 4931.59 | 0.00 | ORB-short ORB[4947.40,4991.40] vol=1.8x ATR=13.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 10:35:00 | 4886.14 | 4926.03 | 0.00 | T1 1.5R @ 4886.14 |
| Target hit | 2024-09-23 14:15:00 | 4893.00 | 4888.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2024-09-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:55:00 | 4967.85 | 4932.48 | 0.00 | ORB-long ORB[4895.20,4918.00] vol=2.6x ATR=12.63 |
| Stop hit — per-position SL triggered | 2024-09-26 10:00:00 | 4955.22 | 4939.04 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-09-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:40:00 | 5020.80 | 5000.51 | 0.00 | ORB-long ORB[4953.35,5013.85] vol=1.5x ATR=15.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 09:55:00 | 5044.05 | 5015.56 | 0.00 | T1 1.5R @ 5044.05 |
| Stop hit — per-position SL triggered | 2024-09-27 10:10:00 | 5020.80 | 5017.89 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:55:00 | 5000.40 | 5048.93 | 0.00 | ORB-short ORB[5032.35,5079.55] vol=3.5x ATR=15.42 |
| Stop hit — per-position SL triggered | 2024-09-30 11:15:00 | 5015.82 | 5046.18 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:00:00 | 4968.65 | 4991.70 | 0.00 | ORB-short ORB[4990.70,5045.95] vol=2.2x ATR=16.90 |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 4985.55 | 4985.26 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-10-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:45:00 | 4688.70 | 4700.79 | 0.00 | ORB-short ORB[4700.00,4748.80] vol=4.9x ATR=14.23 |
| Stop hit — per-position SL triggered | 2024-10-07 10:35:00 | 4702.93 | 4698.25 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-10-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:00:00 | 4754.60 | 4744.24 | 0.00 | ORB-long ORB[4717.10,4749.15] vol=1.9x ATR=13.24 |
| Stop hit — per-position SL triggered | 2024-10-10 10:35:00 | 4741.36 | 4745.78 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-10-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:05:00 | 4780.00 | 4705.91 | 0.00 | ORB-long ORB[4660.00,4705.55] vol=1.9x ATR=18.94 |
| Stop hit — per-position SL triggered | 2024-10-11 10:30:00 | 4761.06 | 4723.23 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-10-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 10:20:00 | 4792.00 | 4776.67 | 0.00 | ORB-long ORB[4730.65,4765.00] vol=5.2x ATR=12.24 |
| Stop hit — per-position SL triggered | 2024-10-14 11:00:00 | 4779.76 | 4782.18 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 4745.65 | 4769.56 | 0.00 | ORB-short ORB[4755.15,4799.90] vol=1.7x ATR=10.24 |
| Stop hit — per-position SL triggered | 2024-10-15 09:35:00 | 4755.89 | 4767.06 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 4655.00 | 4677.00 | 0.00 | ORB-short ORB[4689.90,4734.00] vol=3.0x ATR=8.97 |
| Stop hit — per-position SL triggered | 2024-10-16 11:50:00 | 4663.97 | 4672.68 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-10-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:45:00 | 4787.20 | 4790.20 | 0.00 | ORB-short ORB[4792.05,4826.20] vol=2.0x ATR=14.41 |
| Stop hit — per-position SL triggered | 2024-10-22 11:10:00 | 4801.61 | 4791.93 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 10:50:00 | 4687.70 | 4698.57 | 0.00 | ORB-short ORB[4688.35,4725.45] vol=2.1x ATR=16.36 |
| Stop hit — per-position SL triggered | 2024-10-23 11:10:00 | 4704.06 | 4698.71 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:50:00 | 4523.55 | 4592.31 | 0.00 | ORB-short ORB[4638.10,4672.65] vol=1.8x ATR=15.45 |
| Stop hit — per-position SL triggered | 2024-10-25 10:55:00 | 4539.00 | 4587.16 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-10-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:55:00 | 4908.00 | 4865.12 | 0.00 | ORB-long ORB[4838.50,4876.65] vol=1.8x ATR=14.00 |
| Stop hit — per-position SL triggered | 2024-10-31 11:05:00 | 4894.00 | 4867.64 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-11-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:25:00 | 4848.15 | 4868.78 | 0.00 | ORB-short ORB[4872.35,4920.95] vol=2.6x ATR=12.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 10:30:00 | 4829.97 | 4863.37 | 0.00 | T1 1.5R @ 4829.97 |
| Stop hit — per-position SL triggered | 2024-11-07 11:10:00 | 4848.15 | 4853.08 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-11-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 10:55:00 | 4821.85 | 4845.50 | 0.00 | ORB-short ORB[4828.00,4877.95] vol=1.5x ATR=9.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 11:20:00 | 4807.51 | 4838.31 | 0.00 | T1 1.5R @ 4807.51 |
| Target hit | 2024-11-08 13:30:00 | 4809.90 | 4808.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — SELL (started 2024-11-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:35:00 | 4739.55 | 4783.33 | 0.00 | ORB-short ORB[4776.20,4807.25] vol=1.6x ATR=12.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 11:20:00 | 4720.13 | 4762.98 | 0.00 | T1 1.5R @ 4720.13 |
| Stop hit — per-position SL triggered | 2024-11-12 12:00:00 | 4739.55 | 4755.37 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 4965.70 | 4925.36 | 0.00 | ORB-long ORB[4880.00,4950.00] vol=2.3x ATR=16.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:20:00 | 4990.75 | 4952.15 | 0.00 | T1 1.5R @ 4990.75 |
| Target hit | 2024-11-19 15:00:00 | 4979.05 | 4979.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2024-11-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:05:00 | 4933.40 | 4917.17 | 0.00 | ORB-long ORB[4882.00,4913.00] vol=1.8x ATR=14.41 |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 4918.99 | 4917.58 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 10:35:00 | 5060.10 | 5022.89 | 0.00 | ORB-long ORB[4999.00,5049.00] vol=2.1x ATR=14.01 |
| Stop hit — per-position SL triggered | 2024-11-25 11:25:00 | 5046.09 | 5032.62 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:05:00 | 4916.10 | 4920.26 | 0.00 | ORB-short ORB[4960.40,5014.35] vol=2.1x ATR=13.58 |
| Stop hit — per-position SL triggered | 2024-11-26 11:35:00 | 4929.68 | 4920.36 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 09:45:00 | 4834.60 | 4862.00 | 0.00 | ORB-short ORB[4850.00,4920.00] vol=5.2x ATR=15.79 |
| Stop hit — per-position SL triggered | 2024-11-28 10:00:00 | 4850.39 | 4859.02 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:55:00 | 4833.00 | 4845.72 | 0.00 | ORB-short ORB[4840.00,4874.90] vol=1.7x ATR=7.07 |
| Stop hit — per-position SL triggered | 2024-12-10 11:20:00 | 4840.07 | 4843.36 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 4805.85 | 4817.18 | 0.00 | ORB-short ORB[4813.10,4831.80] vol=1.6x ATR=8.85 |
| Stop hit — per-position SL triggered | 2024-12-16 11:40:00 | 4814.70 | 4817.99 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:40:00 | 4803.90 | 4775.13 | 0.00 | ORB-long ORB[4738.65,4794.55] vol=2.0x ATR=10.83 |
| Stop hit — per-position SL triggered | 2024-12-24 10:45:00 | 4793.07 | 4776.52 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-01-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:25:00 | 4749.35 | 4782.29 | 0.00 | ORB-short ORB[4794.05,4847.00] vol=1.9x ATR=13.31 |
| Stop hit — per-position SL triggered | 2025-01-01 10:50:00 | 4762.66 | 4774.10 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:55:00 | 4901.00 | 4889.46 | 0.00 | ORB-long ORB[4867.25,4895.00] vol=1.6x ATR=13.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:00:00 | 4921.30 | 4898.36 | 0.00 | T1 1.5R @ 4921.30 |
| Stop hit — per-position SL triggered | 2025-01-02 10:10:00 | 4901.00 | 4908.44 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:15:00 | 5253.35 | 5288.40 | 0.00 | ORB-short ORB[5288.90,5350.00] vol=2.0x ATR=14.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:20:00 | 5231.86 | 5286.37 | 0.00 | T1 1.5R @ 5231.86 |
| Stop hit — per-position SL triggered | 2025-01-06 12:10:00 | 5253.35 | 5258.79 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:55:00 | 5210.00 | 5252.47 | 0.00 | ORB-short ORB[5227.60,5292.00] vol=1.6x ATR=17.08 |
| Stop hit — per-position SL triggered | 2025-01-07 11:20:00 | 5227.08 | 5242.16 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 5132.45 | 5157.62 | 0.00 | ORB-short ORB[5141.25,5195.85] vol=1.9x ATR=12.04 |
| Stop hit — per-position SL triggered | 2025-01-09 10:50:00 | 5144.49 | 5156.66 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:00:00 | 4951.85 | 4997.41 | 0.00 | ORB-short ORB[4991.05,5024.20] vol=1.8x ATR=14.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:40:00 | 4930.70 | 4981.29 | 0.00 | T1 1.5R @ 4930.70 |
| Stop hit — per-position SL triggered | 2025-01-13 11:50:00 | 4951.85 | 4979.54 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:00:00 | 5025.60 | 4994.48 | 0.00 | ORB-long ORB[4950.05,5015.95] vol=2.4x ATR=14.74 |
| Stop hit — per-position SL triggered | 2025-01-14 11:10:00 | 5010.86 | 4995.86 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:25:00 | 5061.25 | 5022.64 | 0.00 | ORB-long ORB[4977.80,5013.95] vol=2.7x ATR=13.26 |
| Stop hit — per-position SL triggered | 2025-01-23 11:20:00 | 5047.99 | 5032.80 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:05:00 | 5186.50 | 5144.49 | 0.00 | ORB-long ORB[5080.55,5138.95] vol=1.8x ATR=16.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:25:00 | 5211.16 | 5160.26 | 0.00 | T1 1.5R @ 5211.16 |
| Stop hit — per-position SL triggered | 2025-01-29 13:45:00 | 5186.50 | 5192.46 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 5237.05 | 5211.26 | 0.00 | ORB-long ORB[5162.20,5225.00] vol=2.1x ATR=17.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:00:00 | 5262.96 | 5231.72 | 0.00 | T1 1.5R @ 5262.96 |
| Stop hit — per-position SL triggered | 2025-01-30 10:40:00 | 5237.05 | 5237.92 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-02-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 11:00:00 | 5472.05 | 5508.97 | 0.00 | ORB-short ORB[5500.00,5554.15] vol=2.4x ATR=17.56 |
| Stop hit — per-position SL triggered | 2025-02-04 11:15:00 | 5489.61 | 5508.00 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-02-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:35:00 | 5335.85 | 5346.67 | 0.00 | ORB-short ORB[5338.55,5391.95] vol=2.3x ATR=16.83 |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 5352.68 | 5345.67 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 11:05:00 | 4920.10 | 4949.63 | 0.00 | ORB-short ORB[4945.00,5000.00] vol=1.7x ATR=18.47 |
| Stop hit — per-position SL triggered | 2025-02-12 11:40:00 | 4938.57 | 4945.91 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 11:15:00 | 4722.15 | 4781.80 | 0.00 | ORB-short ORB[4790.25,4850.00] vol=2.4x ATR=16.22 |
| Stop hit — per-position SL triggered | 2025-02-14 11:20:00 | 4738.37 | 4780.73 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:50:00 | 4876.05 | 4835.48 | 0.00 | ORB-long ORB[4777.00,4820.00] vol=2.0x ATR=12.89 |
| Stop hit — per-position SL triggered | 2025-02-20 11:50:00 | 4863.16 | 4850.62 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-02-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 09:40:00 | 4996.75 | 4973.69 | 0.00 | ORB-long ORB[4915.55,4986.95] vol=1.9x ATR=15.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 10:45:00 | 5020.21 | 4992.25 | 0.00 | T1 1.5R @ 5020.21 |
| Stop hit — per-position SL triggered | 2025-02-24 11:15:00 | 4996.75 | 4994.80 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 5066.00 | 5041.93 | 0.00 | ORB-long ORB[4990.25,5053.35] vol=1.8x ATR=14.35 |
| Stop hit — per-position SL triggered | 2025-02-25 09:35:00 | 5051.65 | 5043.05 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 5117.00 | 5095.27 | 0.00 | ORB-long ORB[5065.70,5111.00] vol=1.6x ATR=13.40 |
| Stop hit — per-position SL triggered | 2025-03-18 09:40:00 | 5103.60 | 5096.27 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:20:00 | 5274.65 | 5256.90 | 0.00 | ORB-long ORB[5230.35,5268.80] vol=1.6x ATR=10.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:25:00 | 5290.67 | 5258.53 | 0.00 | T1 1.5R @ 5290.67 |
| Stop hit — per-position SL triggered | 2025-03-21 10:35:00 | 5274.65 | 5262.26 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 10:20:00 | 5488.75 | 5439.66 | 0.00 | ORB-long ORB[5402.40,5460.00] vol=2.7x ATR=16.95 |
| Stop hit — per-position SL triggered | 2025-03-25 10:25:00 | 5471.80 | 5444.26 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-04-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 10:50:00 | 5173.35 | 5163.85 | 0.00 | ORB-long ORB[5109.80,5172.95] vol=2.4x ATR=18.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-08 11:35:00 | 5201.71 | 5169.61 | 0.00 | T1 1.5R @ 5201.71 |
| Target hit | 2025-04-08 15:20:00 | 5230.70 | 5204.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-04-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 11:10:00 | 5472.00 | 5448.70 | 0.00 | ORB-long ORB[5399.00,5464.00] vol=2.2x ATR=11.21 |
| Stop hit — per-position SL triggered | 2025-04-15 11:40:00 | 5460.79 | 5451.79 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:15:00 | 5606.00 | 5581.95 | 0.00 | ORB-long ORB[5539.00,5601.50] vol=1.9x ATR=14.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:35:00 | 5627.81 | 5585.88 | 0.00 | T1 1.5R @ 5627.81 |
| Stop hit — per-position SL triggered | 2025-04-16 13:10:00 | 5606.00 | 5594.02 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:15:00 | 5645.00 | 5627.49 | 0.00 | ORB-long ORB[5576.00,5635.50] vol=1.6x ATR=10.30 |
| Stop hit — per-position SL triggered | 2025-04-17 11:20:00 | 5634.70 | 5627.81 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:55:00 | 5712.50 | 5686.70 | 0.00 | ORB-long ORB[5643.50,5695.50] vol=3.5x ATR=16.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:20:00 | 5737.11 | 5697.86 | 0.00 | T1 1.5R @ 5737.11 |
| Target hit | 2025-04-21 15:20:00 | 5802.50 | 5783.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2025-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:45:00 | 5660.00 | 5732.56 | 0.00 | ORB-short ORB[5759.50,5816.50] vol=4.9x ATR=23.25 |
| Stop hit — per-position SL triggered | 2025-04-23 10:50:00 | 5683.25 | 5719.58 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-04-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:40:00 | 5599.00 | 5631.71 | 0.00 | ORB-short ORB[5620.00,5668.00] vol=1.5x ATR=17.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:00:00 | 5572.66 | 5618.41 | 0.00 | T1 1.5R @ 5572.66 |
| Target hit | 2025-04-25 12:55:00 | 5582.50 | 5579.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — SELL (started 2025-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:55:00 | 5508.50 | 5590.19 | 0.00 | ORB-short ORB[5618.50,5691.50] vol=1.9x ATR=16.38 |
| Stop hit — per-position SL triggered | 2025-04-29 11:00:00 | 5524.88 | 5585.39 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:30:00 | 5555.00 | 5508.75 | 0.00 | ORB-long ORB[5440.50,5520.00] vol=2.4x ATR=17.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 09:45:00 | 5581.10 | 5535.91 | 0.00 | T1 1.5R @ 5581.10 |
| Stop hit — per-position SL triggered | 2025-05-06 09:50:00 | 5555.00 | 5538.12 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 10:35:00 | 5549.50 | 5522.19 | 0.00 | ORB-long ORB[5460.50,5539.50] vol=3.6x ATR=19.84 |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 5529.66 | 5529.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-08-14 10:35:00 | 4760.00 | 2024-08-14 10:40:00 | 4771.66 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-16 09:45:00 | 4805.00 | 2024-08-16 10:00:00 | 4791.07 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-19 10:15:00 | 4788.20 | 2024-08-19 10:45:00 | 4798.72 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-21 11:10:00 | 4903.00 | 2024-08-21 12:40:00 | 4916.39 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-08-21 11:10:00 | 4903.00 | 2024-08-21 14:40:00 | 4903.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-27 10:15:00 | 4852.00 | 2024-08-27 10:25:00 | 4859.68 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-08-28 10:20:00 | 4887.00 | 2024-08-28 10:45:00 | 4904.51 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-08-28 10:20:00 | 4887.00 | 2024-08-28 15:20:00 | 4947.45 | TARGET_HIT | 0.50 | 1.24% |
| BUY | retest1 | 2024-08-29 09:40:00 | 4977.55 | 2024-08-29 09:55:00 | 4965.42 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-04 10:25:00 | 4867.00 | 2024-09-04 13:05:00 | 4856.47 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-06 10:55:00 | 4772.50 | 2024-09-06 11:10:00 | 4754.36 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-06 10:55:00 | 4772.50 | 2024-09-06 15:20:00 | 4759.40 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2024-09-10 10:25:00 | 4729.95 | 2024-09-10 10:30:00 | 4712.32 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-10 10:25:00 | 4729.95 | 2024-09-10 10:35:00 | 4729.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:40:00 | 4750.00 | 2024-09-11 10:50:00 | 4766.61 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-11 10:40:00 | 4750.00 | 2024-09-11 10:55:00 | 4750.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-16 10:05:00 | 4929.00 | 2024-09-16 10:15:00 | 4913.23 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-17 09:30:00 | 4857.45 | 2024-09-17 09:45:00 | 4843.97 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-09-17 09:30:00 | 4857.45 | 2024-09-17 10:20:00 | 4857.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 10:50:00 | 4840.45 | 2024-09-18 11:25:00 | 4849.44 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-19 09:45:00 | 4916.00 | 2024-09-19 09:55:00 | 4901.34 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-23 10:25:00 | 4906.55 | 2024-09-23 10:35:00 | 4886.14 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-23 10:25:00 | 4906.55 | 2024-09-23 14:15:00 | 4893.00 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-09-26 09:55:00 | 4967.85 | 2024-09-26 10:00:00 | 4955.22 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-27 09:40:00 | 5020.80 | 2024-09-27 09:55:00 | 5044.05 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-09-27 09:40:00 | 5020.80 | 2024-09-27 10:10:00 | 5020.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-30 10:55:00 | 5000.40 | 2024-09-30 11:15:00 | 5015.82 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-01 10:00:00 | 4968.65 | 2024-10-01 10:15:00 | 4985.55 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-07 09:45:00 | 4688.70 | 2024-10-07 10:35:00 | 4702.93 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-10 10:00:00 | 4754.60 | 2024-10-10 10:35:00 | 4741.36 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-11 10:05:00 | 4780.00 | 2024-10-11 10:30:00 | 4761.06 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-10-14 10:20:00 | 4792.00 | 2024-10-14 11:00:00 | 4779.76 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-15 09:30:00 | 4745.65 | 2024-10-15 09:35:00 | 4755.89 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-16 11:15:00 | 4655.00 | 2024-10-16 11:50:00 | 4663.97 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-10-22 10:45:00 | 4787.20 | 2024-10-22 11:10:00 | 4801.61 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-23 10:50:00 | 4687.70 | 2024-10-23 11:10:00 | 4704.06 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-25 10:50:00 | 4523.55 | 2024-10-25 10:55:00 | 4539.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-31 10:55:00 | 4908.00 | 2024-10-31 11:05:00 | 4894.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-07 10:25:00 | 4848.15 | 2024-11-07 10:30:00 | 4829.97 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-11-07 10:25:00 | 4848.15 | 2024-11-07 11:10:00 | 4848.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-08 10:55:00 | 4821.85 | 2024-11-08 11:20:00 | 4807.51 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-11-08 10:55:00 | 4821.85 | 2024-11-08 13:30:00 | 4809.90 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2024-11-12 10:35:00 | 4739.55 | 2024-11-12 11:20:00 | 4720.13 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-11-12 10:35:00 | 4739.55 | 2024-11-12 12:00:00 | 4739.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 09:30:00 | 4965.70 | 2024-11-19 10:20:00 | 4990.75 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-11-19 09:30:00 | 4965.70 | 2024-11-19 15:00:00 | 4979.05 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2024-11-22 10:05:00 | 4933.40 | 2024-11-22 10:15:00 | 4918.99 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-25 10:35:00 | 5060.10 | 2024-11-25 11:25:00 | 5046.09 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-26 11:05:00 | 4916.10 | 2024-11-26 11:35:00 | 4929.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-28 09:45:00 | 4834.60 | 2024-11-28 10:00:00 | 4850.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-10 10:55:00 | 4833.00 | 2024-12-10 11:20:00 | 4840.07 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-12-16 11:00:00 | 4805.85 | 2024-12-16 11:40:00 | 4814.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-12-24 10:40:00 | 4803.90 | 2024-12-24 10:45:00 | 4793.07 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-01 10:25:00 | 4749.35 | 2025-01-01 10:50:00 | 4762.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-02 09:55:00 | 4901.00 | 2025-01-02 10:00:00 | 4921.30 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-01-02 09:55:00 | 4901.00 | 2025-01-02 10:10:00 | 4901.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 11:15:00 | 5253.35 | 2025-01-06 11:20:00 | 5231.86 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-01-06 11:15:00 | 5253.35 | 2025-01-06 12:10:00 | 5253.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-07 10:55:00 | 5210.00 | 2025-01-07 11:20:00 | 5227.08 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-09 10:45:00 | 5132.45 | 2025-01-09 10:50:00 | 5144.49 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-13 11:00:00 | 4951.85 | 2025-01-13 11:40:00 | 4930.70 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-13 11:00:00 | 4951.85 | 2025-01-13 11:50:00 | 4951.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-14 11:00:00 | 5025.60 | 2025-01-14 11:10:00 | 5010.86 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-23 10:25:00 | 5061.25 | 2025-01-23 11:20:00 | 5047.99 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-29 10:05:00 | 5186.50 | 2025-01-29 10:25:00 | 5211.16 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-01-29 10:05:00 | 5186.50 | 2025-01-29 13:45:00 | 5186.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-30 09:30:00 | 5237.05 | 2025-01-30 10:00:00 | 5262.96 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-01-30 09:30:00 | 5237.05 | 2025-01-30 10:40:00 | 5237.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-04 11:00:00 | 5472.05 | 2025-02-04 11:15:00 | 5489.61 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-07 10:35:00 | 5335.85 | 2025-02-07 11:15:00 | 5352.68 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-12 11:05:00 | 4920.10 | 2025-02-12 11:40:00 | 4938.57 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-14 11:15:00 | 4722.15 | 2025-02-14 11:20:00 | 4738.37 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-02-20 10:50:00 | 4876.05 | 2025-02-20 11:50:00 | 4863.16 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-02-24 09:40:00 | 4996.75 | 2025-02-24 10:45:00 | 5020.21 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-02-24 09:40:00 | 4996.75 | 2025-02-24 11:15:00 | 4996.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-25 09:30:00 | 5066.00 | 2025-02-25 09:35:00 | 5051.65 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-18 09:35:00 | 5117.00 | 2025-03-18 09:40:00 | 5103.60 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-21 10:20:00 | 5274.65 | 2025-03-21 10:25:00 | 5290.67 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-03-21 10:20:00 | 5274.65 | 2025-03-21 10:35:00 | 5274.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-25 10:20:00 | 5488.75 | 2025-03-25 10:25:00 | 5471.80 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-08 10:50:00 | 5173.35 | 2025-04-08 11:35:00 | 5201.71 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-04-08 10:50:00 | 5173.35 | 2025-04-08 15:20:00 | 5230.70 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2025-04-15 11:10:00 | 5472.00 | 2025-04-15 11:40:00 | 5460.79 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-04-16 11:15:00 | 5606.00 | 2025-04-16 11:35:00 | 5627.81 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-04-16 11:15:00 | 5606.00 | 2025-04-16 13:10:00 | 5606.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-17 11:15:00 | 5645.00 | 2025-04-17 11:20:00 | 5634.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-04-21 09:55:00 | 5712.50 | 2025-04-21 10:20:00 | 5737.11 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-21 09:55:00 | 5712.50 | 2025-04-21 15:20:00 | 5802.50 | TARGET_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2025-04-23 10:45:00 | 5660.00 | 2025-04-23 10:50:00 | 5683.25 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-04-25 09:40:00 | 5599.00 | 2025-04-25 10:00:00 | 5572.66 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-04-25 09:40:00 | 5599.00 | 2025-04-25 12:55:00 | 5582.50 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-04-29 10:55:00 | 5508.50 | 2025-04-29 11:00:00 | 5524.88 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-06 09:30:00 | 5555.00 | 2025-05-06 09:45:00 | 5581.10 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-05-06 09:30:00 | 5555.00 | 2025-05-06 09:50:00 | 5555.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-07 10:35:00 | 5549.50 | 2025-05-07 11:15:00 | 5529.66 | STOP_HIT | 1.00 | -0.36% |
