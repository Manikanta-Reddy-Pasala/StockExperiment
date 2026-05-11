# Avenue Supermarts Ltd. (DMART)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 9 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 56
- **Target hits / Stop hits / Partials:** 9 / 56 / 23
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 8.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 19 | 38.0% | 4 | 31 | 15 | 0.11% | 5.7% |
| BUY @ 2nd Alert (retest1) | 50 | 19 | 38.0% | 4 | 31 | 15 | 0.11% | 5.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 13 | 34.2% | 5 | 25 | 8 | 0.07% | 2.6% |
| SELL @ 2nd Alert (retest1) | 38 | 13 | 34.2% | 5 | 25 | 8 | 0.07% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 88 | 32 | 36.4% | 9 | 56 | 23 | 0.09% | 8.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 11:15:00 | 4675.25 | 4626.21 | 0.00 | ORB-long ORB[4597.10,4649.95] vol=3.4x ATR=14.72 |
| Stop hit — per-position SL triggered | 2024-05-15 11:20:00 | 4660.53 | 4628.54 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:05:00 | 4703.50 | 4689.83 | 0.00 | ORB-long ORB[4665.05,4694.35] vol=1.6x ATR=8.99 |
| Stop hit — per-position SL triggered | 2024-05-17 11:55:00 | 4694.51 | 4691.05 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:30:00 | 4751.85 | 4799.89 | 0.00 | ORB-short ORB[4775.90,4841.40] vol=1.7x ATR=18.60 |
| Stop hit — per-position SL triggered | 2024-05-24 10:45:00 | 4770.45 | 4792.48 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 11:05:00 | 4542.45 | 4555.83 | 0.00 | ORB-short ORB[4550.00,4588.85] vol=2.0x ATR=8.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:45:00 | 4529.56 | 4552.38 | 0.00 | T1 1.5R @ 4529.56 |
| Target hit | 2024-05-28 15:20:00 | 4463.80 | 4502.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:10:00 | 4411.25 | 4448.16 | 0.00 | ORB-short ORB[4441.45,4497.25] vol=2.4x ATR=12.49 |
| Stop hit — per-position SL triggered | 2024-05-30 11:30:00 | 4423.74 | 4446.31 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-06 11:15:00 | 4719.70 | 4783.80 | 0.00 | ORB-short ORB[4801.60,4861.95] vol=3.0x ATR=21.88 |
| Stop hit — per-position SL triggered | 2024-06-06 12:00:00 | 4741.58 | 4769.80 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:50:00 | 4801.00 | 4774.37 | 0.00 | ORB-long ORB[4734.80,4797.55] vol=2.1x ATR=16.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 09:55:00 | 4826.18 | 4792.44 | 0.00 | T1 1.5R @ 4826.18 |
| Target hit | 2024-06-18 11:25:00 | 4849.00 | 4856.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2024-06-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 09:45:00 | 5198.40 | 5123.51 | 0.00 | ORB-long ORB[5067.05,5125.00] vol=2.2x ATR=34.39 |
| Stop hit — per-position SL triggered | 2024-06-19 09:55:00 | 5164.01 | 5141.70 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 4815.15 | 4864.45 | 0.00 | ORB-short ORB[4848.00,4894.40] vol=2.0x ATR=16.13 |
| Stop hit — per-position SL triggered | 2024-06-25 12:30:00 | 4831.28 | 4854.41 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:55:00 | 4716.65 | 4742.79 | 0.00 | ORB-short ORB[4720.05,4780.00] vol=2.9x ATR=12.55 |
| Stop hit — per-position SL triggered | 2024-06-26 11:25:00 | 4729.20 | 4739.05 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 11:10:00 | 4836.30 | 4874.89 | 0.00 | ORB-short ORB[4848.00,4900.10] vol=3.4x ATR=18.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 11:40:00 | 4808.76 | 4864.22 | 0.00 | T1 1.5R @ 4808.76 |
| Stop hit — per-position SL triggered | 2024-06-28 12:20:00 | 4836.30 | 4855.91 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 4904.50 | 4877.49 | 0.00 | ORB-long ORB[4853.00,4896.75] vol=1.9x ATR=13.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:20:00 | 4925.30 | 4884.41 | 0.00 | T1 1.5R @ 4925.30 |
| Stop hit — per-position SL triggered | 2024-07-08 11:50:00 | 4904.50 | 4888.61 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:35:00 | 4980.00 | 4955.98 | 0.00 | ORB-long ORB[4909.00,4968.90] vol=2.8x ATR=15.13 |
| Stop hit — per-position SL triggered | 2024-07-12 09:55:00 | 4964.87 | 4970.02 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:00:00 | 5052.50 | 5048.35 | 0.00 | ORB-long ORB[5012.60,5049.65] vol=2.9x ATR=11.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:00:00 | 5069.70 | 5051.33 | 0.00 | T1 1.5R @ 5069.70 |
| Stop hit — per-position SL triggered | 2024-07-23 11:05:00 | 5052.50 | 5051.87 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 09:35:00 | 4871.00 | 4854.99 | 0.00 | ORB-long ORB[4824.05,4869.00] vol=2.4x ATR=19.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:40:00 | 4900.17 | 4874.23 | 0.00 | T1 1.5R @ 4900.17 |
| Target hit | 2024-08-05 10:20:00 | 4913.10 | 4930.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 11:15:00 | 4961.15 | 4941.63 | 0.00 | ORB-long ORB[4913.30,4959.00] vol=1.7x ATR=11.99 |
| Stop hit — per-position SL triggered | 2024-08-14 11:45:00 | 4949.16 | 4944.22 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:55:00 | 4915.15 | 4940.87 | 0.00 | ORB-short ORB[4946.55,4996.75] vol=2.0x ATR=11.47 |
| Stop hit — per-position SL triggered | 2024-08-16 11:35:00 | 4926.62 | 4936.77 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 4989.55 | 5008.12 | 0.00 | ORB-short ORB[5000.70,5048.00] vol=1.6x ATR=14.23 |
| Stop hit — per-position SL triggered | 2024-08-19 09:40:00 | 5003.78 | 5007.89 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:50:00 | 5077.50 | 5036.10 | 0.00 | ORB-long ORB[4987.70,5020.00] vol=3.8x ATR=15.95 |
| Stop hit — per-position SL triggered | 2024-08-20 10:25:00 | 5061.55 | 5045.89 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 11:10:00 | 5090.45 | 5119.49 | 0.00 | ORB-short ORB[5095.00,5164.00] vol=2.4x ATR=10.90 |
| Stop hit — per-position SL triggered | 2024-08-21 11:15:00 | 5101.35 | 5117.91 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:50:00 | 5096.70 | 5104.02 | 0.00 | ORB-short ORB[5100.30,5145.00] vol=2.1x ATR=12.84 |
| Stop hit — per-position SL triggered | 2024-08-22 12:15:00 | 5109.54 | 5101.47 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:40:00 | 4983.85 | 5029.34 | 0.00 | ORB-short ORB[5037.85,5074.95] vol=2.4x ATR=17.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:55:00 | 4957.93 | 5015.26 | 0.00 | T1 1.5R @ 4957.93 |
| Target hit | 2024-08-23 15:20:00 | 4911.65 | 4929.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-08-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:35:00 | 4935.10 | 4958.10 | 0.00 | ORB-short ORB[4949.90,4995.50] vol=2.3x ATR=15.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:00:00 | 4912.48 | 4945.97 | 0.00 | T1 1.5R @ 4912.48 |
| Target hit | 2024-08-28 12:55:00 | 4930.10 | 4929.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2024-09-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:40:00 | 5035.00 | 5016.05 | 0.00 | ORB-long ORB[5000.00,5029.00] vol=2.1x ATR=8.25 |
| Stop hit — per-position SL triggered | 2024-09-03 11:35:00 | 5026.75 | 5020.02 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 11:05:00 | 5014.25 | 4991.68 | 0.00 | ORB-long ORB[4960.35,4999.85] vol=2.8x ATR=10.48 |
| Stop hit — per-position SL triggered | 2024-09-04 11:20:00 | 5003.77 | 4996.49 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:35:00 | 5155.00 | 5130.04 | 0.00 | ORB-long ORB[5077.10,5136.00] vol=2.0x ATR=14.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:40:00 | 5177.49 | 5147.42 | 0.00 | T1 1.5R @ 5177.49 |
| Target hit | 2024-09-05 15:20:00 | 5294.65 | 5248.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2024-09-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:10:00 | 5354.70 | 5371.16 | 0.00 | ORB-short ORB[5361.40,5422.45] vol=2.6x ATR=21.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:20:00 | 5322.46 | 5366.86 | 0.00 | T1 1.5R @ 5322.46 |
| Stop hit — per-position SL triggered | 2024-09-10 10:55:00 | 5354.70 | 5362.02 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:45:00 | 5422.25 | 5360.75 | 0.00 | ORB-long ORB[5293.10,5357.90] vol=3.1x ATR=21.22 |
| Stop hit — per-position SL triggered | 2024-09-11 11:35:00 | 5401.03 | 5389.90 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:55:00 | 5279.70 | 5307.07 | 0.00 | ORB-short ORB[5307.05,5376.05] vol=2.8x ATR=14.34 |
| Stop hit — per-position SL triggered | 2024-09-12 11:35:00 | 5294.04 | 5296.04 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:55:00 | 5205.15 | 5218.03 | 0.00 | ORB-short ORB[5209.55,5254.00] vol=8.8x ATR=11.35 |
| Stop hit — per-position SL triggered | 2024-09-17 11:25:00 | 5216.50 | 5217.78 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 11:15:00 | 5276.20 | 5299.04 | 0.00 | ORB-short ORB[5285.00,5363.00] vol=3.2x ATR=12.75 |
| Stop hit — per-position SL triggered | 2024-09-23 12:20:00 | 5288.95 | 5291.24 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:55:00 | 5030.80 | 5074.09 | 0.00 | ORB-short ORB[5090.00,5138.00] vol=2.2x ATR=13.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 11:35:00 | 5010.75 | 5062.90 | 0.00 | T1 1.5R @ 5010.75 |
| Target hit | 2024-10-01 15:20:00 | 4952.65 | 4989.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2024-10-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:00:00 | 4111.00 | 4136.94 | 0.00 | ORB-short ORB[4130.85,4180.00] vol=1.5x ATR=10.33 |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 4121.33 | 4134.26 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 09:45:00 | 4004.00 | 3992.56 | 0.00 | ORB-long ORB[3965.90,4000.00] vol=7.1x ATR=10.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 4020.42 | 3997.78 | 0.00 | T1 1.5R @ 4020.42 |
| Stop hit — per-position SL triggered | 2024-10-22 10:35:00 | 4004.00 | 3999.21 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-11-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:25:00 | 3908.60 | 3848.90 | 0.00 | ORB-long ORB[3851.00,3899.65] vol=1.8x ATR=14.32 |
| Stop hit — per-position SL triggered | 2024-11-11 10:30:00 | 3894.28 | 3856.03 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-11-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:25:00 | 3871.00 | 3878.29 | 0.00 | ORB-short ORB[3877.05,3910.00] vol=1.9x ATR=9.30 |
| Stop hit — per-position SL triggered | 2024-11-12 11:50:00 | 3880.30 | 3876.24 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 11:05:00 | 3833.70 | 3811.55 | 0.00 | ORB-long ORB[3769.00,3812.75] vol=1.9x ATR=12.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 11:15:00 | 3852.40 | 3815.48 | 0.00 | T1 1.5R @ 3852.40 |
| Stop hit — per-position SL triggered | 2024-11-14 11:45:00 | 3833.70 | 3819.38 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:55:00 | 3823.95 | 3834.64 | 0.00 | ORB-short ORB[3824.70,3853.00] vol=2.5x ATR=8.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 12:00:00 | 3810.62 | 3831.14 | 0.00 | T1 1.5R @ 3810.62 |
| Target hit | 2024-12-10 15:00:00 | 3814.05 | 3809.01 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — SELL (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:15:00 | 3619.40 | 3640.43 | 0.00 | ORB-short ORB[3621.00,3654.95] vol=2.7x ATR=8.05 |
| Stop hit — per-position SL triggered | 2024-12-16 12:00:00 | 3627.45 | 3633.85 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:35:00 | 3578.00 | 3599.11 | 0.00 | ORB-short ORB[3582.15,3629.95] vol=1.6x ATR=8.18 |
| Stop hit — per-position SL triggered | 2024-12-17 10:45:00 | 3586.18 | 3596.96 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:35:00 | 3503.10 | 3483.44 | 0.00 | ORB-long ORB[3455.70,3500.00] vol=1.9x ATR=11.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 10:10:00 | 3520.88 | 3494.85 | 0.00 | T1 1.5R @ 3520.88 |
| Stop hit — per-position SL triggered | 2024-12-19 10:50:00 | 3503.10 | 3499.25 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:05:00 | 3579.45 | 3561.67 | 0.00 | ORB-long ORB[3536.55,3560.15] vol=3.8x ATR=9.58 |
| Stop hit — per-position SL triggered | 2025-01-01 11:35:00 | 3569.87 | 3565.29 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:40:00 | 3584.15 | 3571.06 | 0.00 | ORB-long ORB[3550.25,3583.90] vol=1.5x ATR=8.20 |
| Stop hit — per-position SL triggered | 2025-01-02 09:45:00 | 3575.95 | 3572.21 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 09:55:00 | 3801.35 | 3820.26 | 0.00 | ORB-short ORB[3805.00,3860.00] vol=5.2x ATR=11.91 |
| Stop hit — per-position SL triggered | 2025-01-09 10:20:00 | 3813.26 | 3816.94 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 11:05:00 | 3639.95 | 3610.69 | 0.00 | ORB-long ORB[3570.00,3603.95] vol=1.8x ATR=9.89 |
| Stop hit — per-position SL triggered | 2025-01-16 11:20:00 | 3630.06 | 3612.50 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-01-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:55:00 | 3617.20 | 3603.92 | 0.00 | ORB-long ORB[3580.95,3613.35] vol=4.1x ATR=8.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 11:00:00 | 3629.57 | 3605.28 | 0.00 | T1 1.5R @ 3629.57 |
| Stop hit — per-position SL triggered | 2025-01-17 11:05:00 | 3617.20 | 3605.61 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:45:00 | 3601.30 | 3613.68 | 0.00 | ORB-short ORB[3609.00,3630.00] vol=1.8x ATR=7.47 |
| Stop hit — per-position SL triggered | 2025-01-21 10:10:00 | 3608.77 | 3609.84 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 10:45:00 | 3613.80 | 3589.47 | 0.00 | ORB-long ORB[3574.05,3604.00] vol=1.6x ATR=11.20 |
| Stop hit — per-position SL triggered | 2025-01-22 10:55:00 | 3602.60 | 3590.95 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 11:05:00 | 3590.90 | 3581.60 | 0.00 | ORB-long ORB[3530.00,3560.00] vol=3.2x ATR=9.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 12:35:00 | 3605.29 | 3585.76 | 0.00 | T1 1.5R @ 3605.29 |
| Stop hit — per-position SL triggered | 2025-01-23 14:35:00 | 3590.90 | 3589.96 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-02-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 11:10:00 | 3708.10 | 3720.81 | 0.00 | ORB-short ORB[3720.00,3764.10] vol=2.8x ATR=11.52 |
| Stop hit — per-position SL triggered | 2025-02-07 11:20:00 | 3719.62 | 3720.01 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-02-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 09:40:00 | 3738.05 | 3720.25 | 0.00 | ORB-long ORB[3688.50,3725.00] vol=3.6x ATR=12.29 |
| Stop hit — per-position SL triggered | 2025-02-14 09:55:00 | 3725.76 | 3723.69 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 10:55:00 | 3662.60 | 3638.12 | 0.00 | ORB-long ORB[3606.25,3648.95] vol=2.2x ATR=10.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 11:40:00 | 3678.18 | 3650.73 | 0.00 | T1 1.5R @ 3678.18 |
| Stop hit — per-position SL triggered | 2025-02-18 12:35:00 | 3662.60 | 3666.22 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-02-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:55:00 | 3697.00 | 3673.86 | 0.00 | ORB-long ORB[3627.45,3676.60] vol=1.7x ATR=11.36 |
| Stop hit — per-position SL triggered | 2025-02-19 10:45:00 | 3685.64 | 3686.54 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-20 10:15:00 | 3648.00 | 3675.08 | 0.00 | ORB-short ORB[3656.40,3708.95] vol=1.5x ATR=12.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 10:55:00 | 3629.65 | 3667.17 | 0.00 | T1 1.5R @ 3629.65 |
| Stop hit — per-position SL triggered | 2025-02-20 11:25:00 | 3648.00 | 3660.10 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-02-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 10:20:00 | 3467.60 | 3478.31 | 0.00 | ORB-short ORB[3490.00,3532.55] vol=5.6x ATR=11.39 |
| Stop hit — per-position SL triggered | 2025-02-28 10:40:00 | 3478.99 | 3476.72 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-03-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 09:45:00 | 3371.05 | 3383.82 | 0.00 | ORB-short ORB[3379.70,3430.00] vol=2.1x ATR=12.62 |
| Stop hit — per-position SL triggered | 2025-03-03 09:55:00 | 3383.67 | 3382.76 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:55:00 | 3447.90 | 3446.77 | 0.00 | ORB-long ORB[3408.40,3447.45] vol=1.6x ATR=9.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 12:10:00 | 3462.84 | 3450.32 | 0.00 | T1 1.5R @ 3462.84 |
| Target hit | 2025-03-05 14:55:00 | 3468.00 | 3468.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — BUY (started 2025-03-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:00:00 | 3606.35 | 3589.91 | 0.00 | ORB-long ORB[3549.95,3591.80] vol=6.7x ATR=15.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 10:05:00 | 3629.76 | 3592.17 | 0.00 | T1 1.5R @ 3629.76 |
| Stop hit — per-position SL triggered | 2025-03-07 12:00:00 | 3606.35 | 3605.40 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:10:00 | 3598.95 | 3576.60 | 0.00 | ORB-long ORB[3551.55,3591.45] vol=2.2x ATR=9.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 11:50:00 | 3612.45 | 3585.34 | 0.00 | T1 1.5R @ 3612.45 |
| Stop hit — per-position SL triggered | 2025-03-11 13:15:00 | 3598.95 | 3589.38 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:50:00 | 3681.00 | 3664.69 | 0.00 | ORB-long ORB[3650.00,3678.95] vol=1.6x ATR=11.29 |
| Stop hit — per-position SL triggered | 2025-03-12 10:00:00 | 3669.71 | 3665.17 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 09:55:00 | 4021.40 | 3981.78 | 0.00 | ORB-long ORB[3932.45,3966.45] vol=1.5x ATR=14.76 |
| Stop hit — per-position SL triggered | 2025-03-27 10:10:00 | 4006.64 | 3994.28 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-04-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-15 11:10:00 | 4089.70 | 4146.88 | 0.00 | ORB-short ORB[4157.00,4207.30] vol=1.9x ATR=11.26 |
| Stop hit — per-position SL triggered | 2025-04-15 11:20:00 | 4100.96 | 4144.33 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:35:00 | 4262.00 | 4233.75 | 0.00 | ORB-long ORB[4200.20,4240.00] vol=2.0x ATR=13.72 |
| Stop hit — per-position SL triggered | 2025-04-16 09:40:00 | 4248.28 | 4234.98 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:05:00 | 4296.00 | 4248.76 | 0.00 | ORB-long ORB[4192.20,4243.40] vol=3.0x ATR=11.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:15:00 | 4312.65 | 4259.93 | 0.00 | T1 1.5R @ 4312.65 |
| Stop hit — per-position SL triggered | 2025-04-17 11:25:00 | 4296.00 | 4265.01 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:50:00 | 4547.90 | 4535.61 | 0.00 | ORB-long ORB[4508.00,4539.00] vol=2.1x ATR=13.27 |
| Stop hit — per-position SL triggered | 2025-04-24 10:40:00 | 4534.63 | 4538.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 11:15:00 | 4675.25 | 2024-05-15 11:20:00 | 4660.53 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-17 11:05:00 | 4703.50 | 2024-05-17 11:55:00 | 4694.51 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-24 10:30:00 | 4751.85 | 2024-05-24 10:45:00 | 4770.45 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-28 11:05:00 | 4542.45 | 2024-05-28 11:45:00 | 4529.56 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-05-28 11:05:00 | 4542.45 | 2024-05-28 15:20:00 | 4463.80 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2024-05-30 11:10:00 | 4411.25 | 2024-05-30 11:30:00 | 4423.74 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-06 11:15:00 | 4719.70 | 2024-06-06 12:00:00 | 4741.58 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-06-18 09:50:00 | 4801.00 | 2024-06-18 09:55:00 | 4826.18 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-06-18 09:50:00 | 4801.00 | 2024-06-18 11:25:00 | 4849.00 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2024-06-19 09:45:00 | 5198.40 | 2024-06-19 09:55:00 | 5164.01 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2024-06-25 11:15:00 | 4815.15 | 2024-06-25 12:30:00 | 4831.28 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-26 10:55:00 | 4716.65 | 2024-06-26 11:25:00 | 4729.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-28 11:10:00 | 4836.30 | 2024-06-28 11:40:00 | 4808.76 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-06-28 11:10:00 | 4836.30 | 2024-06-28 12:20:00 | 4836.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-08 11:10:00 | 4904.50 | 2024-07-08 11:20:00 | 4925.30 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-08 11:10:00 | 4904.50 | 2024-07-08 11:50:00 | 4904.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-12 09:35:00 | 4980.00 | 2024-07-12 09:55:00 | 4964.87 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-23 10:00:00 | 5052.50 | 2024-07-23 11:00:00 | 5069.70 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-07-23 10:00:00 | 5052.50 | 2024-07-23 11:05:00 | 5052.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-05 09:35:00 | 4871.00 | 2024-08-05 09:40:00 | 4900.17 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-08-05 09:35:00 | 4871.00 | 2024-08-05 10:20:00 | 4913.10 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2024-08-14 11:15:00 | 4961.15 | 2024-08-14 11:45:00 | 4949.16 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-16 10:55:00 | 4915.15 | 2024-08-16 11:35:00 | 4926.62 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-19 09:35:00 | 4989.55 | 2024-08-19 09:40:00 | 5003.78 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-20 09:50:00 | 5077.50 | 2024-08-20 10:25:00 | 5061.55 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-21 11:10:00 | 5090.45 | 2024-08-21 11:15:00 | 5101.35 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-22 09:50:00 | 5096.70 | 2024-08-22 12:15:00 | 5109.54 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-23 10:40:00 | 4983.85 | 2024-08-23 10:55:00 | 4957.93 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-08-23 10:40:00 | 4983.85 | 2024-08-23 15:20:00 | 4911.65 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2024-08-28 09:35:00 | 4935.10 | 2024-08-28 10:00:00 | 4912.48 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-28 09:35:00 | 4935.10 | 2024-08-28 12:55:00 | 4930.10 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-09-03 10:40:00 | 5035.00 | 2024-09-03 11:35:00 | 5026.75 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-09-04 11:05:00 | 5014.25 | 2024-09-04 11:20:00 | 5003.77 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-05 09:35:00 | 5155.00 | 2024-09-05 09:40:00 | 5177.49 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-05 09:35:00 | 5155.00 | 2024-09-05 15:20:00 | 5294.65 | TARGET_HIT | 0.50 | 2.71% |
| SELL | retest1 | 2024-09-10 10:10:00 | 5354.70 | 2024-09-10 10:20:00 | 5322.46 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-09-10 10:10:00 | 5354.70 | 2024-09-10 10:55:00 | 5354.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:45:00 | 5422.25 | 2024-09-11 11:35:00 | 5401.03 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-09-12 10:55:00 | 5279.70 | 2024-09-12 11:35:00 | 5294.04 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-17 10:55:00 | 5205.15 | 2024-09-17 11:25:00 | 5216.50 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-23 11:15:00 | 5276.20 | 2024-09-23 12:20:00 | 5288.95 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-10-01 10:55:00 | 5030.80 | 2024-10-01 11:35:00 | 5010.75 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-01 10:55:00 | 5030.80 | 2024-10-01 15:20:00 | 4952.65 | TARGET_HIT | 0.50 | 1.55% |
| SELL | retest1 | 2024-10-17 10:00:00 | 4111.00 | 2024-10-17 10:15:00 | 4121.33 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-10-22 09:45:00 | 4004.00 | 2024-10-22 10:15:00 | 4020.42 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-10-22 09:45:00 | 4004.00 | 2024-10-22 10:35:00 | 4004.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 10:25:00 | 3908.60 | 2024-11-11 10:30:00 | 3894.28 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-11-12 10:25:00 | 3871.00 | 2024-11-12 11:50:00 | 3880.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-14 11:05:00 | 3833.70 | 2024-11-14 11:15:00 | 3852.40 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-11-14 11:05:00 | 3833.70 | 2024-11-14 11:45:00 | 3833.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-10 10:55:00 | 3823.95 | 2024-12-10 12:00:00 | 3810.62 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-12-10 10:55:00 | 3823.95 | 2024-12-10 15:00:00 | 3814.05 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2024-12-16 11:15:00 | 3619.40 | 2024-12-16 12:00:00 | 3627.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-17 10:35:00 | 3578.00 | 2024-12-17 10:45:00 | 3586.18 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-19 09:35:00 | 3503.10 | 2024-12-19 10:10:00 | 3520.88 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-12-19 09:35:00 | 3503.10 | 2024-12-19 10:50:00 | 3503.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 11:05:00 | 3579.45 | 2025-01-01 11:35:00 | 3569.87 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-02 09:40:00 | 3584.15 | 2025-01-02 09:45:00 | 3575.95 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-09 09:55:00 | 3801.35 | 2025-01-09 10:20:00 | 3813.26 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-16 11:05:00 | 3639.95 | 2025-01-16 11:20:00 | 3630.06 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-17 10:55:00 | 3617.20 | 2025-01-17 11:00:00 | 3629.57 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-01-17 10:55:00 | 3617.20 | 2025-01-17 11:05:00 | 3617.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 09:45:00 | 3601.30 | 2025-01-21 10:10:00 | 3608.77 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-01-22 10:45:00 | 3613.80 | 2025-01-22 10:55:00 | 3602.60 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-23 11:05:00 | 3590.90 | 2025-01-23 12:35:00 | 3605.29 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-23 11:05:00 | 3590.90 | 2025-01-23 14:35:00 | 3590.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-07 11:10:00 | 3708.10 | 2025-02-07 11:20:00 | 3719.62 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-02-14 09:40:00 | 3738.05 | 2025-02-14 09:55:00 | 3725.76 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-18 10:55:00 | 3662.60 | 2025-02-18 11:40:00 | 3678.18 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-02-18 10:55:00 | 3662.60 | 2025-02-18 12:35:00 | 3662.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-19 09:55:00 | 3697.00 | 2025-02-19 10:45:00 | 3685.64 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-02-20 10:15:00 | 3648.00 | 2025-02-20 10:55:00 | 3629.65 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-02-20 10:15:00 | 3648.00 | 2025-02-20 11:25:00 | 3648.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-28 10:20:00 | 3467.60 | 2025-02-28 10:40:00 | 3478.99 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-03-03 09:45:00 | 3371.05 | 2025-03-03 09:55:00 | 3383.67 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-05 09:55:00 | 3447.90 | 2025-03-05 12:10:00 | 3462.84 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-03-05 09:55:00 | 3447.90 | 2025-03-05 14:55:00 | 3468.00 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-03-07 10:00:00 | 3606.35 | 2025-03-07 10:05:00 | 3629.76 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-03-07 10:00:00 | 3606.35 | 2025-03-07 12:00:00 | 3606.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-11 11:10:00 | 3598.95 | 2025-03-11 11:50:00 | 3612.45 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-03-11 11:10:00 | 3598.95 | 2025-03-11 13:15:00 | 3598.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-12 09:50:00 | 3681.00 | 2025-03-12 10:00:00 | 3669.71 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-27 09:55:00 | 4021.40 | 2025-03-27 10:10:00 | 4006.64 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-15 11:10:00 | 4089.70 | 2025-04-15 11:20:00 | 4100.96 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-16 09:35:00 | 4262.00 | 2025-04-16 09:40:00 | 4248.28 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-17 11:05:00 | 4296.00 | 2025-04-17 11:15:00 | 4312.65 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-04-17 11:05:00 | 4296.00 | 2025-04-17 11:25:00 | 4296.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-24 09:50:00 | 4547.90 | 2025-04-24 10:40:00 | 4534.63 | STOP_HIT | 1.00 | -0.29% |
