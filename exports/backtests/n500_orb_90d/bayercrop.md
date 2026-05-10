# Bayer Cropscience Ltd. (BAYERCROP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4600.10
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 17
- **Target hits / Stop hits / Partials:** 6 / 17 / 10
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 6.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 4 | 28.6% | 1 | 10 | 3 | 0.06% | 0.9% |
| BUY @ 2nd Alert (retest1) | 14 | 4 | 28.6% | 1 | 10 | 3 | 0.06% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 12 | 63.2% | 5 | 7 | 7 | 0.29% | 5.5% |
| SELL @ 2nd Alert (retest1) | 19 | 12 | 63.2% | 5 | 7 | 7 | 0.29% | 5.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 16 | 48.5% | 6 | 17 | 10 | 0.19% | 6.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 4559.00 | 4514.71 | 0.00 | ORB-long ORB[4483.20,4515.10] vol=1.7x ATR=19.33 |
| Stop hit — per-position SL triggered | 2026-02-09 12:00:00 | 4539.67 | 4525.70 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 4700.00 | 4651.06 | 0.00 | ORB-long ORB[4601.00,4662.60] vol=2.6x ATR=21.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 14:20:00 | 4731.53 | 4696.75 | 0.00 | T1 1.5R @ 4731.53 |
| Stop hit — per-position SL triggered | 2026-02-10 14:55:00 | 4700.00 | 4700.91 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 4751.90 | 4710.34 | 0.00 | ORB-long ORB[4640.00,4709.20] vol=1.9x ATR=14.62 |
| Stop hit — per-position SL triggered | 2026-02-17 15:20:00 | 4750.10 | 4737.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 4572.60 | 4545.43 | 0.00 | ORB-long ORB[4490.60,4542.00] vol=2.0x ATR=19.37 |
| Stop hit — per-position SL triggered | 2026-03-06 10:00:00 | 4553.23 | 4551.75 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:05:00 | 4529.60 | 4513.38 | 0.00 | ORB-long ORB[4485.00,4524.90] vol=1.5x ATR=12.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:15:00 | 4548.35 | 4518.15 | 0.00 | T1 1.5R @ 4548.35 |
| Target hit | 2026-03-10 15:20:00 | 4613.00 | 4578.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:50:00 | 4600.80 | 4614.41 | 0.00 | ORB-short ORB[4604.30,4639.70] vol=1.6x ATR=10.53 |
| Stop hit — per-position SL triggered | 2026-03-11 09:55:00 | 4611.33 | 4614.55 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:05:00 | 4517.60 | 4490.88 | 0.00 | ORB-long ORB[4456.60,4502.40] vol=3.2x ATR=10.31 |
| Stop hit — per-position SL triggered | 2026-03-12 11:20:00 | 4507.29 | 4493.04 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:30:00 | 4462.60 | 4482.07 | 0.00 | ORB-short ORB[4463.50,4522.50] vol=1.8x ATR=12.29 |
| Stop hit — per-position SL triggered | 2026-03-13 12:20:00 | 4474.89 | 4472.10 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:50:00 | 4423.10 | 4459.02 | 0.00 | ORB-short ORB[4456.20,4516.60] vol=2.7x ATR=15.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 11:00:00 | 4400.21 | 4447.27 | 0.00 | T1 1.5R @ 4400.21 |
| Stop hit — per-position SL triggered | 2026-03-16 11:05:00 | 4423.10 | 4447.13 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:40:00 | 4420.80 | 4448.34 | 0.00 | ORB-short ORB[4428.00,4478.90] vol=2.1x ATR=16.77 |
| Stop hit — per-position SL triggered | 2026-03-27 11:10:00 | 4437.57 | 4444.73 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:40:00 | 4798.00 | 4754.01 | 0.00 | ORB-long ORB[4700.00,4753.30] vol=4.7x ATR=18.17 |
| Stop hit — per-position SL triggered | 2026-04-08 09:50:00 | 4779.83 | 4757.27 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:20:00 | 4754.60 | 4764.10 | 0.00 | ORB-short ORB[4775.30,4837.80] vol=17.6x ATR=16.44 |
| Stop hit — per-position SL triggered | 2026-04-09 10:35:00 | 4771.04 | 4764.09 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:45:00 | 4834.70 | 4801.11 | 0.00 | ORB-long ORB[4745.10,4805.00] vol=1.6x ATR=18.83 |
| Stop hit — per-position SL triggered | 2026-04-13 10:00:00 | 4815.87 | 4808.11 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 4920.60 | 4894.34 | 0.00 | ORB-long ORB[4853.30,4894.90] vol=3.8x ATR=17.07 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 4903.53 | 4898.69 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:55:00 | 4813.00 | 4827.43 | 0.00 | ORB-short ORB[4820.00,4863.90] vol=2.8x ATR=10.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:50:00 | 4796.55 | 4814.70 | 0.00 | T1 1.5R @ 4796.55 |
| Target hit | 2026-04-22 15:20:00 | 4789.90 | 4797.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-04-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:40:00 | 4723.30 | 4744.45 | 0.00 | ORB-short ORB[4745.40,4789.90] vol=2.1x ATR=12.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:05:00 | 4704.03 | 4708.53 | 0.00 | T1 1.5R @ 4704.03 |
| Target hit | 2026-04-24 15:20:00 | 4672.00 | 4703.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2026-04-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:20:00 | 4728.10 | 4711.48 | 0.00 | ORB-long ORB[4687.00,4717.60] vol=2.8x ATR=9.09 |
| Stop hit — per-position SL triggered | 2026-04-28 10:40:00 | 4719.01 | 4714.97 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:40:00 | 4725.00 | 4697.60 | 0.00 | ORB-long ORB[4672.00,4718.00] vol=1.9x ATR=12.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:10:00 | 4743.68 | 4705.58 | 0.00 | T1 1.5R @ 4743.68 |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 4725.00 | 4706.31 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 4755.70 | 4797.88 | 0.00 | ORB-short ORB[4787.90,4832.80] vol=1.6x ATR=15.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 12:05:00 | 4732.30 | 4780.82 | 0.00 | T1 1.5R @ 4732.30 |
| Target hit | 2026-05-04 15:20:00 | 4709.00 | 4736.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2026-05-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:00:00 | 4680.00 | 4693.13 | 0.00 | ORB-short ORB[4680.20,4747.10] vol=3.3x ATR=14.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:25:00 | 4658.03 | 4686.16 | 0.00 | T1 1.5R @ 4658.03 |
| Target hit | 2026-05-05 15:20:00 | 4640.20 | 4656.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 4640.40 | 4658.42 | 0.00 | ORB-short ORB[4649.90,4691.80] vol=1.9x ATR=8.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 12:30:00 | 4627.35 | 4654.78 | 0.00 | T1 1.5R @ 4627.35 |
| Stop hit — per-position SL triggered | 2026-05-06 13:05:00 | 4640.40 | 4651.65 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 4650.10 | 4664.90 | 0.00 | ORB-short ORB[4652.20,4687.00] vol=4.5x ATR=8.77 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 4658.87 | 4660.70 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2026-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:30:00 | 4633.00 | 4640.38 | 0.00 | ORB-short ORB[4639.00,4667.70] vol=3.3x ATR=11.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:55:00 | 4615.25 | 4635.51 | 0.00 | T1 1.5R @ 4615.25 |
| Target hit | 2026-05-08 15:20:00 | 4606.00 | 4620.33 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 4559.00 | 2026-02-09 12:00:00 | 4539.67 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-10 09:45:00 | 4700.00 | 2026-02-10 14:20:00 | 4731.53 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-02-10 09:45:00 | 4700.00 | 2026-02-10 14:55:00 | 4700.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:05:00 | 4751.90 | 2026-02-17 15:20:00 | 4750.10 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest1 | 2026-03-06 09:30:00 | 4572.60 | 2026-03-06 10:00:00 | 4553.23 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-10 10:05:00 | 4529.60 | 2026-03-10 10:15:00 | 4548.35 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-03-10 10:05:00 | 4529.60 | 2026-03-10 15:20:00 | 4613.00 | TARGET_HIT | 0.50 | 1.84% |
| SELL | retest1 | 2026-03-11 09:50:00 | 4600.80 | 2026-03-11 09:55:00 | 4611.33 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-03-12 11:05:00 | 4517.60 | 2026-03-12 11:20:00 | 4507.29 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-13 10:30:00 | 4462.60 | 2026-03-13 12:20:00 | 4474.89 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-16 10:50:00 | 4423.10 | 2026-03-16 11:00:00 | 4400.21 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-03-16 10:50:00 | 4423.10 | 2026-03-16 11:05:00 | 4423.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 10:40:00 | 4420.80 | 2026-03-27 11:10:00 | 4437.57 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-08 09:40:00 | 4798.00 | 2026-04-08 09:50:00 | 4779.83 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-09 10:20:00 | 4754.60 | 2026-04-09 10:35:00 | 4771.04 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-13 09:45:00 | 4834.70 | 2026-04-13 10:00:00 | 4815.87 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-21 09:30:00 | 4920.60 | 2026-04-21 09:40:00 | 4903.53 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-22 09:55:00 | 4813.00 | 2026-04-22 10:50:00 | 4796.55 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-04-22 09:55:00 | 4813.00 | 2026-04-22 15:20:00 | 4789.90 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-24 10:40:00 | 4723.30 | 2026-04-24 11:05:00 | 4704.03 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-24 10:40:00 | 4723.30 | 2026-04-24 15:20:00 | 4672.00 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2026-04-28 10:20:00 | 4728.10 | 2026-04-28 10:40:00 | 4719.01 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-04-30 10:40:00 | 4725.00 | 2026-04-30 11:10:00 | 4743.68 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-30 10:40:00 | 4725.00 | 2026-04-30 11:15:00 | 4725.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 10:30:00 | 4755.70 | 2026-05-04 12:05:00 | 4732.30 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-05-04 10:30:00 | 4755.70 | 2026-05-04 15:20:00 | 4709.00 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2026-05-05 10:00:00 | 4680.00 | 2026-05-05 10:25:00 | 4658.03 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-05-05 10:00:00 | 4680.00 | 2026-05-05 15:20:00 | 4640.20 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2026-05-06 11:15:00 | 4640.40 | 2026-05-06 12:30:00 | 4627.35 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-05-06 11:15:00 | 4640.40 | 2026-05-06 13:05:00 | 4640.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 11:00:00 | 4650.10 | 2026-05-07 11:30:00 | 4658.87 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-05-08 10:30:00 | 4633.00 | 2026-05-08 10:55:00 | 4615.25 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-05-08 10:30:00 | 4633.00 | 2026-05-08 15:20:00 | 4606.00 | TARGET_HIT | 0.50 | 0.58% |
