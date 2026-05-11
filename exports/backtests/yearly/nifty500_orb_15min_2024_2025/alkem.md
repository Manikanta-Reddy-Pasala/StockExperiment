# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 5560.00
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
| ENTRY1 | 80 |
| ENTRY2 | 0 |
| PARTIAL | 37 |
| TARGET_HIT | 19 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 117 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 61
- **Target hits / Stop hits / Partials:** 19 / 61 / 37
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 18.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 27 | 43.5% | 9 | 35 | 18 | 0.09% | 5.9% |
| BUY @ 2nd Alert (retest1) | 62 | 27 | 43.5% | 9 | 35 | 18 | 0.09% | 5.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 55 | 29 | 52.7% | 10 | 26 | 19 | 0.23% | 12.4% |
| SELL @ 2nd Alert (retest1) | 55 | 29 | 52.7% | 10 | 26 | 19 | 0.23% | 12.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 117 | 56 | 47.9% | 19 | 61 | 37 | 0.16% | 18.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:35:00 | 5301.00 | 5231.65 | 0.00 | ORB-long ORB[5191.15,5259.85] vol=5.2x ATR=26.04 |
| Stop hit — per-position SL triggered | 2024-05-16 11:15:00 | 5274.96 | 5248.77 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:10:00 | 5359.90 | 5326.12 | 0.00 | ORB-long ORB[5270.20,5350.00] vol=2.7x ATR=18.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 11:25:00 | 5387.45 | 5335.29 | 0.00 | T1 1.5R @ 5387.45 |
| Target hit | 2024-05-17 12:40:00 | 5373.90 | 5375.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2024-06-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:55:00 | 4922.50 | 4891.49 | 0.00 | ORB-long ORB[4827.20,4887.95] vol=1.5x ATR=19.01 |
| Stop hit — per-position SL triggered | 2024-06-07 11:15:00 | 4903.49 | 4895.84 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:55:00 | 5064.20 | 5030.38 | 0.00 | ORB-long ORB[5001.55,5045.95] vol=3.5x ATR=13.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:10:00 | 5084.61 | 5038.89 | 0.00 | T1 1.5R @ 5084.61 |
| Stop hit — per-position SL triggered | 2024-06-11 11:20:00 | 5064.20 | 5041.03 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:05:00 | 5105.90 | 5091.59 | 0.00 | ORB-long ORB[5054.05,5102.95] vol=11.6x ATR=12.99 |
| Stop hit — per-position SL triggered | 2024-06-12 11:10:00 | 5092.91 | 5091.36 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 10:55:00 | 5059.00 | 5070.96 | 0.00 | ORB-short ORB[5065.00,5105.00] vol=1.7x ATR=10.07 |
| Stop hit — per-position SL triggered | 2024-06-13 11:00:00 | 5069.07 | 5070.81 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 09:40:00 | 5153.95 | 5165.63 | 0.00 | ORB-short ORB[5155.55,5180.90] vol=1.5x ATR=12.51 |
| Stop hit — per-position SL triggered | 2024-06-21 10:20:00 | 5166.46 | 5161.38 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:35:00 | 4975.00 | 4991.97 | 0.00 | ORB-short ORB[4995.10,5034.30] vol=1.5x ATR=15.24 |
| Stop hit — per-position SL triggered | 2024-06-27 11:00:00 | 4990.24 | 4987.71 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:35:00 | 4928.30 | 4934.70 | 0.00 | ORB-short ORB[4952.00,4979.90] vol=1.6x ATR=9.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:35:00 | 4914.19 | 4924.99 | 0.00 | T1 1.5R @ 4914.19 |
| Target hit | 2024-07-02 15:05:00 | 4912.25 | 4912.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 11:15:00 | 4991.15 | 4967.24 | 0.00 | ORB-long ORB[4926.05,4960.05] vol=5.2x ATR=10.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:20:00 | 5006.80 | 4975.85 | 0.00 | T1 1.5R @ 5006.80 |
| Target hit | 2024-07-03 13:00:00 | 4994.95 | 4997.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-07-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 11:10:00 | 5000.00 | 4981.47 | 0.00 | ORB-long ORB[4955.25,4986.50] vol=7.4x ATR=11.40 |
| Stop hit — per-position SL triggered | 2024-07-04 11:50:00 | 4988.60 | 4988.26 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:35:00 | 5047.95 | 5038.59 | 0.00 | ORB-long ORB[4990.60,5039.45] vol=2.6x ATR=15.56 |
| Stop hit — per-position SL triggered | 2024-07-05 09:45:00 | 5032.39 | 5038.58 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:30:00 | 5122.35 | 5100.22 | 0.00 | ORB-long ORB[5075.40,5120.95] vol=2.2x ATR=13.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 10:40:00 | 5143.03 | 5110.81 | 0.00 | T1 1.5R @ 5143.03 |
| Target hit | 2024-07-08 15:20:00 | 5220.25 | 5167.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-07-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 11:00:00 | 5321.10 | 5278.97 | 0.00 | ORB-long ORB[5221.85,5283.00] vol=2.8x ATR=18.69 |
| Stop hit — per-position SL triggered | 2024-07-09 11:10:00 | 5302.41 | 5283.17 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:25:00 | 5213.10 | 5239.69 | 0.00 | ORB-short ORB[5293.30,5338.00] vol=4.5x ATR=17.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:40:00 | 5186.37 | 5233.74 | 0.00 | T1 1.5R @ 5186.37 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 5213.10 | 5232.48 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:20:00 | 5206.60 | 5216.64 | 0.00 | ORB-short ORB[5208.85,5268.00] vol=9.1x ATR=15.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:25:00 | 5182.81 | 5216.19 | 0.00 | T1 1.5R @ 5182.81 |
| Stop hit — per-position SL triggered | 2024-07-12 10:30:00 | 5206.60 | 5216.08 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 5177.40 | 5201.63 | 0.00 | ORB-short ORB[5187.95,5234.90] vol=1.6x ATR=15.40 |
| Stop hit — per-position SL triggered | 2024-07-23 11:35:00 | 5192.80 | 5199.06 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:15:00 | 5245.35 | 5210.03 | 0.00 | ORB-long ORB[5160.65,5208.00] vol=1.7x ATR=17.06 |
| Stop hit — per-position SL triggered | 2024-07-24 10:25:00 | 5228.29 | 5218.56 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:00:00 | 5297.55 | 5249.03 | 0.00 | ORB-long ORB[5189.25,5239.40] vol=2.9x ATR=18.72 |
| Stop hit — per-position SL triggered | 2024-07-26 10:05:00 | 5278.83 | 5251.41 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 10:50:00 | 5279.15 | 5294.95 | 0.00 | ORB-short ORB[5286.25,5347.95] vol=2.6x ATR=12.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 11:45:00 | 5260.38 | 5287.00 | 0.00 | T1 1.5R @ 5260.38 |
| Target hit | 2024-07-29 15:20:00 | 5208.25 | 5237.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-08-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:35:00 | 5314.35 | 5348.75 | 0.00 | ORB-short ORB[5332.55,5405.85] vol=2.6x ATR=18.77 |
| Stop hit — per-position SL triggered | 2024-08-01 10:45:00 | 5333.12 | 5346.10 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 09:50:00 | 5261.95 | 5231.47 | 0.00 | ORB-long ORB[5164.75,5238.15] vol=2.1x ATR=21.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 10:00:00 | 5294.75 | 5244.19 | 0.00 | T1 1.5R @ 5294.75 |
| Stop hit — per-position SL triggered | 2024-08-05 10:25:00 | 5261.95 | 5265.23 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:10:00 | 5433.00 | 5373.38 | 0.00 | ORB-long ORB[5325.70,5400.00] vol=2.5x ATR=23.28 |
| Stop hit — per-position SL triggered | 2024-08-06 10:30:00 | 5409.72 | 5383.63 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:35:00 | 5555.55 | 5532.09 | 0.00 | ORB-long ORB[5493.40,5550.00] vol=2.5x ATR=22.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 09:45:00 | 5588.63 | 5548.86 | 0.00 | T1 1.5R @ 5588.63 |
| Stop hit — per-position SL triggered | 2024-08-08 09:50:00 | 5555.55 | 5550.00 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 10:05:00 | 5764.45 | 5720.61 | 0.00 | ORB-long ORB[5672.00,5734.05] vol=2.6x ATR=20.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 10:40:00 | 5794.97 | 5745.43 | 0.00 | T1 1.5R @ 5794.97 |
| Stop hit — per-position SL triggered | 2024-08-13 10:45:00 | 5764.45 | 5745.93 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:25:00 | 5776.40 | 5740.09 | 0.00 | ORB-long ORB[5677.60,5750.00] vol=2.0x ATR=21.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 10:40:00 | 5807.90 | 5757.53 | 0.00 | T1 1.5R @ 5807.90 |
| Stop hit — per-position SL triggered | 2024-08-14 11:00:00 | 5776.40 | 5765.53 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:15:00 | 5790.35 | 5732.58 | 0.00 | ORB-long ORB[5713.50,5754.20] vol=2.4x ATR=16.38 |
| Stop hit — per-position SL triggered | 2024-08-20 10:20:00 | 5773.97 | 5733.87 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:55:00 | 5771.60 | 5758.56 | 0.00 | ORB-long ORB[5723.80,5767.95] vol=1.6x ATR=16.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:30:00 | 5795.63 | 5767.31 | 0.00 | T1 1.5R @ 5795.63 |
| Target hit | 2024-08-21 11:10:00 | 5777.00 | 5782.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2024-08-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:50:00 | 5940.00 | 5914.96 | 0.00 | ORB-long ORB[5862.05,5915.95] vol=7.9x ATR=19.91 |
| Stop hit — per-position SL triggered | 2024-08-28 10:00:00 | 5920.09 | 5916.18 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 6063.45 | 6038.17 | 0.00 | ORB-long ORB[5982.50,6050.00] vol=3.2x ATR=20.34 |
| Stop hit — per-position SL triggered | 2024-08-29 09:35:00 | 6043.11 | 6039.37 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:35:00 | 6065.50 | 6038.20 | 0.00 | ORB-long ORB[5998.50,6049.90] vol=1.7x ATR=18.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 09:40:00 | 6093.23 | 6055.23 | 0.00 | T1 1.5R @ 6093.23 |
| Target hit | 2024-08-30 15:20:00 | 6164.40 | 6141.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2024-09-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 11:00:00 | 6118.70 | 6133.69 | 0.00 | ORB-short ORB[6141.00,6190.00] vol=1.6x ATR=15.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:35:00 | 6096.09 | 6130.98 | 0.00 | T1 1.5R @ 6096.09 |
| Stop hit — per-position SL triggered | 2024-09-02 11:45:00 | 6118.70 | 6130.32 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 6133.00 | 6104.95 | 0.00 | ORB-long ORB[6040.05,6120.00] vol=2.4x ATR=17.46 |
| Stop hit — per-position SL triggered | 2024-09-03 10:05:00 | 6115.54 | 6114.68 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:40:00 | 6151.40 | 6126.65 | 0.00 | ORB-long ORB[6059.00,6125.00] vol=1.5x ATR=13.68 |
| Stop hit — per-position SL triggered | 2024-09-04 10:50:00 | 6137.72 | 6127.65 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:45:00 | 6267.30 | 6240.04 | 0.00 | ORB-long ORB[6191.95,6248.00] vol=2.6x ATR=14.69 |
| Stop hit — per-position SL triggered | 2024-09-05 09:55:00 | 6252.61 | 6246.13 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:00:00 | 6362.80 | 6331.31 | 0.00 | ORB-long ORB[6268.05,6326.85] vol=5.0x ATR=17.07 |
| Stop hit — per-position SL triggered | 2024-09-10 10:10:00 | 6345.73 | 6339.33 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:10:00 | 6359.35 | 6338.50 | 0.00 | ORB-long ORB[6281.45,6336.20] vol=4.1x ATR=18.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:35:00 | 6387.48 | 6366.78 | 0.00 | T1 1.5R @ 6387.48 |
| Target hit | 2024-09-12 10:50:00 | 6368.00 | 6381.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 6313.90 | 6328.58 | 0.00 | ORB-short ORB[6322.85,6357.95] vol=2.9x ATR=12.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:35:00 | 6295.82 | 6317.98 | 0.00 | T1 1.5R @ 6295.82 |
| Target hit | 2024-09-17 10:20:00 | 6300.65 | 6288.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — SELL (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 6036.05 | 6088.40 | 0.00 | ORB-short ORB[6096.55,6135.00] vol=1.6x ATR=16.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:10:00 | 6011.99 | 6083.44 | 0.00 | T1 1.5R @ 6011.99 |
| Target hit | 2024-09-23 15:20:00 | 5984.35 | 6023.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2024-09-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 10:55:00 | 6100.00 | 6070.88 | 0.00 | ORB-long ORB[6024.05,6075.00] vol=2.6x ATR=14.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 11:00:00 | 6122.31 | 6079.65 | 0.00 | T1 1.5R @ 6122.31 |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 6100.00 | 6082.86 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-09-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 10:30:00 | 6149.45 | 6169.35 | 0.00 | ORB-short ORB[6165.00,6213.00] vol=1.7x ATR=18.99 |
| Stop hit — per-position SL triggered | 2024-09-27 10:50:00 | 6168.44 | 6162.57 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:55:00 | 6085.20 | 6144.54 | 0.00 | ORB-short ORB[6180.40,6221.65] vol=1.9x ATR=16.69 |
| Stop hit — per-position SL triggered | 2024-09-30 11:05:00 | 6101.89 | 6140.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 10:45:00 | 6160.00 | 6129.59 | 0.00 | ORB-long ORB[6071.00,6144.00] vol=11.7x ATR=21.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:05:00 | 6191.87 | 6134.24 | 0.00 | T1 1.5R @ 6191.87 |
| Stop hit — per-position SL triggered | 2024-10-03 11:10:00 | 6160.00 | 6135.01 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:50:00 | 6198.50 | 6168.71 | 0.00 | ORB-long ORB[6104.30,6146.15] vol=3.1x ATR=19.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 11:00:00 | 6227.34 | 6171.97 | 0.00 | T1 1.5R @ 6227.34 |
| Stop hit — per-position SL triggered | 2024-10-08 11:50:00 | 6198.50 | 6189.64 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:10:00 | 6220.05 | 6267.74 | 0.00 | ORB-short ORB[6274.05,6340.55] vol=1.7x ATR=14.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:30:00 | 6198.45 | 6258.50 | 0.00 | T1 1.5R @ 6198.45 |
| Stop hit — per-position SL triggered | 2024-10-10 11:35:00 | 6220.05 | 6255.02 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:10:00 | 6134.60 | 6186.18 | 0.00 | ORB-short ORB[6214.80,6299.95] vol=2.3x ATR=18.14 |
| Stop hit — per-position SL triggered | 2024-10-16 10:20:00 | 6152.74 | 6178.44 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 6105.60 | 6125.28 | 0.00 | ORB-short ORB[6107.05,6196.00] vol=2.6x ATR=20.68 |
| Stop hit — per-position SL triggered | 2024-10-21 09:35:00 | 6126.28 | 6124.33 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:15:00 | 6138.90 | 6165.61 | 0.00 | ORB-short ORB[6155.30,6220.75] vol=6.3x ATR=21.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:40:00 | 6107.35 | 6155.58 | 0.00 | T1 1.5R @ 6107.35 |
| Stop hit — per-position SL triggered | 2024-10-22 10:50:00 | 6138.90 | 6154.67 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-10-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 10:45:00 | 6032.50 | 5976.88 | 0.00 | ORB-long ORB[5938.05,6007.40] vol=2.5x ATR=21.85 |
| Stop hit — per-position SL triggered | 2024-10-28 10:55:00 | 6010.65 | 5980.24 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-10-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 11:05:00 | 5875.00 | 5923.40 | 0.00 | ORB-short ORB[6006.70,6060.20] vol=2.2x ATR=20.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 11:25:00 | 5844.18 | 5915.47 | 0.00 | T1 1.5R @ 5844.18 |
| Stop hit — per-position SL triggered | 2024-10-29 13:20:00 | 5875.00 | 5883.44 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-30 10:35:00 | 5896.40 | 5919.51 | 0.00 | ORB-short ORB[5896.80,5975.40] vol=2.1x ATR=19.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 10:50:00 | 5867.56 | 5907.88 | 0.00 | T1 1.5R @ 5867.56 |
| Target hit | 2024-10-30 15:20:00 | 5801.50 | 5844.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 5852.15 | 5900.79 | 0.00 | ORB-short ORB[5871.00,5958.90] vol=1.6x ATR=22.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 09:40:00 | 5817.88 | 5880.96 | 0.00 | T1 1.5R @ 5817.88 |
| Target hit | 2024-11-07 15:20:00 | 5726.50 | 5763.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2024-11-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 10:40:00 | 5397.10 | 5434.70 | 0.00 | ORB-short ORB[5440.90,5517.45] vol=1.6x ATR=24.98 |
| Stop hit — per-position SL triggered | 2024-11-13 10:50:00 | 5422.08 | 5432.62 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:30:00 | 5725.05 | 5719.16 | 0.00 | ORB-long ORB[5650.00,5714.70] vol=1.7x ATR=17.36 |
| Stop hit — per-position SL triggered | 2024-12-02 10:35:00 | 5707.69 | 5719.10 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:35:00 | 5608.65 | 5621.95 | 0.00 | ORB-short ORB[5619.05,5649.85] vol=1.5x ATR=13.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:50:00 | 5588.01 | 5609.85 | 0.00 | T1 1.5R @ 5588.01 |
| Target hit | 2024-12-06 15:20:00 | 5508.85 | 5537.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2024-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:55:00 | 5498.35 | 5489.61 | 0.00 | ORB-long ORB[5425.05,5494.00] vol=4.1x ATR=11.37 |
| Stop hit — per-position SL triggered | 2024-12-11 12:15:00 | 5486.98 | 5491.55 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:45:00 | 5349.55 | 5388.33 | 0.00 | ORB-short ORB[5412.35,5447.00] vol=2.6x ATR=13.48 |
| Stop hit — per-position SL triggered | 2024-12-13 11:00:00 | 5363.03 | 5384.02 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-12-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:35:00 | 5421.00 | 5395.86 | 0.00 | ORB-long ORB[5351.65,5399.00] vol=1.8x ATR=12.81 |
| Stop hit — per-position SL triggered | 2024-12-16 11:25:00 | 5408.19 | 5402.97 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 5401.45 | 5422.24 | 0.00 | ORB-short ORB[5416.55,5452.25] vol=2.2x ATR=12.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:00:00 | 5383.39 | 5411.97 | 0.00 | T1 1.5R @ 5383.39 |
| Target hit | 2024-12-17 14:35:00 | 5389.15 | 5381.45 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:15:00 | 5480.65 | 5424.80 | 0.00 | ORB-long ORB[5390.00,5465.15] vol=2.4x ATR=15.25 |
| Stop hit — per-position SL triggered | 2024-12-23 11:55:00 | 5465.40 | 5443.32 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-12-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:55:00 | 5376.55 | 5391.36 | 0.00 | ORB-short ORB[5379.85,5455.90] vol=3.0x ATR=16.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:35:00 | 5352.31 | 5379.72 | 0.00 | T1 1.5R @ 5352.31 |
| Stop hit — per-position SL triggered | 2024-12-26 13:40:00 | 5376.55 | 5370.93 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-12-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 11:05:00 | 5614.70 | 5598.89 | 0.00 | ORB-long ORB[5550.00,5609.00] vol=2.3x ATR=15.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 11:25:00 | 5637.67 | 5606.33 | 0.00 | T1 1.5R @ 5637.67 |
| Target hit | 2024-12-31 14:20:00 | 5634.95 | 5635.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — BUY (started 2025-01-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:45:00 | 5565.30 | 5552.33 | 0.00 | ORB-long ORB[5517.50,5555.95] vol=1.5x ATR=15.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 10:15:00 | 5588.74 | 5562.12 | 0.00 | T1 1.5R @ 5588.74 |
| Target hit | 2025-01-07 12:15:00 | 5595.30 | 5609.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — SELL (started 2025-01-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 10:45:00 | 5520.05 | 5527.35 | 0.00 | ORB-short ORB[5532.25,5578.70] vol=1.6x ATR=16.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:00:00 | 5494.66 | 5521.21 | 0.00 | T1 1.5R @ 5494.66 |
| Target hit | 2025-01-08 12:25:00 | 5512.80 | 5512.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — SELL (started 2025-01-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:45:00 | 5227.80 | 5253.95 | 0.00 | ORB-short ORB[5235.35,5299.00] vol=2.4x ATR=14.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 11:05:00 | 5205.76 | 5249.43 | 0.00 | T1 1.5R @ 5205.76 |
| Stop hit — per-position SL triggered | 2025-01-21 11:45:00 | 5227.80 | 5242.40 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 10:00:00 | 5151.50 | 5182.23 | 0.00 | ORB-short ORB[5177.40,5236.15] vol=2.3x ATR=17.24 |
| Stop hit — per-position SL triggered | 2025-01-22 10:30:00 | 5168.74 | 5175.14 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:50:00 | 4881.05 | 4891.77 | 0.00 | ORB-short ORB[4902.40,4964.85] vol=2.9x ATR=16.72 |
| Stop hit — per-position SL triggered | 2025-01-28 11:00:00 | 4897.77 | 4889.77 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:55:00 | 5075.00 | 5054.54 | 0.00 | ORB-long ORB[4999.05,5031.95] vol=2.2x ATR=15.02 |
| Stop hit — per-position SL triggered | 2025-01-30 10:20:00 | 5059.98 | 5057.81 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 11:15:00 | 4709.90 | 4732.17 | 0.00 | ORB-short ORB[4721.25,4764.05] vol=1.8x ATR=14.31 |
| Stop hit — per-position SL triggered | 2025-02-13 12:30:00 | 4724.21 | 4728.58 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-02-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:25:00 | 4647.90 | 4664.51 | 0.00 | ORB-short ORB[4679.00,4745.00] vol=2.4x ATR=13.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 10:35:00 | 4627.25 | 4659.93 | 0.00 | T1 1.5R @ 4627.25 |
| Target hit | 2025-02-21 15:20:00 | 4601.50 | 4616.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2025-03-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 10:35:00 | 4750.00 | 4726.86 | 0.00 | ORB-long ORB[4679.55,4717.25] vol=1.7x ATR=16.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 10:40:00 | 4774.83 | 4730.44 | 0.00 | T1 1.5R @ 4774.83 |
| Stop hit — per-position SL triggered | 2025-03-06 11:00:00 | 4750.00 | 4736.25 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 11:15:00 | 5030.00 | 5010.34 | 0.00 | ORB-long ORB[4985.00,5029.45] vol=1.9x ATR=16.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 11:35:00 | 5054.40 | 5014.57 | 0.00 | T1 1.5R @ 5054.40 |
| Target hit | 2025-03-24 13:55:00 | 5047.20 | 5048.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — SELL (started 2025-03-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:45:00 | 4994.50 | 5011.13 | 0.00 | ORB-short ORB[5029.00,5080.00] vol=2.2x ATR=15.07 |
| Stop hit — per-position SL triggered | 2025-03-26 11:00:00 | 5009.57 | 5010.80 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-03-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 10:30:00 | 4922.00 | 4944.72 | 0.00 | ORB-short ORB[4942.15,5003.00] vol=6.8x ATR=15.15 |
| Stop hit — per-position SL triggered | 2025-03-27 11:40:00 | 4937.15 | 4939.80 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-04-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-11 09:50:00 | 4856.10 | 4878.43 | 0.00 | ORB-short ORB[4870.05,4909.35] vol=1.9x ATR=16.70 |
| Stop hit — per-position SL triggered | 2025-04-11 10:20:00 | 4872.80 | 4874.78 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-04-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:20:00 | 4975.00 | 4960.48 | 0.00 | ORB-long ORB[4921.00,4960.00] vol=5.8x ATR=12.95 |
| Stop hit — per-position SL triggered | 2025-04-21 10:30:00 | 4962.05 | 4962.29 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:40:00 | 5052.00 | 5032.15 | 0.00 | ORB-long ORB[5000.00,5038.60] vol=1.5x ATR=14.79 |
| Stop hit — per-position SL triggered | 2025-04-22 09:50:00 | 5037.21 | 5033.95 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 11:15:00 | 5067.30 | 5055.91 | 0.00 | ORB-long ORB[4995.00,5062.00] vol=1.5x ATR=13.28 |
| Stop hit — per-position SL triggered | 2025-04-28 11:50:00 | 5054.02 | 5058.03 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:30:00 | 5125.70 | 5099.37 | 0.00 | ORB-long ORB[5041.00,5107.50] vol=3.0x ATR=16.23 |
| Stop hit — per-position SL triggered | 2025-04-30 09:40:00 | 5109.47 | 5103.65 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:45:00 | 4965.00 | 4984.21 | 0.00 | ORB-short ORB[4967.00,5033.00] vol=3.6x ATR=14.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:00:00 | 4942.74 | 4963.82 | 0.00 | T1 1.5R @ 4942.74 |
| Stop hit — per-position SL triggered | 2025-05-08 10:05:00 | 4965.00 | 4964.70 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 10:35:00 | 5301.00 | 2024-05-16 11:15:00 | 5274.96 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-05-17 11:10:00 | 5359.90 | 2024-05-17 11:25:00 | 5387.45 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-05-17 11:10:00 | 5359.90 | 2024-05-17 12:40:00 | 5373.90 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-06-07 10:55:00 | 4922.50 | 2024-06-07 11:15:00 | 4903.49 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-06-11 10:55:00 | 5064.20 | 2024-06-11 11:10:00 | 5084.61 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-11 10:55:00 | 5064.20 | 2024-06-11 11:20:00 | 5064.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 11:05:00 | 5105.90 | 2024-06-12 11:10:00 | 5092.91 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-06-13 10:55:00 | 5059.00 | 2024-06-13 11:00:00 | 5069.07 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-06-21 09:40:00 | 5153.95 | 2024-06-21 10:20:00 | 5166.46 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-06-27 10:35:00 | 4975.00 | 2024-06-27 11:00:00 | 4990.24 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-02 10:35:00 | 4928.30 | 2024-07-02 12:35:00 | 4914.19 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-07-02 10:35:00 | 4928.30 | 2024-07-02 15:05:00 | 4912.25 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2024-07-03 11:15:00 | 4991.15 | 2024-07-03 11:20:00 | 5006.80 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-07-03 11:15:00 | 4991.15 | 2024-07-03 13:00:00 | 4994.95 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2024-07-04 11:10:00 | 5000.00 | 2024-07-04 11:50:00 | 4988.60 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-05 09:35:00 | 5047.95 | 2024-07-05 09:45:00 | 5032.39 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-08 10:30:00 | 5122.35 | 2024-07-08 10:40:00 | 5143.03 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-08 10:30:00 | 5122.35 | 2024-07-08 15:20:00 | 5220.25 | TARGET_HIT | 0.50 | 1.91% |
| BUY | retest1 | 2024-07-09 11:00:00 | 5321.10 | 2024-07-09 11:10:00 | 5302.41 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-10 10:25:00 | 5213.10 | 2024-07-10 10:40:00 | 5186.37 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-10 10:25:00 | 5213.10 | 2024-07-10 10:55:00 | 5213.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 10:20:00 | 5206.60 | 2024-07-12 10:25:00 | 5182.81 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-07-12 10:20:00 | 5206.60 | 2024-07-12 10:30:00 | 5206.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 11:15:00 | 5177.40 | 2024-07-23 11:35:00 | 5192.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-24 10:15:00 | 5245.35 | 2024-07-24 10:25:00 | 5228.29 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-26 10:00:00 | 5297.55 | 2024-07-26 10:05:00 | 5278.83 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-29 10:50:00 | 5279.15 | 2024-07-29 11:45:00 | 5260.38 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-07-29 10:50:00 | 5279.15 | 2024-07-29 15:20:00 | 5208.25 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2024-08-01 10:35:00 | 5314.35 | 2024-08-01 10:45:00 | 5333.12 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-05 09:50:00 | 5261.95 | 2024-08-05 10:00:00 | 5294.75 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-08-05 09:50:00 | 5261.95 | 2024-08-05 10:25:00 | 5261.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-06 10:10:00 | 5433.00 | 2024-08-06 10:30:00 | 5409.72 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-08-08 09:35:00 | 5555.55 | 2024-08-08 09:45:00 | 5588.63 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-08-08 09:35:00 | 5555.55 | 2024-08-08 09:50:00 | 5555.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-13 10:05:00 | 5764.45 | 2024-08-13 10:40:00 | 5794.97 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-08-13 10:05:00 | 5764.45 | 2024-08-13 10:45:00 | 5764.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-14 10:25:00 | 5776.40 | 2024-08-14 10:40:00 | 5807.90 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-08-14 10:25:00 | 5776.40 | 2024-08-14 11:00:00 | 5776.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 10:15:00 | 5790.35 | 2024-08-20 10:20:00 | 5773.97 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-21 09:55:00 | 5771.60 | 2024-08-21 10:30:00 | 5795.63 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-08-21 09:55:00 | 5771.60 | 2024-08-21 11:10:00 | 5777.00 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-08-28 09:50:00 | 5940.00 | 2024-08-28 10:00:00 | 5920.09 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-29 09:30:00 | 6063.45 | 2024-08-29 09:35:00 | 6043.11 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-30 09:35:00 | 6065.50 | 2024-08-30 09:40:00 | 6093.23 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-30 09:35:00 | 6065.50 | 2024-08-30 15:20:00 | 6164.40 | TARGET_HIT | 0.50 | 1.63% |
| SELL | retest1 | 2024-09-02 11:00:00 | 6118.70 | 2024-09-02 11:35:00 | 6096.09 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-02 11:00:00 | 6118.70 | 2024-09-02 11:45:00 | 6118.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:35:00 | 6133.00 | 2024-09-03 10:05:00 | 6115.54 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-04 10:40:00 | 6151.40 | 2024-09-04 10:50:00 | 6137.72 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-05 09:45:00 | 6267.30 | 2024-09-05 09:55:00 | 6252.61 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-10 10:00:00 | 6362.80 | 2024-09-10 10:10:00 | 6345.73 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-12 10:10:00 | 6359.35 | 2024-09-12 10:35:00 | 6387.48 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-12 10:10:00 | 6359.35 | 2024-09-12 10:50:00 | 6368.00 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2024-09-17 09:30:00 | 6313.90 | 2024-09-17 09:35:00 | 6295.82 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-09-17 09:30:00 | 6313.90 | 2024-09-17 10:20:00 | 6300.65 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2024-09-23 11:00:00 | 6036.05 | 2024-09-23 11:10:00 | 6011.99 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-09-23 11:00:00 | 6036.05 | 2024-09-23 15:20:00 | 5984.35 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2024-09-25 10:55:00 | 6100.00 | 2024-09-25 11:00:00 | 6122.31 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-25 10:55:00 | 6100.00 | 2024-09-25 11:15:00 | 6100.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-27 10:30:00 | 6149.45 | 2024-09-27 10:50:00 | 6168.44 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-30 10:55:00 | 6085.20 | 2024-09-30 11:05:00 | 6101.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-03 10:45:00 | 6160.00 | 2024-10-03 11:05:00 | 6191.87 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-10-03 10:45:00 | 6160.00 | 2024-10-03 11:10:00 | 6160.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-08 10:50:00 | 6198.50 | 2024-10-08 11:00:00 | 6227.34 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-10-08 10:50:00 | 6198.50 | 2024-10-08 11:50:00 | 6198.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-10 11:10:00 | 6220.05 | 2024-10-10 11:30:00 | 6198.45 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-10-10 11:10:00 | 6220.05 | 2024-10-10 11:35:00 | 6220.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 10:10:00 | 6134.60 | 2024-10-16 10:20:00 | 6152.74 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-21 09:30:00 | 6105.60 | 2024-10-21 09:35:00 | 6126.28 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-22 10:15:00 | 6138.90 | 2024-10-22 10:40:00 | 6107.35 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-22 10:15:00 | 6138.90 | 2024-10-22 10:50:00 | 6138.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-28 10:45:00 | 6032.50 | 2024-10-28 10:55:00 | 6010.65 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-29 11:05:00 | 5875.00 | 2024-10-29 11:25:00 | 5844.18 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-29 11:05:00 | 5875.00 | 2024-10-29 13:20:00 | 5875.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-30 10:35:00 | 5896.40 | 2024-10-30 10:50:00 | 5867.56 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-30 10:35:00 | 5896.40 | 2024-10-30 15:20:00 | 5801.50 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2024-11-07 09:35:00 | 5852.15 | 2024-11-07 09:40:00 | 5817.88 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-11-07 09:35:00 | 5852.15 | 2024-11-07 15:20:00 | 5726.50 | TARGET_HIT | 0.50 | 2.15% |
| SELL | retest1 | 2024-11-13 10:40:00 | 5397.10 | 2024-11-13 10:50:00 | 5422.08 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-12-02 10:30:00 | 5725.05 | 2024-12-02 10:35:00 | 5707.69 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-06 09:35:00 | 5608.65 | 2024-12-06 09:50:00 | 5588.01 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-06 09:35:00 | 5608.65 | 2024-12-06 15:20:00 | 5508.85 | TARGET_HIT | 0.50 | 1.78% |
| BUY | retest1 | 2024-12-11 10:55:00 | 5498.35 | 2024-12-11 12:15:00 | 5486.98 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-12-13 10:45:00 | 5349.55 | 2024-12-13 11:00:00 | 5363.03 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-16 10:35:00 | 5421.00 | 2024-12-16 11:25:00 | 5408.19 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-17 09:30:00 | 5401.45 | 2024-12-17 10:00:00 | 5383.39 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-12-17 09:30:00 | 5401.45 | 2024-12-17 14:35:00 | 5389.15 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-12-23 11:15:00 | 5480.65 | 2024-12-23 11:55:00 | 5465.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-26 09:55:00 | 5376.55 | 2024-12-26 11:35:00 | 5352.31 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-26 09:55:00 | 5376.55 | 2024-12-26 13:40:00 | 5376.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-31 11:05:00 | 5614.70 | 2024-12-31 11:25:00 | 5637.67 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-12-31 11:05:00 | 5614.70 | 2024-12-31 14:20:00 | 5634.95 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2025-01-07 09:45:00 | 5565.30 | 2025-01-07 10:15:00 | 5588.74 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-07 09:45:00 | 5565.30 | 2025-01-07 12:15:00 | 5595.30 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2025-01-08 10:45:00 | 5520.05 | 2025-01-08 11:00:00 | 5494.66 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-08 10:45:00 | 5520.05 | 2025-01-08 12:25:00 | 5512.80 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2025-01-21 10:45:00 | 5227.80 | 2025-01-21 11:05:00 | 5205.76 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-21 10:45:00 | 5227.80 | 2025-01-21 11:45:00 | 5227.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-22 10:00:00 | 5151.50 | 2025-01-22 10:30:00 | 5168.74 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-28 10:50:00 | 4881.05 | 2025-01-28 11:00:00 | 4897.77 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-30 09:55:00 | 5075.00 | 2025-01-30 10:20:00 | 5059.98 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-13 11:15:00 | 4709.90 | 2025-02-13 12:30:00 | 4724.21 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-21 10:25:00 | 4647.90 | 2025-02-21 10:35:00 | 4627.25 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-02-21 10:25:00 | 4647.90 | 2025-02-21 15:20:00 | 4601.50 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2025-03-06 10:35:00 | 4750.00 | 2025-03-06 10:40:00 | 4774.83 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-03-06 10:35:00 | 4750.00 | 2025-03-06 11:00:00 | 4750.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-24 11:15:00 | 5030.00 | 2025-03-24 11:35:00 | 5054.40 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-03-24 11:15:00 | 5030.00 | 2025-03-24 13:55:00 | 5047.20 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-03-26 10:45:00 | 4994.50 | 2025-03-26 11:00:00 | 5009.57 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-27 10:30:00 | 4922.00 | 2025-03-27 11:40:00 | 4937.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-11 09:50:00 | 4856.10 | 2025-04-11 10:20:00 | 4872.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-21 10:20:00 | 4975.00 | 2025-04-21 10:30:00 | 4962.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-04-22 09:40:00 | 5052.00 | 2025-04-22 09:50:00 | 5037.21 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-28 11:15:00 | 5067.30 | 2025-04-28 11:50:00 | 5054.02 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-04-30 09:30:00 | 5125.70 | 2025-04-30 09:40:00 | 5109.47 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-05-08 09:45:00 | 4965.00 | 2025-05-08 10:00:00 | 4942.74 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-05-08 09:45:00 | 4965.00 | 2025-05-08 10:05:00 | 4965.00 | STOP_HIT | 0.50 | 0.00% |
