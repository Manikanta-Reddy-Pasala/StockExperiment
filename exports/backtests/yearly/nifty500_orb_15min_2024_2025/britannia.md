# Britannia Industries Ltd. (BRITANNIA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-10-01 15:25:00 (25983 bars)
- **Last close:** 5973.00
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
| ENTRY1 | 84 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 7 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 114 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 77
- **Target hits / Stop hits / Partials:** 7 / 77 / 30
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 4.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 23 | 32.9% | 4 | 47 | 19 | 0.05% | 3.3% |
| BUY @ 2nd Alert (retest1) | 70 | 23 | 32.9% | 4 | 47 | 19 | 0.05% | 3.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 14 | 31.8% | 3 | 30 | 11 | 0.02% | 0.9% |
| SELL @ 2nd Alert (retest1) | 44 | 14 | 31.8% | 3 | 30 | 11 | 0.02% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 114 | 37 | 32.5% | 7 | 77 | 30 | 0.04% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-13 10:45:00 | 5118.95 | 5102.48 | 0.00 | ORB-long ORB[5067.00,5118.35] vol=2.4x ATR=29.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 11:30:00 | 5163.10 | 5111.41 | 0.00 | T1 1.5R @ 5163.10 |
| Stop hit — per-position SL triggered | 2024-05-13 14:40:00 | 5118.95 | 5123.22 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:45:00 | 5062.60 | 5098.27 | 0.00 | ORB-short ORB[5105.00,5130.55] vol=1.6x ATR=13.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:25:00 | 5042.09 | 5087.95 | 0.00 | T1 1.5R @ 5042.09 |
| Stop hit — per-position SL triggered | 2024-05-14 11:45:00 | 5062.60 | 5083.63 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 11:00:00 | 5090.00 | 5101.86 | 0.00 | ORB-short ORB[5109.40,5164.75] vol=1.6x ATR=15.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 11:25:00 | 5067.08 | 5099.03 | 0.00 | T1 1.5R @ 5067.08 |
| Stop hit — per-position SL triggered | 2024-05-15 12:25:00 | 5090.00 | 5093.82 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 5028.35 | 5055.75 | 0.00 | ORB-short ORB[5040.00,5089.15] vol=3.0x ATR=15.63 |
| Stop hit — per-position SL triggered | 2024-05-16 10:05:00 | 5043.98 | 5039.18 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:40:00 | 5118.85 | 5087.99 | 0.00 | ORB-long ORB[5045.05,5105.00] vol=1.9x ATR=14.08 |
| Stop hit — per-position SL triggered | 2024-05-21 11:05:00 | 5104.77 | 5094.14 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:50:00 | 5269.15 | 5232.95 | 0.00 | ORB-long ORB[5195.00,5225.95] vol=3.5x ATR=17.67 |
| Stop hit — per-position SL triggered | 2024-05-22 10:05:00 | 5251.48 | 5242.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-05-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:45:00 | 5280.15 | 5265.85 | 0.00 | ORB-long ORB[5231.00,5266.95] vol=3.1x ATR=9.81 |
| Stop hit — per-position SL triggered | 2024-05-24 11:05:00 | 5270.34 | 5267.48 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-05-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:40:00 | 5197.40 | 5222.44 | 0.00 | ORB-short ORB[5203.55,5265.00] vol=2.0x ATR=13.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:10:00 | 5176.83 | 5204.86 | 0.00 | T1 1.5R @ 5176.83 |
| Target hit | 2024-05-27 11:50:00 | 5178.75 | 5174.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2024-05-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 11:05:00 | 5252.00 | 5227.81 | 0.00 | ORB-long ORB[5180.55,5236.50] vol=1.8x ATR=10.38 |
| Stop hit — per-position SL triggered | 2024-05-28 11:25:00 | 5241.62 | 5229.84 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-05-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:50:00 | 5248.90 | 5228.43 | 0.00 | ORB-long ORB[5200.00,5238.10] vol=2.5x ATR=13.14 |
| Stop hit — per-position SL triggered | 2024-05-29 11:05:00 | 5235.76 | 5231.22 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-04 09:40:00 | 5230.00 | 5188.01 | 0.00 | ORB-long ORB[5140.10,5215.40] vol=3.9x ATR=20.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:45:00 | 5261.27 | 5204.14 | 0.00 | T1 1.5R @ 5261.27 |
| Stop hit — per-position SL triggered | 2024-06-04 10:00:00 | 5230.00 | 5218.31 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:35:00 | 5516.60 | 5467.65 | 0.00 | ORB-long ORB[5406.10,5485.10] vol=2.9x ATR=19.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 11:15:00 | 5545.19 | 5496.47 | 0.00 | T1 1.5R @ 5545.19 |
| Stop hit — per-position SL triggered | 2024-06-10 11:55:00 | 5516.60 | 5508.29 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 11:10:00 | 5447.30 | 5417.29 | 0.00 | ORB-long ORB[5388.80,5429.80] vol=2.4x ATR=14.07 |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 5433.23 | 5417.79 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:55:00 | 5445.65 | 5460.26 | 0.00 | ORB-short ORB[5450.10,5500.00] vol=1.7x ATR=10.11 |
| Stop hit — per-position SL triggered | 2024-07-02 11:05:00 | 5455.76 | 5459.50 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 11:10:00 | 5471.70 | 5461.70 | 0.00 | ORB-long ORB[5433.95,5470.00] vol=1.9x ATR=9.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 11:20:00 | 5485.41 | 5465.94 | 0.00 | T1 1.5R @ 5485.41 |
| Target hit | 2024-07-05 15:20:00 | 5551.90 | 5514.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2024-07-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:50:00 | 5595.15 | 5570.61 | 0.00 | ORB-long ORB[5534.05,5583.40] vol=2.7x ATR=17.29 |
| Stop hit — per-position SL triggered | 2024-07-08 10:05:00 | 5577.86 | 5578.25 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 09:40:00 | 5751.85 | 5716.94 | 0.00 | ORB-long ORB[5670.05,5737.45] vol=2.3x ATR=17.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:50:00 | 5777.65 | 5734.50 | 0.00 | T1 1.5R @ 5777.65 |
| Stop hit — per-position SL triggered | 2024-07-10 10:00:00 | 5751.85 | 5740.96 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:30:00 | 5838.50 | 5815.77 | 0.00 | ORB-long ORB[5792.10,5820.00] vol=1.7x ATR=11.59 |
| Stop hit — per-position SL triggered | 2024-07-15 10:25:00 | 5826.91 | 5827.90 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 11:15:00 | 5839.25 | 5814.12 | 0.00 | ORB-long ORB[5770.10,5838.75] vol=3.7x ATR=9.22 |
| Stop hit — per-position SL triggered | 2024-07-16 11:50:00 | 5830.03 | 5818.28 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 10:40:00 | 5916.50 | 5889.19 | 0.00 | ORB-long ORB[5830.00,5893.65] vol=1.7x ATR=14.13 |
| Stop hit — per-position SL triggered | 2024-07-18 11:00:00 | 5902.37 | 5894.15 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 10:25:00 | 5963.00 | 5928.22 | 0.00 | ORB-long ORB[5846.65,5903.80] vol=2.0x ATR=19.27 |
| Stop hit — per-position SL triggered | 2024-07-19 10:50:00 | 5943.73 | 5938.58 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 5919.00 | 5873.76 | 0.00 | ORB-long ORB[5865.10,5915.95] vol=2.2x ATR=13.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:25:00 | 5939.26 | 5879.39 | 0.00 | T1 1.5R @ 5939.26 |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 5919.00 | 5906.34 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 11:10:00 | 5816.55 | 5793.45 | 0.00 | ORB-long ORB[5760.20,5790.05] vol=2.0x ATR=12.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:50:00 | 5835.63 | 5796.39 | 0.00 | T1 1.5R @ 5835.63 |
| Stop hit — per-position SL triggered | 2024-08-01 12:30:00 | 5816.55 | 5801.20 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:55:00 | 5743.45 | 5759.61 | 0.00 | ORB-short ORB[5751.60,5785.55] vol=1.9x ATR=11.40 |
| Stop hit — per-position SL triggered | 2024-08-09 11:05:00 | 5754.85 | 5759.09 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 11:15:00 | 5701.90 | 5745.74 | 0.00 | ORB-short ORB[5722.05,5774.50] vol=1.9x ATR=12.91 |
| Stop hit — per-position SL triggered | 2024-08-12 11:30:00 | 5714.81 | 5744.79 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 11:05:00 | 5708.00 | 5688.69 | 0.00 | ORB-long ORB[5654.00,5687.90] vol=1.7x ATR=11.91 |
| Stop hit — per-position SL triggered | 2024-08-13 11:10:00 | 5696.09 | 5688.92 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:45:00 | 5685.10 | 5679.33 | 0.00 | ORB-long ORB[5662.50,5681.20] vol=2.1x ATR=13.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 10:00:00 | 5705.20 | 5687.42 | 0.00 | T1 1.5R @ 5705.20 |
| Target hit | 2024-08-16 10:15:00 | 5688.20 | 5690.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — SELL (started 2024-08-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 11:00:00 | 5815.00 | 5830.91 | 0.00 | ORB-short ORB[5827.50,5850.00] vol=2.6x ATR=9.35 |
| Stop hit — per-position SL triggered | 2024-08-23 11:10:00 | 5824.35 | 5830.37 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:30:00 | 5855.10 | 5823.11 | 0.00 | ORB-long ORB[5776.25,5809.45] vol=1.6x ATR=10.16 |
| Stop hit — per-position SL triggered | 2024-08-26 10:40:00 | 5844.94 | 5837.12 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:45:00 | 5701.45 | 5733.30 | 0.00 | ORB-short ORB[5736.10,5775.40] vol=1.6x ATR=10.68 |
| Stop hit — per-position SL triggered | 2024-08-28 11:20:00 | 5712.13 | 5725.66 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 11:15:00 | 5751.00 | 5733.35 | 0.00 | ORB-long ORB[5696.15,5727.00] vol=4.0x ATR=10.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 12:05:00 | 5766.02 | 5741.58 | 0.00 | T1 1.5R @ 5766.02 |
| Target hit | 2024-08-29 14:20:00 | 5788.00 | 5789.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2024-09-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 09:30:00 | 5875.00 | 5854.99 | 0.00 | ORB-long ORB[5834.00,5860.80] vol=1.6x ATR=12.56 |
| Stop hit — per-position SL triggered | 2024-09-06 09:35:00 | 5862.44 | 5857.02 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 09:40:00 | 5909.95 | 5884.52 | 0.00 | ORB-long ORB[5859.95,5893.50] vol=1.7x ATR=17.88 |
| Stop hit — per-position SL triggered | 2024-09-09 09:50:00 | 5892.07 | 5888.82 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:55:00 | 6019.95 | 5999.14 | 0.00 | ORB-long ORB[5947.25,6002.00] vol=1.7x ATR=13.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:55:00 | 6039.72 | 6010.97 | 0.00 | T1 1.5R @ 6039.72 |
| Stop hit — per-position SL triggered | 2024-09-11 12:05:00 | 6019.95 | 6015.10 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 10:05:00 | 6189.00 | 6166.13 | 0.00 | ORB-long ORB[6127.15,6169.15] vol=4.3x ATR=16.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:40:00 | 6214.02 | 6181.94 | 0.00 | T1 1.5R @ 6214.02 |
| Stop hit — per-position SL triggered | 2024-09-19 11:45:00 | 6189.00 | 6192.02 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:40:00 | 6148.40 | 6159.14 | 0.00 | ORB-short ORB[6150.35,6203.00] vol=1.7x ATR=12.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:05:00 | 6129.32 | 6155.99 | 0.00 | T1 1.5R @ 6129.32 |
| Stop hit — per-position SL triggered | 2024-09-25 12:35:00 | 6148.40 | 6140.17 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:55:00 | 6308.60 | 6255.33 | 0.00 | ORB-long ORB[6180.80,6249.30] vol=2.5x ATR=13.96 |
| Stop hit — per-position SL triggered | 2024-09-27 11:10:00 | 6294.64 | 6263.47 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 11:15:00 | 6349.85 | 6423.01 | 0.00 | ORB-short ORB[6407.90,6469.90] vol=1.6x ATR=16.14 |
| Stop hit — per-position SL triggered | 2024-10-03 11:20:00 | 6365.99 | 6420.80 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:10:00 | 6130.15 | 6155.14 | 0.00 | ORB-short ORB[6160.00,6235.00] vol=3.2x ATR=19.15 |
| Stop hit — per-position SL triggered | 2024-10-07 11:25:00 | 6149.30 | 6153.91 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 10:10:00 | 6117.50 | 6133.52 | 0.00 | ORB-short ORB[6131.35,6185.00] vol=2.1x ATR=23.29 |
| Stop hit — per-position SL triggered | 2024-10-09 11:20:00 | 6140.79 | 6128.86 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 6016.00 | 6088.03 | 0.00 | ORB-short ORB[6097.25,6159.85] vol=2.4x ATR=17.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:55:00 | 5989.50 | 6059.15 | 0.00 | T1 1.5R @ 5989.50 |
| Stop hit — per-position SL triggered | 2024-10-10 12:20:00 | 6016.00 | 6051.80 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:35:00 | 5910.00 | 5958.35 | 0.00 | ORB-short ORB[5930.15,6015.00] vol=1.7x ATR=15.10 |
| Stop hit — per-position SL triggered | 2024-10-14 10:40:00 | 5925.10 | 5956.06 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:15:00 | 6032.55 | 6044.45 | 0.00 | ORB-short ORB[6044.60,6095.90] vol=1.5x ATR=13.84 |
| Stop hit — per-position SL triggered | 2024-10-17 11:30:00 | 6046.39 | 6043.59 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:00:00 | 5801.20 | 5820.78 | 0.00 | ORB-short ORB[5850.00,5895.50] vol=1.9x ATR=13.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:10:00 | 5780.42 | 5816.88 | 0.00 | T1 1.5R @ 5780.42 |
| Stop hit — per-position SL triggered | 2024-10-21 13:20:00 | 5801.20 | 5806.28 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 5694.85 | 5666.72 | 0.00 | ORB-long ORB[5610.00,5683.80] vol=1.6x ATR=15.60 |
| Stop hit — per-position SL triggered | 2024-10-25 09:35:00 | 5679.25 | 5670.72 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-10-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 11:05:00 | 5757.10 | 5729.25 | 0.00 | ORB-long ORB[5669.40,5710.55] vol=1.6x ATR=13.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 11:30:00 | 5778.04 | 5736.03 | 0.00 | T1 1.5R @ 5778.04 |
| Stop hit — per-position SL triggered | 2024-10-28 11:40:00 | 5757.10 | 5738.58 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:15:00 | 5864.00 | 5808.81 | 0.00 | ORB-long ORB[5751.05,5812.60] vol=2.0x ATR=26.53 |
| Stop hit — per-position SL triggered | 2024-11-11 10:25:00 | 5837.47 | 5820.55 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 11:15:00 | 4991.85 | 4993.59 | 0.00 | ORB-short ORB[4995.10,5050.50] vol=1.5x ATR=14.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 12:20:00 | 4970.15 | 4991.88 | 0.00 | T1 1.5R @ 4970.15 |
| Target hit | 2024-11-14 15:20:00 | 4919.20 | 4958.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2024-11-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:50:00 | 4953.00 | 4965.93 | 0.00 | ORB-short ORB[4966.05,5025.00] vol=1.7x ATR=10.27 |
| Stop hit — per-position SL triggered | 2024-11-27 10:55:00 | 4963.27 | 4965.75 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 4948.35 | 4982.06 | 0.00 | ORB-short ORB[4957.45,5008.30] vol=1.9x ATR=14.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:55:00 | 4926.91 | 4968.06 | 0.00 | T1 1.5R @ 4926.91 |
| Target hit | 2024-11-28 15:20:00 | 4931.95 | 4941.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 4820.00 | 4828.83 | 0.00 | ORB-short ORB[4830.00,4864.25] vol=3.3x ATR=10.30 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 4830.30 | 4826.46 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:35:00 | 4866.65 | 4836.32 | 0.00 | ORB-long ORB[4793.25,4837.95] vol=2.6x ATR=9.28 |
| Stop hit — per-position SL triggered | 2024-12-11 09:40:00 | 4857.37 | 4841.29 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 4809.40 | 4829.74 | 0.00 | ORB-short ORB[4833.20,4869.00] vol=2.3x ATR=7.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 11:10:00 | 4798.13 | 4821.86 | 0.00 | T1 1.5R @ 4798.13 |
| Stop hit — per-position SL triggered | 2024-12-16 11:20:00 | 4809.40 | 4818.08 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:20:00 | 4782.30 | 4805.16 | 0.00 | ORB-short ORB[4806.00,4839.90] vol=2.1x ATR=9.91 |
| Stop hit — per-position SL triggered | 2024-12-17 10:40:00 | 4792.21 | 4800.89 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 11:15:00 | 4750.95 | 4733.83 | 0.00 | ORB-long ORB[4690.05,4750.00] vol=2.1x ATR=10.54 |
| Stop hit — per-position SL triggered | 2024-12-24 14:10:00 | 4740.41 | 4745.12 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:50:00 | 4722.55 | 4736.62 | 0.00 | ORB-short ORB[4736.00,4755.50] vol=1.8x ATR=10.46 |
| Stop hit — per-position SL triggered | 2024-12-26 09:55:00 | 4733.01 | 4735.95 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 11:15:00 | 4760.45 | 4761.03 | 0.00 | ORB-short ORB[4765.05,4795.00] vol=2.2x ATR=9.89 |
| Stop hit — per-position SL triggered | 2024-12-31 11:20:00 | 4770.34 | 4761.27 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 4812.00 | 4789.32 | 0.00 | ORB-long ORB[4752.20,4798.00] vol=1.8x ATR=11.23 |
| Stop hit — per-position SL triggered | 2025-01-01 11:20:00 | 4800.77 | 4793.09 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 4745.00 | 4765.30 | 0.00 | ORB-short ORB[4751.00,4806.95] vol=2.5x ATR=10.01 |
| Stop hit — per-position SL triggered | 2025-01-02 09:35:00 | 4755.01 | 4763.26 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:30:00 | 4831.10 | 4814.26 | 0.00 | ORB-long ORB[4785.75,4824.45] vol=2.2x ATR=9.51 |
| Stop hit — per-position SL triggered | 2025-01-03 10:45:00 | 4821.59 | 4816.87 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:40:00 | 4806.80 | 4840.10 | 0.00 | ORB-short ORB[4841.35,4873.50] vol=2.8x ATR=13.57 |
| Stop hit — per-position SL triggered | 2025-01-06 09:50:00 | 4820.37 | 4833.93 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:30:00 | 4863.10 | 4838.61 | 0.00 | ORB-long ORB[4804.95,4856.90] vol=1.5x ATR=15.71 |
| Stop hit — per-position SL triggered | 2025-01-07 12:05:00 | 4847.39 | 4854.18 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:55:00 | 4874.30 | 4861.04 | 0.00 | ORB-long ORB[4840.05,4868.65] vol=1.7x ATR=11.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:05:00 | 4892.10 | 4872.85 | 0.00 | T1 1.5R @ 4892.10 |
| Target hit | 2025-01-09 11:30:00 | 4941.95 | 4958.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — SELL (started 2025-01-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:45:00 | 4760.15 | 4824.45 | 0.00 | ORB-short ORB[4831.55,4900.00] vol=1.9x ATR=13.67 |
| Stop hit — per-position SL triggered | 2025-01-16 11:05:00 | 4773.82 | 4799.66 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 4877.10 | 4889.16 | 0.00 | ORB-short ORB[4881.35,4912.15] vol=3.5x ATR=11.82 |
| Stop hit — per-position SL triggered | 2025-01-21 10:45:00 | 4888.92 | 4883.75 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 10:10:00 | 5068.70 | 5045.94 | 0.00 | ORB-long ORB[5013.15,5057.35] vol=2.1x ATR=13.14 |
| Stop hit — per-position SL triggered | 2025-01-24 10:15:00 | 5055.56 | 5048.46 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:40:00 | 5127.10 | 5139.49 | 0.00 | ORB-short ORB[5148.15,5216.25] vol=3.0x ATR=15.46 |
| Stop hit — per-position SL triggered | 2025-01-28 11:00:00 | 5142.56 | 5137.29 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:20:00 | 5085.05 | 5074.87 | 0.00 | ORB-long ORB[5034.95,5072.20] vol=4.7x ATR=13.86 |
| Stop hit — per-position SL triggered | 2025-01-30 11:25:00 | 5071.19 | 5078.86 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-02-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 10:25:00 | 4937.35 | 4925.83 | 0.00 | ORB-long ORB[4873.75,4905.20] vol=5.7x ATR=14.82 |
| Stop hit — per-position SL triggered | 2025-02-14 10:50:00 | 4922.53 | 4926.58 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-02-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 10:45:00 | 4704.85 | 4735.34 | 0.00 | ORB-short ORB[4743.00,4786.45] vol=2.9x ATR=12.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 10:55:00 | 4686.17 | 4728.65 | 0.00 | T1 1.5R @ 4686.17 |
| Stop hit — per-position SL triggered | 2025-02-28 11:05:00 | 4704.85 | 4724.90 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-03 11:00:00 | 4639.00 | 4630.89 | 0.00 | ORB-long ORB[4570.80,4622.00] vol=2.2x ATR=13.30 |
| Stop hit — per-position SL triggered | 2025-03-03 12:30:00 | 4625.70 | 4632.73 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:50:00 | 4633.45 | 4599.31 | 0.00 | ORB-long ORB[4560.00,4607.95] vol=1.6x ATR=10.71 |
| Stop hit — per-position SL triggered | 2025-03-05 10:55:00 | 4622.74 | 4600.19 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:35:00 | 4785.05 | 4763.56 | 0.00 | ORB-long ORB[4725.30,4772.45] vol=1.8x ATR=13.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 09:55:00 | 4804.85 | 4774.17 | 0.00 | T1 1.5R @ 4804.85 |
| Stop hit — per-position SL triggered | 2025-03-10 10:30:00 | 4785.05 | 4784.71 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-03-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:10:00 | 4829.00 | 4792.69 | 0.00 | ORB-long ORB[4755.00,4792.05] vol=1.5x ATR=13.67 |
| Stop hit — per-position SL triggered | 2025-03-13 11:00:00 | 4815.33 | 4806.74 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 4767.70 | 4736.29 | 0.00 | ORB-long ORB[4682.70,4747.90] vol=1.7x ATR=14.52 |
| Stop hit — per-position SL triggered | 2025-03-18 09:40:00 | 4753.18 | 4738.63 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 11:10:00 | 4733.70 | 4751.81 | 0.00 | ORB-short ORB[4746.55,4797.00] vol=2.5x ATR=7.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 11:55:00 | 4722.64 | 4744.92 | 0.00 | T1 1.5R @ 4722.64 |
| Stop hit — per-position SL triggered | 2025-03-19 12:15:00 | 4733.70 | 4743.18 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:00:00 | 4854.50 | 4829.96 | 0.00 | ORB-long ORB[4780.05,4832.50] vol=2.6x ATR=12.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 11:35:00 | 4873.14 | 4837.03 | 0.00 | T1 1.5R @ 4873.14 |
| Stop hit — per-position SL triggered | 2025-03-27 14:50:00 | 4854.50 | 4859.65 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-04-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 09:40:00 | 5027.45 | 5017.93 | 0.00 | ORB-long ORB[4982.00,5026.45] vol=5.4x ATR=16.88 |
| Stop hit — per-position SL triggered | 2025-04-03 09:45:00 | 5010.57 | 5017.07 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-04-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 10:05:00 | 5100.65 | 5081.73 | 0.00 | ORB-long ORB[5015.05,5088.05] vol=1.5x ATR=17.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 10:20:00 | 5126.36 | 5088.29 | 0.00 | T1 1.5R @ 5126.36 |
| Stop hit — per-position SL triggered | 2025-04-04 10:30:00 | 5100.65 | 5090.49 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 5447.80 | 5437.24 | 0.00 | ORB-long ORB[5406.30,5444.70] vol=1.6x ATR=14.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:00:00 | 5469.07 | 5443.77 | 0.00 | T1 1.5R @ 5469.07 |
| Stop hit — per-position SL triggered | 2025-04-23 10:10:00 | 5447.80 | 5444.36 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-04-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:10:00 | 5586.20 | 5553.00 | 0.00 | ORB-long ORB[5510.10,5576.90] vol=1.7x ATR=19.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:20:00 | 5615.53 | 5568.29 | 0.00 | T1 1.5R @ 5615.53 |
| Stop hit — per-position SL triggered | 2025-04-24 10:25:00 | 5586.20 | 5569.51 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 11:15:00 | 5371.00 | 5434.11 | 0.00 | ORB-short ORB[5462.10,5521.00] vol=1.6x ATR=16.23 |
| Stop hit — per-position SL triggered | 2025-04-25 12:10:00 | 5387.23 | 5428.08 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 10:30:00 | 5484.00 | 5448.00 | 0.00 | ORB-long ORB[5420.00,5472.80] vol=2.4x ATR=15.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 11:00:00 | 5506.67 | 5465.89 | 0.00 | T1 1.5R @ 5506.67 |
| Stop hit — per-position SL triggered | 2025-04-29 11:15:00 | 5484.00 | 5468.17 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-05-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:35:00 | 5380.00 | 5350.89 | 0.00 | ORB-long ORB[5314.50,5367.50] vol=1.6x ATR=13.20 |
| Stop hit — per-position SL triggered | 2025-05-08 10:55:00 | 5366.80 | 5354.45 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-13 10:45:00 | 5118.95 | 2024-05-13 11:30:00 | 5163.10 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2024-05-13 10:45:00 | 5118.95 | 2024-05-13 14:40:00 | 5118.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-14 10:45:00 | 5062.60 | 2024-05-14 11:25:00 | 5042.09 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-14 10:45:00 | 5062.60 | 2024-05-14 11:45:00 | 5062.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-15 11:00:00 | 5090.00 | 2024-05-15 11:25:00 | 5067.08 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-15 11:00:00 | 5090.00 | 2024-05-15 12:25:00 | 5090.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 09:30:00 | 5028.35 | 2024-05-16 10:05:00 | 5043.98 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-21 10:40:00 | 5118.85 | 2024-05-21 11:05:00 | 5104.77 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-22 09:50:00 | 5269.15 | 2024-05-22 10:05:00 | 5251.48 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-05-24 10:45:00 | 5280.15 | 2024-05-24 11:05:00 | 5270.34 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-27 09:40:00 | 5197.40 | 2024-05-27 10:10:00 | 5176.83 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-05-27 09:40:00 | 5197.40 | 2024-05-27 11:50:00 | 5178.75 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2024-05-28 11:05:00 | 5252.00 | 2024-05-28 11:25:00 | 5241.62 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-05-29 10:50:00 | 5248.90 | 2024-05-29 11:05:00 | 5235.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-04 09:40:00 | 5230.00 | 2024-06-04 09:45:00 | 5261.27 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-06-04 09:40:00 | 5230.00 | 2024-06-04 10:00:00 | 5230.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-10 10:35:00 | 5516.60 | 2024-06-10 11:15:00 | 5545.19 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-06-10 10:35:00 | 5516.60 | 2024-06-10 11:55:00 | 5516.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 11:10:00 | 5447.30 | 2024-06-27 11:15:00 | 5433.23 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-02 10:55:00 | 5445.65 | 2024-07-02 11:05:00 | 5455.76 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-07-05 11:10:00 | 5471.70 | 2024-07-05 11:20:00 | 5485.41 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2024-07-05 11:10:00 | 5471.70 | 2024-07-05 15:20:00 | 5551.90 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2024-07-08 09:50:00 | 5595.15 | 2024-07-08 10:05:00 | 5577.86 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-10 09:40:00 | 5751.85 | 2024-07-10 09:50:00 | 5777.65 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-10 09:40:00 | 5751.85 | 2024-07-10 10:00:00 | 5751.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-15 09:30:00 | 5838.50 | 2024-07-15 10:25:00 | 5826.91 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-16 11:15:00 | 5839.25 | 2024-07-16 11:50:00 | 5830.03 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-07-18 10:40:00 | 5916.50 | 2024-07-18 11:00:00 | 5902.37 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-19 10:25:00 | 5963.00 | 2024-07-19 10:50:00 | 5943.73 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-23 11:15:00 | 5919.00 | 2024-07-23 11:25:00 | 5939.26 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-07-23 11:15:00 | 5919.00 | 2024-07-23 12:15:00 | 5919.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-01 11:10:00 | 5816.55 | 2024-08-01 11:50:00 | 5835.63 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-01 11:10:00 | 5816.55 | 2024-08-01 12:30:00 | 5816.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-09 10:55:00 | 5743.45 | 2024-08-09 11:05:00 | 5754.85 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-12 11:15:00 | 5701.90 | 2024-08-12 11:30:00 | 5714.81 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-13 11:05:00 | 5708.00 | 2024-08-13 11:10:00 | 5696.09 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-16 09:45:00 | 5685.10 | 2024-08-16 10:00:00 | 5705.20 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-16 09:45:00 | 5685.10 | 2024-08-16 10:15:00 | 5688.20 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2024-08-23 11:00:00 | 5815.00 | 2024-08-23 11:10:00 | 5824.35 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-08-26 10:30:00 | 5855.10 | 2024-08-26 10:40:00 | 5844.94 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-08-28 10:45:00 | 5701.45 | 2024-08-28 11:20:00 | 5712.13 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-08-29 11:15:00 | 5751.00 | 2024-08-29 12:05:00 | 5766.02 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-08-29 11:15:00 | 5751.00 | 2024-08-29 14:20:00 | 5788.00 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2024-09-06 09:30:00 | 5875.00 | 2024-09-06 09:35:00 | 5862.44 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-09 09:40:00 | 5909.95 | 2024-09-09 09:50:00 | 5892.07 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-11 10:55:00 | 6019.95 | 2024-09-11 11:55:00 | 6039.72 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-09-11 10:55:00 | 6019.95 | 2024-09-11 12:05:00 | 6019.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 10:05:00 | 6189.00 | 2024-09-19 10:40:00 | 6214.02 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-19 10:05:00 | 6189.00 | 2024-09-19 11:45:00 | 6189.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 09:40:00 | 6148.40 | 2024-09-25 10:05:00 | 6129.32 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-09-25 09:40:00 | 6148.40 | 2024-09-25 12:35:00 | 6148.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 10:55:00 | 6308.60 | 2024-09-27 11:10:00 | 6294.64 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-03 11:15:00 | 6349.85 | 2024-10-03 11:20:00 | 6365.99 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-07 11:10:00 | 6130.15 | 2024-10-07 11:25:00 | 6149.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-09 10:10:00 | 6117.50 | 2024-10-09 11:20:00 | 6140.79 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-10 11:00:00 | 6016.00 | 2024-10-10 11:55:00 | 5989.50 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-10 11:00:00 | 6016.00 | 2024-10-10 12:20:00 | 6016.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-14 10:35:00 | 5910.00 | 2024-10-14 10:40:00 | 5925.10 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-17 11:15:00 | 6032.55 | 2024-10-17 11:30:00 | 6046.39 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-10-21 11:00:00 | 5801.20 | 2024-10-21 11:10:00 | 5780.42 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-10-21 11:00:00 | 5801.20 | 2024-10-21 13:20:00 | 5801.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-25 09:30:00 | 5694.85 | 2024-10-25 09:35:00 | 5679.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-28 11:05:00 | 5757.10 | 2024-10-28 11:30:00 | 5778.04 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-10-28 11:05:00 | 5757.10 | 2024-10-28 11:40:00 | 5757.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 10:15:00 | 5864.00 | 2024-11-11 10:25:00 | 5837.47 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-11-14 11:15:00 | 4991.85 | 2024-11-14 12:20:00 | 4970.15 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-11-14 11:15:00 | 4991.85 | 2024-11-14 15:20:00 | 4919.20 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2024-11-27 10:50:00 | 4953.00 | 2024-11-27 10:55:00 | 4963.27 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-11-28 10:35:00 | 4948.35 | 2024-11-28 11:55:00 | 4926.91 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-11-28 10:35:00 | 4948.35 | 2024-11-28 15:20:00 | 4931.95 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2024-12-05 10:55:00 | 4820.00 | 2024-12-05 12:05:00 | 4830.30 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-11 09:35:00 | 4866.65 | 2024-12-11 09:40:00 | 4857.37 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-16 11:00:00 | 4809.40 | 2024-12-16 11:10:00 | 4798.13 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2024-12-16 11:00:00 | 4809.40 | 2024-12-16 11:20:00 | 4809.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 10:20:00 | 4782.30 | 2024-12-17 10:40:00 | 4792.21 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-24 11:15:00 | 4750.95 | 2024-12-24 14:10:00 | 4740.41 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-26 09:50:00 | 4722.55 | 2024-12-26 09:55:00 | 4733.01 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-31 11:15:00 | 4760.45 | 2024-12-31 11:20:00 | 4770.34 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-01-01 10:50:00 | 4812.00 | 2025-01-01 11:20:00 | 4800.77 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-02 09:30:00 | 4745.00 | 2025-01-02 09:35:00 | 4755.01 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-01-03 10:30:00 | 4831.10 | 2025-01-03 10:45:00 | 4821.59 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-01-06 09:40:00 | 4806.80 | 2025-01-06 09:50:00 | 4820.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-07 09:30:00 | 4863.10 | 2025-01-07 12:05:00 | 4847.39 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-09 09:55:00 | 4874.30 | 2025-01-09 10:05:00 | 4892.10 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-01-09 09:55:00 | 4874.30 | 2025-01-09 11:30:00 | 4941.95 | TARGET_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2025-01-16 10:45:00 | 4760.15 | 2025-01-16 11:05:00 | 4773.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-21 10:20:00 | 4877.10 | 2025-01-21 10:45:00 | 4888.92 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-24 10:10:00 | 5068.70 | 2025-01-24 10:15:00 | 5055.56 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-28 10:40:00 | 5127.10 | 2025-01-28 11:00:00 | 5142.56 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-30 10:20:00 | 5085.05 | 2025-01-30 11:25:00 | 5071.19 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-02-14 10:25:00 | 4937.35 | 2025-02-14 10:50:00 | 4922.53 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-28 10:45:00 | 4704.85 | 2025-02-28 10:55:00 | 4686.17 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-02-28 10:45:00 | 4704.85 | 2025-02-28 11:05:00 | 4704.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-03 11:00:00 | 4639.00 | 2025-03-03 12:30:00 | 4625.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-05 10:50:00 | 4633.45 | 2025-03-05 10:55:00 | 4622.74 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-03-10 09:35:00 | 4785.05 | 2025-03-10 09:55:00 | 4804.85 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-03-10 09:35:00 | 4785.05 | 2025-03-10 10:30:00 | 4785.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-13 10:10:00 | 4829.00 | 2025-03-13 11:00:00 | 4815.33 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-18 09:35:00 | 4767.70 | 2025-03-18 09:40:00 | 4753.18 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-19 11:10:00 | 4733.70 | 2025-03-19 11:55:00 | 4722.64 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-03-19 11:10:00 | 4733.70 | 2025-03-19 12:15:00 | 4733.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-27 11:00:00 | 4854.50 | 2025-03-27 11:35:00 | 4873.14 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-03-27 11:00:00 | 4854.50 | 2025-03-27 14:50:00 | 4854.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-03 09:40:00 | 5027.45 | 2025-04-03 09:45:00 | 5010.57 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-04 10:05:00 | 5100.65 | 2025-04-04 10:20:00 | 5126.36 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-04 10:05:00 | 5100.65 | 2025-04-04 10:30:00 | 5100.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-23 09:35:00 | 5447.80 | 2025-04-23 10:00:00 | 5469.07 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-04-23 09:35:00 | 5447.80 | 2025-04-23 10:10:00 | 5447.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-24 10:10:00 | 5586.20 | 2025-04-24 10:20:00 | 5615.53 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-24 10:10:00 | 5586.20 | 2025-04-24 10:25:00 | 5586.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 11:15:00 | 5371.00 | 2025-04-25 12:10:00 | 5387.23 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-29 10:30:00 | 5484.00 | 2025-04-29 11:00:00 | 5506.67 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-04-29 10:30:00 | 5484.00 | 2025-04-29 11:15:00 | 5484.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-08 10:35:00 | 5380.00 | 2025-05-08 10:55:00 | 5366.80 | STOP_HIT | 1.00 | -0.25% |
