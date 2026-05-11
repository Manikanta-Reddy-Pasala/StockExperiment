# Pfizer Ltd. (PFIZER)

## Backtest Summary

- **Window:** 2025-08-11 09:15:00 → 2026-05-08 15:25:00 (13588 bars)
- **Last close:** 4793.00
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
| ENTRY1 | 69 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 10 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 59
- **Target hits / Stop hits / Partials:** 10 / 59 / 19
- **Avg / median % per leg:** 0.04% / -0.18%
- **Sum % (uncompounded):** 3.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 9 | 20.0% | 2 | 36 | 7 | -0.03% | -1.1% |
| BUY @ 2nd Alert (retest1) | 45 | 9 | 20.0% | 2 | 36 | 7 | -0.03% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 20 | 46.5% | 8 | 23 | 12 | 0.12% | 5.1% |
| SELL @ 2nd Alert (retest1) | 43 | 20 | 46.5% | 8 | 23 | 12 | 0.12% | 5.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 88 | 29 | 33.0% | 10 | 59 | 19 | 0.04% | 4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 10:55:00 | 5643.50 | 5665.61 | 0.00 | ORB-short ORB[5650.00,5702.00] vol=2.3x ATR=15.37 |
| Stop hit — per-position SL triggered | 2025-08-20 11:30:00 | 5658.87 | 5664.47 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-08-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:55:00 | 5740.00 | 5689.35 | 0.00 | ORB-long ORB[5656.50,5714.50] vol=3.8x ATR=15.52 |
| Stop hit — per-position SL triggered | 2025-08-21 11:00:00 | 5724.48 | 5693.23 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:40:00 | 5730.00 | 5720.82 | 0.00 | ORB-long ORB[5683.00,5729.00] vol=3.6x ATR=20.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:15:00 | 5761.03 | 5731.11 | 0.00 | T1 1.5R @ 5761.03 |
| Stop hit — per-position SL triggered | 2025-08-22 10:30:00 | 5730.00 | 5732.54 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:35:00 | 5140.00 | 5121.47 | 0.00 | ORB-long ORB[5085.50,5129.50] vol=1.7x ATR=17.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 11:15:00 | 5166.14 | 5136.17 | 0.00 | T1 1.5R @ 5166.14 |
| Stop hit — per-position SL triggered | 2025-09-03 14:55:00 | 5140.00 | 5155.26 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-09-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 10:25:00 | 5142.00 | 5115.97 | 0.00 | ORB-long ORB[5082.50,5141.00] vol=2.1x ATR=14.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 11:35:00 | 5163.87 | 5126.01 | 0.00 | T1 1.5R @ 5163.87 |
| Stop hit — per-position SL triggered | 2025-09-09 11:55:00 | 5142.00 | 5127.18 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-09-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:40:00 | 5169.00 | 5155.37 | 0.00 | ORB-long ORB[5121.50,5166.00] vol=1.6x ATR=10.78 |
| Stop hit — per-position SL triggered | 2025-09-10 11:20:00 | 5158.22 | 5157.31 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-09-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:45:00 | 5209.00 | 5169.74 | 0.00 | ORB-long ORB[5132.50,5187.00] vol=2.4x ATR=16.89 |
| Stop hit — per-position SL triggered | 2025-09-12 09:55:00 | 5192.11 | 5175.77 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-09-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 11:00:00 | 5181.00 | 5172.21 | 0.00 | ORB-long ORB[5140.50,5180.00] vol=2.4x ATR=13.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 12:00:00 | 5201.31 | 5177.79 | 0.00 | T1 1.5R @ 5201.31 |
| Stop hit — per-position SL triggered | 2025-09-17 13:50:00 | 5181.00 | 5181.33 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:45:00 | 5224.50 | 5204.45 | 0.00 | ORB-long ORB[5172.00,5217.00] vol=3.6x ATR=16.87 |
| Stop hit — per-position SL triggered | 2025-09-18 10:00:00 | 5207.63 | 5211.63 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:55:00 | 5121.50 | 5148.38 | 0.00 | ORB-short ORB[5160.00,5190.00] vol=3.6x ATR=16.94 |
| Stop hit — per-position SL triggered | 2025-09-19 10:05:00 | 5138.44 | 5146.59 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-10-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 10:20:00 | 5240.00 | 5217.74 | 0.00 | ORB-long ORB[5175.00,5235.00] vol=1.9x ATR=11.58 |
| Stop hit — per-position SL triggered | 2025-10-07 11:25:00 | 5228.42 | 5228.79 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-10-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:30:00 | 5298.00 | 5270.42 | 0.00 | ORB-long ORB[5235.00,5276.50] vol=1.5x ATR=12.44 |
| Stop hit — per-position SL triggered | 2025-10-10 10:45:00 | 5285.56 | 5272.89 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-10-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:45:00 | 5363.50 | 5330.82 | 0.00 | ORB-long ORB[5270.00,5348.00] vol=2.3x ATR=20.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 12:20:00 | 5394.14 | 5367.56 | 0.00 | T1 1.5R @ 5394.14 |
| Target hit | 2025-10-13 15:20:00 | 5522.50 | 5457.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-10-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:10:00 | 5406.00 | 5455.90 | 0.00 | ORB-short ORB[5470.00,5527.00] vol=3.8x ATR=15.61 |
| Stop hit — per-position SL triggered | 2025-10-14 12:10:00 | 5421.61 | 5450.11 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-10-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 10:20:00 | 5382.00 | 5404.01 | 0.00 | ORB-short ORB[5390.00,5431.50] vol=2.8x ATR=14.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 12:40:00 | 5359.87 | 5385.76 | 0.00 | T1 1.5R @ 5359.87 |
| Target hit | 2025-10-15 15:20:00 | 5330.50 | 5372.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-10-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:40:00 | 5370.00 | 5354.15 | 0.00 | ORB-long ORB[5322.50,5360.50] vol=1.7x ATR=10.95 |
| Stop hit — per-position SL triggered | 2025-10-16 09:50:00 | 5359.05 | 5354.65 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-10-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:50:00 | 5265.00 | 5289.67 | 0.00 | ORB-short ORB[5270.00,5325.00] vol=2.4x ATR=14.90 |
| Stop hit — per-position SL triggered | 2025-10-17 11:05:00 | 5279.90 | 5288.52 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-10-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 11:10:00 | 5306.50 | 5329.34 | 0.00 | ORB-short ORB[5325.00,5357.00] vol=1.7x ATR=10.73 |
| Stop hit — per-position SL triggered | 2025-10-20 11:45:00 | 5317.23 | 5327.72 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 11:05:00 | 5410.00 | 5343.71 | 0.00 | ORB-long ORB[5310.00,5360.50] vol=7.1x ATR=18.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 11:10:00 | 5438.43 | 5368.28 | 0.00 | T1 1.5R @ 5438.43 |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 5410.00 | 5378.91 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-10-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:50:00 | 5235.50 | 5264.04 | 0.00 | ORB-short ORB[5277.50,5300.00] vol=1.8x ATR=9.22 |
| Stop hit — per-position SL triggered | 2025-10-28 12:10:00 | 5244.72 | 5257.38 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-10-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:10:00 | 5270.00 | 5283.31 | 0.00 | ORB-short ORB[5271.50,5303.50] vol=3.8x ATR=9.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:15:00 | 5255.36 | 5276.42 | 0.00 | T1 1.5R @ 5255.36 |
| Stop hit — per-position SL triggered | 2025-10-29 11:50:00 | 5270.00 | 5274.88 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-10-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 11:10:00 | 5250.00 | 5267.24 | 0.00 | ORB-short ORB[5271.50,5322.00] vol=1.7x ATR=9.74 |
| Stop hit — per-position SL triggered | 2025-10-31 11:20:00 | 5259.74 | 5266.74 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:15:00 | 5144.00 | 5170.01 | 0.00 | ORB-short ORB[5150.50,5180.00] vol=2.1x ATR=9.19 |
| Stop hit — per-position SL triggered | 2025-11-04 12:20:00 | 5153.19 | 5165.78 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 11:15:00 | 5127.00 | 5109.04 | 0.00 | ORB-long ORB[5084.50,5110.00] vol=6.0x ATR=8.69 |
| Stop hit — per-position SL triggered | 2025-11-06 11:25:00 | 5118.31 | 5109.26 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-11-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:40:00 | 5062.50 | 5102.27 | 0.00 | ORB-short ORB[5086.00,5144.50] vol=1.7x ATR=18.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:30:00 | 5034.65 | 5068.88 | 0.00 | T1 1.5R @ 5034.65 |
| Target hit | 2025-11-11 15:20:00 | 4995.00 | 5021.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-11-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 09:40:00 | 5080.00 | 5062.80 | 0.00 | ORB-long ORB[5047.00,5074.50] vol=2.1x ATR=11.38 |
| Stop hit — per-position SL triggered | 2025-11-18 09:55:00 | 5068.62 | 5068.09 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:30:00 | 5012.00 | 5020.01 | 0.00 | ORB-short ORB[5013.50,5041.00] vol=1.8x ATR=11.16 |
| Stop hit — per-position SL triggered | 2025-11-19 11:05:00 | 5023.16 | 5013.77 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-12-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 09:35:00 | 5030.40 | 5001.41 | 0.00 | ORB-long ORB[4951.00,4991.30] vol=5.4x ATR=13.63 |
| Stop hit — per-position SL triggered | 2025-12-02 10:05:00 | 5016.77 | 5013.79 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-12-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:05:00 | 4930.50 | 4944.68 | 0.00 | ORB-short ORB[4950.00,4984.30] vol=1.7x ATR=11.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:15:00 | 4912.75 | 4940.92 | 0.00 | T1 1.5R @ 4912.75 |
| Stop hit — per-position SL triggered | 2025-12-03 10:20:00 | 4930.50 | 4940.81 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-12-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:10:00 | 4938.90 | 4964.55 | 0.00 | ORB-short ORB[4951.00,4996.70] vol=2.0x ATR=13.44 |
| Stop hit — per-position SL triggered | 2025-12-08 11:50:00 | 4952.34 | 4955.50 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-12-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 09:45:00 | 5002.00 | 4973.53 | 0.00 | ORB-long ORB[4951.70,4979.00] vol=2.2x ATR=15.48 |
| Stop hit — per-position SL triggered | 2025-12-09 09:55:00 | 4986.52 | 4980.43 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-12-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:40:00 | 5012.00 | 4993.69 | 0.00 | ORB-long ORB[4961.00,4995.60] vol=3.9x ATR=10.86 |
| Stop hit — per-position SL triggered | 2025-12-10 10:10:00 | 5001.14 | 4999.73 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-12-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:45:00 | 5030.00 | 5017.22 | 0.00 | ORB-long ORB[4982.50,5024.30] vol=1.9x ATR=10.41 |
| Stop hit — per-position SL triggered | 2025-12-15 12:30:00 | 5019.59 | 5021.84 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:35:00 | 5061.00 | 5036.56 | 0.00 | ORB-long ORB[4995.20,5051.50] vol=1.5x ATR=15.17 |
| Stop hit — per-position SL triggered | 2025-12-17 09:45:00 | 5045.83 | 5039.66 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-12-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 09:40:00 | 5054.00 | 5063.16 | 0.00 | ORB-short ORB[5061.40,5100.00] vol=2.8x ATR=13.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 10:30:00 | 5033.61 | 5057.19 | 0.00 | T1 1.5R @ 5033.61 |
| Target hit | 2025-12-23 12:10:00 | 5041.60 | 5040.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — BUY (started 2025-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:35:00 | 5108.00 | 5085.54 | 0.00 | ORB-long ORB[5035.00,5091.00] vol=1.7x ATR=13.98 |
| Stop hit — per-position SL triggered | 2025-12-24 12:25:00 | 5094.02 | 5101.82 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-12-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:35:00 | 5016.70 | 5029.45 | 0.00 | ORB-short ORB[5022.20,5062.00] vol=1.9x ATR=10.44 |
| Stop hit — per-position SL triggered | 2025-12-29 09:40:00 | 5027.14 | 5029.55 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:50:00 | 5003.90 | 4964.02 | 0.00 | ORB-long ORB[4920.00,4962.50] vol=5.8x ATR=14.90 |
| Stop hit — per-position SL triggered | 2025-12-31 11:10:00 | 4989.00 | 4973.97 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-01-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:45:00 | 4903.30 | 4890.59 | 0.00 | ORB-long ORB[4870.00,4899.90] vol=1.5x ATR=10.45 |
| Stop hit — per-position SL triggered | 2026-01-05 10:00:00 | 4892.85 | 4891.49 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-01-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:05:00 | 4843.20 | 4845.87 | 0.00 | ORB-short ORB[4844.10,4868.20] vol=1.8x ATR=7.66 |
| Stop hit — per-position SL triggered | 2026-01-06 10:30:00 | 4850.86 | 4844.48 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-01-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:30:00 | 4993.00 | 4962.84 | 0.00 | ORB-long ORB[4917.30,4957.50] vol=3.5x ATR=16.07 |
| Stop hit — per-position SL triggered | 2026-01-07 09:45:00 | 4976.93 | 4971.14 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 4858.80 | 4884.68 | 0.00 | ORB-short ORB[4880.00,4922.70] vol=1.6x ATR=7.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:00:00 | 4847.49 | 4876.66 | 0.00 | T1 1.5R @ 4847.49 |
| Stop hit — per-position SL triggered | 2026-01-08 12:10:00 | 4858.80 | 4875.91 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-01-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:45:00 | 4836.30 | 4838.19 | 0.00 | ORB-short ORB[4840.00,4857.30] vol=1.6x ATR=11.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:55:00 | 4819.58 | 4833.18 | 0.00 | T1 1.5R @ 4819.58 |
| Target hit | 2026-01-14 12:15:00 | 4830.50 | 4828.12 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — SELL (started 2026-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 09:35:00 | 4797.60 | 4809.89 | 0.00 | ORB-short ORB[4807.90,4838.20] vol=2.1x ATR=9.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:50:00 | 4783.34 | 4799.62 | 0.00 | T1 1.5R @ 4783.34 |
| Target hit | 2026-01-16 15:20:00 | 4761.90 | 4777.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2026-01-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:35:00 | 4615.40 | 4579.90 | 0.00 | ORB-long ORB[4544.50,4587.80] vol=1.6x ATR=14.91 |
| Stop hit — per-position SL triggered | 2026-01-27 09:40:00 | 4600.49 | 4585.64 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-01-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 09:45:00 | 4675.40 | 4658.97 | 0.00 | ORB-long ORB[4600.80,4668.00] vol=1.6x ATR=18.88 |
| Stop hit — per-position SL triggered | 2026-01-28 10:25:00 | 4656.52 | 4660.22 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:00:00 | 4597.10 | 4565.60 | 0.00 | ORB-long ORB[4550.00,4582.80] vol=3.3x ATR=8.43 |
| Stop hit — per-position SL triggered | 2026-02-01 11:05:00 | 4588.67 | 4566.81 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-02-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 10:05:00 | 4496.70 | 4507.50 | 0.00 | ORB-short ORB[4499.30,4550.00] vol=3.4x ATR=15.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:50:00 | 4473.14 | 4498.70 | 0.00 | T1 1.5R @ 4473.14 |
| Target hit | 2026-02-02 12:25:00 | 4494.80 | 4494.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 49 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 5029.60 | 5042.05 | 0.00 | ORB-short ORB[5031.20,5072.30] vol=1.7x ATR=12.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:05:00 | 5010.29 | 5039.03 | 0.00 | T1 1.5R @ 5010.29 |
| Stop hit — per-position SL triggered | 2026-02-16 12:40:00 | 5029.60 | 5038.25 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 5138.80 | 5129.37 | 0.00 | ORB-long ORB[5070.60,5110.00] vol=2.9x ATR=16.06 |
| Stop hit — per-position SL triggered | 2026-02-25 11:25:00 | 5122.74 | 5132.43 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-02-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:25:00 | 5127.70 | 5108.72 | 0.00 | ORB-long ORB[5060.00,5111.00] vol=2.2x ATR=16.47 |
| Stop hit — per-position SL triggered | 2026-02-27 10:45:00 | 5111.23 | 5109.63 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-03-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:55:00 | 4811.00 | 4831.92 | 0.00 | ORB-short ORB[4817.00,4864.00] vol=2.9x ATR=20.25 |
| Stop hit — per-position SL triggered | 2026-03-04 10:00:00 | 4831.25 | 4831.70 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-03-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:50:00 | 4898.00 | 4877.58 | 0.00 | ORB-long ORB[4854.00,4894.50] vol=2.6x ATR=18.06 |
| Stop hit — per-position SL triggered | 2026-03-05 10:10:00 | 4879.94 | 4878.29 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 4763.00 | 4774.78 | 0.00 | ORB-short ORB[4766.50,4825.00] vol=2.2x ATR=17.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:10:00 | 4736.02 | 4766.39 | 0.00 | T1 1.5R @ 4736.02 |
| Target hit | 2026-03-11 15:20:00 | 4703.00 | 4740.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2026-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:35:00 | 4617.00 | 4639.29 | 0.00 | ORB-short ORB[4633.00,4690.50] vol=1.8x ATR=14.82 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 4631.82 | 4638.54 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-03-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 09:55:00 | 4682.50 | 4625.45 | 0.00 | ORB-long ORB[4587.00,4653.00] vol=2.2x ATR=28.72 |
| Stop hit — per-position SL triggered | 2026-03-24 10:10:00 | 4653.78 | 4636.12 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 4716.00 | 4691.08 | 0.00 | ORB-long ORB[4651.00,4707.00] vol=5.0x ATR=12.61 |
| Stop hit — per-position SL triggered | 2026-03-27 11:20:00 | 4703.39 | 4692.41 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 11:15:00 | 4722.00 | 4661.67 | 0.00 | ORB-long ORB[4611.00,4671.00] vol=5.9x ATR=15.80 |
| Stop hit — per-position SL triggered | 2026-03-30 11:25:00 | 4706.20 | 4669.75 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-04-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:30:00 | 4774.80 | 4792.85 | 0.00 | ORB-short ORB[4781.00,4839.90] vol=1.7x ATR=15.26 |
| Stop hit — per-position SL triggered | 2026-04-07 09:40:00 | 4790.06 | 4789.63 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 11:00:00 | 4802.10 | 4808.06 | 0.00 | ORB-short ORB[4812.50,4860.00] vol=1.7x ATR=7.36 |
| Stop hit — per-position SL triggered | 2026-04-09 11:05:00 | 4809.46 | 4808.04 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 4802.80 | 4794.17 | 0.00 | ORB-long ORB[4750.00,4783.30] vol=3.8x ATR=14.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:50:00 | 4824.97 | 4803.34 | 0.00 | T1 1.5R @ 4824.97 |
| Target hit | 2026-04-10 15:20:00 | 4868.00 | 4828.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2026-04-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:35:00 | 4923.20 | 4898.04 | 0.00 | ORB-long ORB[4865.40,4920.30] vol=1.8x ATR=14.38 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 4908.82 | 4899.62 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-04-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:35:00 | 4979.40 | 4933.03 | 0.00 | ORB-long ORB[4888.00,4950.00] vol=2.6x ATR=16.06 |
| Stop hit — per-position SL triggered | 2026-04-16 10:40:00 | 4963.34 | 4935.20 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-04-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:10:00 | 4934.10 | 4903.76 | 0.00 | ORB-long ORB[4864.80,4887.90] vol=2.0x ATR=12.16 |
| Stop hit — per-position SL triggered | 2026-04-17 10:20:00 | 4921.94 | 4905.98 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:15:00 | 4875.20 | 4885.76 | 0.00 | ORB-short ORB[4880.40,4917.20] vol=1.5x ATR=10.19 |
| Stop hit — per-position SL triggered | 2026-04-21 10:30:00 | 4885.39 | 4885.37 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 4783.30 | 4796.46 | 0.00 | ORB-short ORB[4790.00,4825.00] vol=1.8x ATR=8.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 13:10:00 | 4771.03 | 4789.10 | 0.00 | T1 1.5R @ 4771.03 |
| Target hit | 2026-04-29 15:20:00 | 4760.10 | 4780.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2026-04-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:50:00 | 4712.30 | 4731.75 | 0.00 | ORB-short ORB[4721.00,4789.00] vol=2.5x ATR=11.73 |
| Stop hit — per-position SL triggered | 2026-04-30 11:10:00 | 4724.03 | 4726.96 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 4767.60 | 4748.50 | 0.00 | ORB-long ORB[4701.10,4756.90] vol=2.0x ATR=14.55 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 4753.05 | 4755.89 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:50:00 | 4752.10 | 4777.97 | 0.00 | ORB-short ORB[4783.00,4802.30] vol=3.0x ATR=12.00 |
| Stop hit — per-position SL triggered | 2026-05-07 10:05:00 | 4764.10 | 4773.91 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-08-20 10:55:00 | 5643.50 | 2025-08-20 11:30:00 | 5658.87 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-21 10:55:00 | 5740.00 | 2025-08-21 11:00:00 | 5724.48 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-22 09:40:00 | 5730.00 | 2025-08-22 10:15:00 | 5761.03 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-08-22 09:40:00 | 5730.00 | 2025-08-22 10:30:00 | 5730.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 09:35:00 | 5140.00 | 2025-09-03 11:15:00 | 5166.14 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-09-03 09:35:00 | 5140.00 | 2025-09-03 14:55:00 | 5140.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-09 10:25:00 | 5142.00 | 2025-09-09 11:35:00 | 5163.87 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-09 10:25:00 | 5142.00 | 2025-09-09 11:55:00 | 5142.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-10 10:40:00 | 5169.00 | 2025-09-10 11:20:00 | 5158.22 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-12 09:45:00 | 5209.00 | 2025-09-12 09:55:00 | 5192.11 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-17 11:00:00 | 5181.00 | 2025-09-17 12:00:00 | 5201.31 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-17 11:00:00 | 5181.00 | 2025-09-17 13:50:00 | 5181.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 09:45:00 | 5224.50 | 2025-09-18 10:00:00 | 5207.63 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-19 09:55:00 | 5121.50 | 2025-09-19 10:05:00 | 5138.44 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-07 10:20:00 | 5240.00 | 2025-10-07 11:25:00 | 5228.42 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-10 10:30:00 | 5298.00 | 2025-10-10 10:45:00 | 5285.56 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-13 09:45:00 | 5363.50 | 2025-10-13 12:20:00 | 5394.14 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-13 09:45:00 | 5363.50 | 2025-10-13 15:20:00 | 5522.50 | TARGET_HIT | 0.50 | 2.96% |
| SELL | retest1 | 2025-10-14 11:10:00 | 5406.00 | 2025-10-14 12:10:00 | 5421.61 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-15 10:20:00 | 5382.00 | 2025-10-15 12:40:00 | 5359.87 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-15 10:20:00 | 5382.00 | 2025-10-15 15:20:00 | 5330.50 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2025-10-16 09:40:00 | 5370.00 | 2025-10-16 09:50:00 | 5359.05 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-17 10:50:00 | 5265.00 | 2025-10-17 11:05:00 | 5279.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-20 11:10:00 | 5306.50 | 2025-10-20 11:45:00 | 5317.23 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-23 11:05:00 | 5410.00 | 2025-10-23 11:10:00 | 5438.43 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-23 11:05:00 | 5410.00 | 2025-10-23 11:15:00 | 5410.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-28 10:50:00 | 5235.50 | 2025-10-28 12:10:00 | 5244.72 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-29 11:10:00 | 5270.00 | 2025-10-29 11:15:00 | 5255.36 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-10-29 11:10:00 | 5270.00 | 2025-10-29 11:50:00 | 5270.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-31 11:10:00 | 5250.00 | 2025-10-31 11:20:00 | 5259.74 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-04 11:15:00 | 5144.00 | 2025-11-04 12:20:00 | 5153.19 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-06 11:15:00 | 5127.00 | 2025-11-06 11:25:00 | 5118.31 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-11-11 09:40:00 | 5062.50 | 2025-11-11 10:30:00 | 5034.65 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-11-11 09:40:00 | 5062.50 | 2025-11-11 15:20:00 | 4995.00 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2025-11-18 09:40:00 | 5080.00 | 2025-11-18 09:55:00 | 5068.62 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-19 09:30:00 | 5012.00 | 2025-11-19 11:05:00 | 5023.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-02 09:35:00 | 5030.40 | 2025-12-02 10:05:00 | 5016.77 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-03 10:05:00 | 4930.50 | 2025-12-03 10:15:00 | 4912.75 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-12-03 10:05:00 | 4930.50 | 2025-12-03 10:20:00 | 4930.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 11:10:00 | 4938.90 | 2025-12-08 11:50:00 | 4952.34 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-09 09:45:00 | 5002.00 | 2025-12-09 09:55:00 | 4986.52 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-10 09:40:00 | 5012.00 | 2025-12-10 10:10:00 | 5001.14 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-15 10:45:00 | 5030.00 | 2025-12-15 12:30:00 | 5019.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-17 09:35:00 | 5061.00 | 2025-12-17 09:45:00 | 5045.83 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-23 09:40:00 | 5054.00 | 2025-12-23 10:30:00 | 5033.61 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-23 09:40:00 | 5054.00 | 2025-12-23 12:10:00 | 5041.60 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-12-24 09:35:00 | 5108.00 | 2025-12-24 12:25:00 | 5094.02 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-29 09:35:00 | 5016.70 | 2025-12-29 09:40:00 | 5027.14 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-31 10:50:00 | 5003.90 | 2025-12-31 11:10:00 | 4989.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-05 09:45:00 | 4903.30 | 2026-01-05 10:00:00 | 4892.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-06 10:05:00 | 4843.20 | 2026-01-06 10:30:00 | 4850.86 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-01-07 09:30:00 | 4993.00 | 2026-01-07 09:45:00 | 4976.93 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-08 11:10:00 | 4858.80 | 2026-01-08 12:00:00 | 4847.49 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2026-01-08 11:10:00 | 4858.80 | 2026-01-08 12:10:00 | 4858.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-14 09:45:00 | 4836.30 | 2026-01-14 10:55:00 | 4819.58 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-14 09:45:00 | 4836.30 | 2026-01-14 12:15:00 | 4830.50 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2026-01-16 09:35:00 | 4797.60 | 2026-01-16 11:50:00 | 4783.34 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-01-16 09:35:00 | 4797.60 | 2026-01-16 15:20:00 | 4761.90 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2026-01-27 09:35:00 | 4615.40 | 2026-01-27 09:40:00 | 4600.49 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-01-28 09:45:00 | 4675.40 | 2026-01-28 10:25:00 | 4656.52 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-01 11:00:00 | 4597.10 | 2026-02-01 11:05:00 | 4588.67 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-02 10:05:00 | 4496.70 | 2026-02-02 10:50:00 | 4473.14 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-02 10:05:00 | 4496.70 | 2026-02-02 12:25:00 | 4494.80 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2026-02-16 11:15:00 | 5029.60 | 2026-02-16 12:05:00 | 5010.29 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-16 11:15:00 | 5029.60 | 2026-02-16 12:40:00 | 5029.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:05:00 | 5138.80 | 2026-02-25 11:25:00 | 5122.74 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-27 10:25:00 | 5127.70 | 2026-02-27 10:45:00 | 5111.23 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-04 09:55:00 | 4811.00 | 2026-03-04 10:00:00 | 4831.25 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-05 09:50:00 | 4898.00 | 2026-03-05 10:10:00 | 4879.94 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-11 09:40:00 | 4763.00 | 2026-03-11 13:10:00 | 4736.02 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-11 09:40:00 | 4763.00 | 2026-03-11 15:20:00 | 4703.00 | TARGET_HIT | 0.50 | 1.26% |
| SELL | retest1 | 2026-03-13 10:35:00 | 4617.00 | 2026-03-13 10:50:00 | 4631.82 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-24 09:55:00 | 4682.50 | 2026-03-24 10:10:00 | 4653.78 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2026-03-27 11:05:00 | 4716.00 | 2026-03-27 11:20:00 | 4703.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-30 11:15:00 | 4722.00 | 2026-03-30 11:25:00 | 4706.20 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-07 09:30:00 | 4774.80 | 2026-04-07 09:40:00 | 4790.06 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-09 11:00:00 | 4802.10 | 2026-04-09 11:05:00 | 4809.46 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2026-04-10 09:30:00 | 4802.80 | 2026-04-10 09:50:00 | 4824.97 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-10 09:30:00 | 4802.80 | 2026-04-10 15:20:00 | 4868.00 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2026-04-15 10:35:00 | 4923.20 | 2026-04-15 10:50:00 | 4908.82 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-16 10:35:00 | 4979.40 | 2026-04-16 10:40:00 | 4963.34 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-17 10:10:00 | 4934.10 | 2026-04-17 10:20:00 | 4921.94 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-21 10:15:00 | 4875.20 | 2026-04-21 10:30:00 | 4885.39 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-04-29 10:45:00 | 4783.30 | 2026-04-29 13:10:00 | 4771.03 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-29 10:45:00 | 4783.30 | 2026-04-29 15:20:00 | 4760.10 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-30 09:50:00 | 4712.30 | 2026-04-30 11:10:00 | 4724.03 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-04 09:40:00 | 4767.60 | 2026-05-04 10:15:00 | 4753.05 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-07 09:50:00 | 4752.10 | 2026-05-07 10:05:00 | 4764.10 | STOP_HIT | 1.00 | -0.25% |
