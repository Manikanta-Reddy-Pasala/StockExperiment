# ABB India Ltd. (ABB)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16888 bars)
- **Last close:** 7010.00
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
| ENTRY1 | 91 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 13 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 130 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 78
- **Target hits / Stop hits / Partials:** 13 / 78 / 39
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 9.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 29 | 39.2% | 7 | 45 | 22 | 0.06% | 4.3% |
| BUY @ 2nd Alert (retest1) | 74 | 29 | 39.2% | 7 | 45 | 22 | 0.06% | 4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 23 | 41.1% | 6 | 33 | 17 | 0.10% | 5.7% |
| SELL @ 2nd Alert (retest1) | 56 | 23 | 41.1% | 6 | 33 | 17 | 0.10% | 5.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 130 | 52 | 40.0% | 13 | 78 | 39 | 0.08% | 10.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:30:00 | 5623.50 | 5583.85 | 0.00 | ORB-long ORB[5528.00,5605.00] vol=2.7x ATR=23.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 09:35:00 | 5659.49 | 5597.28 | 0.00 | T1 1.5R @ 5659.49 |
| Stop hit — per-position SL triggered | 2025-05-13 09:55:00 | 5623.50 | 5604.15 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:45:00 | 5696.00 | 5669.95 | 0.00 | ORB-long ORB[5608.00,5681.00] vol=2.1x ATR=16.43 |
| Stop hit — per-position SL triggered | 2025-05-15 11:55:00 | 5679.57 | 5674.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:35:00 | 5809.00 | 5771.55 | 0.00 | ORB-long ORB[5701.00,5779.50] vol=2.1x ATR=17.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 09:40:00 | 5834.67 | 5780.12 | 0.00 | T1 1.5R @ 5834.67 |
| Stop hit — per-position SL triggered | 2025-05-16 09:50:00 | 5809.00 | 5789.69 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:10:00 | 5845.00 | 5794.65 | 0.00 | ORB-long ORB[5722.00,5782.00] vol=1.6x ATR=17.60 |
| Stop hit — per-position SL triggered | 2025-05-21 11:20:00 | 5827.40 | 5797.08 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 5972.50 | 5929.27 | 0.00 | ORB-long ORB[5875.00,5954.50] vol=1.5x ATR=20.68 |
| Stop hit — per-position SL triggered | 2025-05-23 09:35:00 | 5951.82 | 5933.09 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:40:00 | 6031.50 | 6014.07 | 0.00 | ORB-long ORB[5965.00,6030.00] vol=2.1x ATR=15.08 |
| Stop hit — per-position SL triggered | 2025-05-26 10:40:00 | 6016.42 | 6024.76 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:30:00 | 5969.50 | 5990.69 | 0.00 | ORB-short ORB[5974.00,6017.00] vol=1.7x ATR=12.11 |
| Stop hit — per-position SL triggered | 2025-05-27 09:35:00 | 5981.61 | 5989.32 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-05-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 11:10:00 | 6031.00 | 6077.82 | 0.00 | ORB-short ORB[6068.00,6118.00] vol=2.1x ATR=15.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 11:45:00 | 6008.06 | 6071.64 | 0.00 | T1 1.5R @ 6008.06 |
| Stop hit — per-position SL triggered | 2025-05-30 12:05:00 | 6031.00 | 6068.85 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 6045.00 | 6010.85 | 0.00 | ORB-long ORB[5966.00,6018.00] vol=3.4x ATR=16.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 09:35:00 | 6069.59 | 6031.39 | 0.00 | T1 1.5R @ 6069.59 |
| Stop hit — per-position SL triggered | 2025-06-03 09:45:00 | 6045.00 | 6036.09 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:15:00 | 5981.00 | 5998.32 | 0.00 | ORB-short ORB[5990.00,6037.50] vol=1.6x ATR=10.80 |
| Stop hit — per-position SL triggered | 2025-06-04 13:05:00 | 5991.80 | 5994.53 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:45:00 | 6099.50 | 6088.35 | 0.00 | ORB-long ORB[6071.00,6095.50] vol=2.3x ATR=15.13 |
| Stop hit — per-position SL triggered | 2025-06-09 09:55:00 | 6084.37 | 6090.52 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:40:00 | 6064.00 | 6040.82 | 0.00 | ORB-long ORB[6014.00,6045.00] vol=1.6x ATR=13.27 |
| Stop hit — per-position SL triggered | 2025-06-17 10:00:00 | 6050.73 | 6048.37 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 10:05:00 | 6076.50 | 6043.84 | 0.00 | ORB-long ORB[6000.00,6054.00] vol=1.5x ATR=16.34 |
| Stop hit — per-position SL triggered | 2025-06-18 10:25:00 | 6060.16 | 6056.04 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:30:00 | 5965.00 | 5912.15 | 0.00 | ORB-long ORB[5853.50,5930.00] vol=1.7x ATR=19.77 |
| Target hit | 2025-06-20 15:20:00 | 5967.00 | 5957.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2025-06-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:10:00 | 5950.00 | 5916.50 | 0.00 | ORB-long ORB[5880.00,5918.00] vol=4.0x ATR=15.70 |
| Stop hit — per-position SL triggered | 2025-06-23 12:00:00 | 5934.30 | 5920.85 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-06-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:50:00 | 6011.00 | 5970.19 | 0.00 | ORB-long ORB[5945.00,5994.00] vol=2.9x ATR=19.55 |
| Stop hit — per-position SL triggered | 2025-06-24 10:10:00 | 5991.45 | 5982.29 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:45:00 | 6110.00 | 6081.51 | 0.00 | ORB-long ORB[6030.00,6099.00] vol=3.0x ATR=17.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:55:00 | 6136.23 | 6097.32 | 0.00 | T1 1.5R @ 6136.23 |
| Target hit | 2025-06-27 10:35:00 | 6120.50 | 6122.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2025-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:35:00 | 6142.00 | 6121.16 | 0.00 | ORB-long ORB[6094.50,6128.50] vol=2.7x ATR=13.32 |
| Stop hit — per-position SL triggered | 2025-07-01 09:40:00 | 6128.68 | 6119.89 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:50:00 | 5887.00 | 5917.61 | 0.00 | ORB-short ORB[5940.50,5989.50] vol=3.4x ATR=12.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 11:00:00 | 5868.50 | 5911.47 | 0.00 | T1 1.5R @ 5868.50 |
| Stop hit — per-position SL triggered | 2025-07-02 11:10:00 | 5887.00 | 5908.23 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 11:05:00 | 5819.00 | 5835.92 | 0.00 | ORB-short ORB[5822.00,5862.00] vol=1.6x ATR=11.75 |
| Stop hit — per-position SL triggered | 2025-07-07 12:20:00 | 5830.75 | 5829.04 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 11:00:00 | 5874.50 | 5869.13 | 0.00 | ORB-long ORB[5821.00,5853.50] vol=5.8x ATR=11.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 11:15:00 | 5892.35 | 5872.01 | 0.00 | T1 1.5R @ 5892.35 |
| Stop hit — per-position SL triggered | 2025-07-09 11:25:00 | 5874.50 | 5872.25 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:45:00 | 5855.00 | 5880.69 | 0.00 | ORB-short ORB[5882.50,5929.00] vol=7.2x ATR=12.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:05:00 | 5836.04 | 5876.57 | 0.00 | T1 1.5R @ 5836.04 |
| Stop hit — per-position SL triggered | 2025-07-10 13:05:00 | 5855.00 | 5866.23 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:30:00 | 5785.50 | 5831.32 | 0.00 | ORB-short ORB[5852.00,5897.00] vol=2.8x ATR=13.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:50:00 | 5765.60 | 5820.12 | 0.00 | T1 1.5R @ 5765.60 |
| Stop hit — per-position SL triggered | 2025-07-11 10:55:00 | 5785.50 | 5819.41 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 09:35:00 | 5714.00 | 5748.60 | 0.00 | ORB-short ORB[5720.50,5781.50] vol=2.2x ATR=19.68 |
| Stop hit — per-position SL triggered | 2025-07-14 09:40:00 | 5733.68 | 5744.63 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:10:00 | 5704.00 | 5688.77 | 0.00 | ORB-long ORB[5660.50,5695.00] vol=2.8x ATR=14.30 |
| Stop hit — per-position SL triggered | 2025-07-15 12:30:00 | 5689.70 | 5685.67 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-07-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:40:00 | 5727.50 | 5700.61 | 0.00 | ORB-long ORB[5635.50,5708.50] vol=1.6x ATR=17.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:45:00 | 5754.24 | 5716.49 | 0.00 | T1 1.5R @ 5754.24 |
| Target hit | 2025-07-21 11:20:00 | 5749.50 | 5758.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 5715.50 | 5744.85 | 0.00 | ORB-short ORB[5733.00,5788.00] vol=1.5x ATR=14.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 09:35:00 | 5694.07 | 5733.26 | 0.00 | T1 1.5R @ 5694.07 |
| Target hit | 2025-07-23 12:20:00 | 5699.50 | 5697.94 | 0.00 | Trail-exit close>VWAP |

### Cycle 28 — SELL (started 2025-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:30:00 | 5648.50 | 5667.93 | 0.00 | ORB-short ORB[5661.00,5699.00] vol=1.5x ATR=10.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:35:00 | 5633.15 | 5662.98 | 0.00 | T1 1.5R @ 5633.15 |
| Stop hit — per-position SL triggered | 2025-07-25 10:45:00 | 5648.50 | 5640.74 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-07-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 10:45:00 | 5593.50 | 5613.74 | 0.00 | ORB-short ORB[5606.50,5644.00] vol=2.2x ATR=15.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 11:45:00 | 5570.74 | 5598.83 | 0.00 | T1 1.5R @ 5570.74 |
| Target hit | 2025-07-28 15:20:00 | 5540.00 | 5571.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2025-07-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 11:10:00 | 5497.50 | 5524.87 | 0.00 | ORB-short ORB[5523.00,5570.00] vol=1.7x ATR=12.42 |
| Stop hit — per-position SL triggered | 2025-07-29 11:20:00 | 5509.92 | 5522.83 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:15:00 | 5050.00 | 5056.07 | 0.00 | ORB-short ORB[5070.00,5124.00] vol=2.2x ATR=13.07 |
| Stop hit — per-position SL triggered | 2025-08-06 12:00:00 | 5063.07 | 5054.93 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:40:00 | 5090.00 | 5038.80 | 0.00 | ORB-long ORB[5012.00,5065.00] vol=2.0x ATR=18.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:00:00 | 5117.33 | 5051.94 | 0.00 | T1 1.5R @ 5117.33 |
| Stop hit — per-position SL triggered | 2025-08-11 11:30:00 | 5090.00 | 5062.64 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:00:00 | 5025.00 | 5053.30 | 0.00 | ORB-short ORB[5052.00,5079.50] vol=1.5x ATR=11.36 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 5036.36 | 5048.04 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-08-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:35:00 | 5050.00 | 5067.80 | 0.00 | ORB-short ORB[5081.50,5105.50] vol=2.6x ATR=10.52 |
| Stop hit — per-position SL triggered | 2025-08-14 10:55:00 | 5060.52 | 5064.80 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 11:15:00 | 5029.50 | 5048.94 | 0.00 | ORB-short ORB[5036.00,5090.00] vol=2.0x ATR=8.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 12:10:00 | 5016.64 | 5044.18 | 0.00 | T1 1.5R @ 5016.64 |
| Stop hit — per-position SL triggered | 2025-08-18 14:50:00 | 5029.50 | 5033.03 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-08-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:55:00 | 5064.00 | 5034.43 | 0.00 | ORB-long ORB[5021.00,5051.50] vol=1.7x ATR=9.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:15:00 | 5078.84 | 5040.45 | 0.00 | T1 1.5R @ 5078.84 |
| Stop hit — per-position SL triggered | 2025-08-19 11:45:00 | 5064.00 | 5043.94 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-08-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:10:00 | 5120.00 | 5099.88 | 0.00 | ORB-long ORB[5072.50,5110.00] vol=3.8x ATR=10.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:25:00 | 5135.49 | 5111.47 | 0.00 | T1 1.5R @ 5135.49 |
| Stop hit — per-position SL triggered | 2025-08-20 10:40:00 | 5120.00 | 5114.06 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:55:00 | 5154.00 | 5133.66 | 0.00 | ORB-long ORB[5112.00,5150.00] vol=3.5x ATR=10.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 10:00:00 | 5169.27 | 5140.22 | 0.00 | T1 1.5R @ 5169.27 |
| Target hit | 2025-08-21 11:20:00 | 5161.00 | 5161.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2025-08-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:40:00 | 5097.00 | 5126.66 | 0.00 | ORB-short ORB[5127.00,5170.00] vol=3.4x ATR=10.56 |
| Stop hit — per-position SL triggered | 2025-08-22 10:55:00 | 5107.56 | 5124.00 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-08-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 11:05:00 | 5108.00 | 5075.37 | 0.00 | ORB-long ORB[5047.00,5100.00] vol=1.5x ATR=8.94 |
| Stop hit — per-position SL triggered | 2025-08-25 11:20:00 | 5099.06 | 5078.44 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-08-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:50:00 | 4947.00 | 4971.49 | 0.00 | ORB-short ORB[4981.50,5020.00] vol=2.0x ATR=11.45 |
| Stop hit — per-position SL triggered | 2025-08-29 09:55:00 | 4958.45 | 4970.14 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:40:00 | 5062.20 | 5044.29 | 0.00 | ORB-long ORB[5017.60,5044.00] vol=1.5x ATR=12.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 10:50:00 | 5081.44 | 5056.37 | 0.00 | T1 1.5R @ 5081.44 |
| Stop hit — per-position SL triggered | 2025-09-01 10:55:00 | 5062.20 | 5057.93 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:45:00 | 5188.00 | 5169.86 | 0.00 | ORB-long ORB[5148.20,5179.90] vol=1.7x ATR=12.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 10:10:00 | 5206.11 | 5181.65 | 0.00 | T1 1.5R @ 5206.11 |
| Stop hit — per-position SL triggered | 2025-09-03 10:35:00 | 5188.00 | 5185.25 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:10:00 | 5124.80 | 5155.90 | 0.00 | ORB-short ORB[5154.40,5185.20] vol=1.8x ATR=13.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 5104.39 | 5148.95 | 0.00 | T1 1.5R @ 5104.39 |
| Stop hit — per-position SL triggered | 2025-09-05 10:25:00 | 5124.80 | 5144.71 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-09-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 11:05:00 | 5226.60 | 5203.77 | 0.00 | ORB-long ORB[5157.10,5210.00] vol=6.7x ATR=8.60 |
| Stop hit — per-position SL triggered | 2025-09-11 11:25:00 | 5218.00 | 5211.20 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-09-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:45:00 | 5301.00 | 5281.10 | 0.00 | ORB-long ORB[5241.20,5270.60] vol=2.8x ATR=10.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 10:00:00 | 5316.38 | 5290.54 | 0.00 | T1 1.5R @ 5316.38 |
| Stop hit — per-position SL triggered | 2025-09-15 10:05:00 | 5301.00 | 5291.19 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 11:15:00 | 5371.90 | 5352.57 | 0.00 | ORB-long ORB[5316.60,5370.00] vol=2.6x ATR=10.60 |
| Stop hit — per-position SL triggered | 2025-09-16 12:05:00 | 5361.30 | 5356.26 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:30:00 | 5393.10 | 5380.10 | 0.00 | ORB-long ORB[5360.00,5377.00] vol=3.2x ATR=10.94 |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 5382.16 | 5387.44 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-09-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:40:00 | 5450.80 | 5421.21 | 0.00 | ORB-long ORB[5396.00,5430.00] vol=2.2x ATR=13.71 |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 5437.09 | 5439.28 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-09-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:55:00 | 5315.50 | 5345.77 | 0.00 | ORB-short ORB[5340.00,5375.40] vol=1.8x ATR=12.88 |
| Stop hit — per-position SL triggered | 2025-09-23 10:00:00 | 5328.38 | 5344.34 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-09-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:55:00 | 5151.40 | 5157.60 | 0.00 | ORB-short ORB[5155.00,5210.00] vol=6.1x ATR=13.87 |
| Stop hit — per-position SL triggered | 2025-09-26 11:35:00 | 5165.27 | 5157.02 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:00:00 | 5155.50 | 5174.12 | 0.00 | ORB-short ORB[5181.00,5203.00] vol=1.5x ATR=9.10 |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 5164.60 | 5173.04 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-10-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:50:00 | 5103.50 | 5128.01 | 0.00 | ORB-short ORB[5141.50,5169.50] vol=2.8x ATR=10.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:05:00 | 5088.04 | 5125.27 | 0.00 | T1 1.5R @ 5088.04 |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 5103.50 | 5119.06 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:30:00 | 5156.50 | 5131.76 | 0.00 | ORB-long ORB[5097.50,5140.00] vol=2.1x ATR=9.77 |
| Stop hit — per-position SL triggered | 2025-10-15 09:35:00 | 5146.73 | 5134.55 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:30:00 | 5205.50 | 5192.00 | 0.00 | ORB-long ORB[5170.00,5198.00] vol=1.6x ATR=9.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 09:35:00 | 5219.12 | 5198.55 | 0.00 | T1 1.5R @ 5219.12 |
| Target hit | 2025-10-16 10:15:00 | 5214.00 | 5223.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — BUY (started 2025-10-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:30:00 | 5256.00 | 5237.81 | 0.00 | ORB-long ORB[5213.00,5250.00] vol=4.1x ATR=14.05 |
| Stop hit — per-position SL triggered | 2025-10-20 09:35:00 | 5241.95 | 5238.32 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-10-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 11:05:00 | 5205.50 | 5191.74 | 0.00 | ORB-long ORB[5172.00,5201.50] vol=2.1x ATR=7.64 |
| Stop hit — per-position SL triggered | 2025-10-27 11:10:00 | 5197.86 | 5191.91 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-10-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:55:00 | 5211.50 | 5239.35 | 0.00 | ORB-short ORB[5226.50,5275.00] vol=1.7x ATR=8.52 |
| Stop hit — per-position SL triggered | 2025-10-28 11:05:00 | 5220.02 | 5237.92 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 5271.00 | 5235.76 | 0.00 | ORB-long ORB[5175.50,5244.00] vol=1.5x ATR=13.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:45:00 | 5291.49 | 5247.74 | 0.00 | T1 1.5R @ 5291.49 |
| Stop hit — per-position SL triggered | 2025-10-29 11:10:00 | 5271.00 | 5252.98 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:35:00 | 5186.00 | 5140.43 | 0.00 | ORB-long ORB[5098.00,5141.50] vol=2.2x ATR=13.51 |
| Stop hit — per-position SL triggered | 2025-12-10 10:45:00 | 5172.49 | 5144.78 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 09:40:00 | 5219.50 | 5199.60 | 0.00 | ORB-long ORB[5147.00,5215.00] vol=1.6x ATR=13.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 09:45:00 | 5239.58 | 5210.68 | 0.00 | T1 1.5R @ 5239.58 |
| Stop hit — per-position SL triggered | 2025-12-11 10:20:00 | 5219.50 | 5225.00 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:25:00 | 5210.50 | 5226.36 | 0.00 | ORB-short ORB[5221.50,5288.00] vol=2.4x ATR=11.98 |
| Stop hit — per-position SL triggered | 2025-12-15 10:55:00 | 5222.48 | 5217.01 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:45:00 | 5216.50 | 5227.95 | 0.00 | ORB-short ORB[5229.00,5253.00] vol=4.2x ATR=9.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 11:25:00 | 5202.59 | 5225.61 | 0.00 | T1 1.5R @ 5202.59 |
| Target hit | 2025-12-17 15:20:00 | 5165.50 | 5193.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2025-12-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:40:00 | 5262.50 | 5245.90 | 0.00 | ORB-long ORB[5204.00,5255.00] vol=1.7x ATR=10.43 |
| Stop hit — per-position SL triggered | 2025-12-24 09:45:00 | 5252.07 | 5246.48 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:30:00 | 5233.50 | 5226.65 | 0.00 | ORB-long ORB[5195.50,5231.50] vol=1.6x ATR=9.51 |
| Stop hit — per-position SL triggered | 2025-12-26 10:55:00 | 5223.99 | 5226.66 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-12-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:50:00 | 5126.50 | 5142.70 | 0.00 | ORB-short ORB[5142.00,5168.50] vol=2.9x ATR=10.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 12:20:00 | 5110.41 | 5127.84 | 0.00 | T1 1.5R @ 5110.41 |
| Stop hit — per-position SL triggered | 2025-12-30 12:40:00 | 5126.50 | 5125.83 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:50:00 | 5223.50 | 5187.37 | 0.00 | ORB-long ORB[5160.50,5181.00] vol=1.7x ATR=12.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:10:00 | 5242.37 | 5210.93 | 0.00 | T1 1.5R @ 5242.37 |
| Stop hit — per-position SL triggered | 2026-01-06 10:20:00 | 5223.50 | 5213.25 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 09:30:00 | 5301.50 | 5335.31 | 0.00 | ORB-short ORB[5302.00,5368.00] vol=1.7x ATR=17.10 |
| Stop hit — per-position SL triggered | 2026-01-08 09:35:00 | 5318.60 | 5333.63 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 10:45:00 | 4857.50 | 4875.56 | 0.00 | ORB-short ORB[4871.50,4942.00] vol=1.6x ATR=15.17 |
| Stop hit — per-position SL triggered | 2026-01-19 10:55:00 | 4872.67 | 4875.09 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:00:00 | 4747.50 | 4783.84 | 0.00 | ORB-short ORB[4802.00,4855.00] vol=2.3x ATR=14.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:05:00 | 4725.31 | 4760.68 | 0.00 | T1 1.5R @ 4725.31 |
| Target hit | 2026-01-20 15:20:00 | 4685.00 | 4726.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2026-01-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-21 11:10:00 | 4715.50 | 4678.41 | 0.00 | ORB-long ORB[4659.00,4706.00] vol=1.8x ATR=19.06 |
| Stop hit — per-position SL triggered | 2026-01-21 11:35:00 | 4696.44 | 4680.60 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-01-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:55:00 | 4721.00 | 4748.89 | 0.00 | ORB-short ORB[4741.00,4775.00] vol=2.1x ATR=9.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:05:00 | 4706.17 | 4744.64 | 0.00 | T1 1.5R @ 4706.17 |
| Target hit | 2026-01-23 14:50:00 | 4714.00 | 4710.97 | 0.00 | Trail-exit close>VWAP |

### Cycle 73 — BUY (started 2026-02-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 11:10:00 | 5760.00 | 5711.70 | 0.00 | ORB-long ORB[5675.00,5751.00] vol=1.9x ATR=16.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 11:20:00 | 5784.10 | 5723.21 | 0.00 | T1 1.5R @ 5784.10 |
| Stop hit — per-position SL triggered | 2026-02-05 11:35:00 | 5760.00 | 5730.83 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 5810.00 | 5831.56 | 0.00 | ORB-short ORB[5830.50,5861.00] vol=2.2x ATR=11.22 |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 5821.22 | 5831.02 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 5791.00 | 5816.57 | 0.00 | ORB-short ORB[5803.50,5864.50] vol=2.4x ATR=11.51 |
| Stop hit — per-position SL triggered | 2026-02-12 11:10:00 | 5802.51 | 5815.16 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:55:00 | 5804.00 | 5771.23 | 0.00 | ORB-long ORB[5750.00,5798.00] vol=3.4x ATR=15.75 |
| Stop hit — per-position SL triggered | 2026-02-13 11:00:00 | 5788.25 | 5771.73 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 5815.00 | 5794.93 | 0.00 | ORB-long ORB[5742.50,5811.50] vol=1.6x ATR=12.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:20:00 | 5834.45 | 5797.93 | 0.00 | T1 1.5R @ 5834.45 |
| Target hit | 2026-02-16 15:20:00 | 5894.00 | 5860.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 5789.50 | 5816.19 | 0.00 | ORB-short ORB[5801.00,5858.50] vol=1.7x ATR=11.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:25:00 | 5771.69 | 5809.62 | 0.00 | T1 1.5R @ 5771.69 |
| Stop hit — per-position SL triggered | 2026-02-18 11:35:00 | 5789.50 | 5808.63 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 5922.00 | 5868.92 | 0.00 | ORB-long ORB[5833.50,5896.00] vol=2.8x ATR=18.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:00:00 | 5950.10 | 5887.32 | 0.00 | T1 1.5R @ 5950.10 |
| Stop hit — per-position SL triggered | 2026-03-05 11:10:00 | 5922.00 | 5889.58 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-03-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:10:00 | 6016.50 | 6006.64 | 0.00 | ORB-long ORB[5869.00,5959.00] vol=1.9x ATR=19.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:35:00 | 6046.27 | 6011.06 | 0.00 | T1 1.5R @ 6046.27 |
| Target hit | 2026-03-06 15:20:00 | 6069.00 | 6047.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — BUY (started 2026-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:40:00 | 6392.00 | 6354.01 | 0.00 | ORB-long ORB[6287.50,6343.00] vol=2.1x ATR=21.36 |
| Stop hit — per-position SL triggered | 2026-03-18 11:35:00 | 6370.64 | 6370.16 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:25:00 | 6506.00 | 6430.44 | 0.00 | ORB-long ORB[6370.00,6465.00] vol=1.8x ATR=26.58 |
| Stop hit — per-position SL triggered | 2026-04-08 10:35:00 | 6479.42 | 6435.62 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-04-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:30:00 | 6510.00 | 6526.50 | 0.00 | ORB-short ORB[6511.50,6592.00] vol=1.5x ATR=17.20 |
| Stop hit — per-position SL triggered | 2026-04-09 10:35:00 | 6527.20 | 6526.45 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 6975.50 | 6925.86 | 0.00 | ORB-long ORB[6875.00,6949.50] vol=4.6x ATR=29.07 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 6946.43 | 6947.76 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 6806.50 | 6854.11 | 0.00 | ORB-short ORB[6814.00,6907.00] vol=2.1x ATR=22.25 |
| Stop hit — per-position SL triggered | 2026-04-16 09:50:00 | 6828.75 | 6843.99 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:10:00 | 7293.50 | 7265.43 | 0.00 | ORB-long ORB[7184.00,7287.50] vol=1.6x ATR=16.63 |
| Stop hit — per-position SL triggered | 2026-04-21 11:35:00 | 7276.87 | 7269.02 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 7417.00 | 7356.94 | 0.00 | ORB-long ORB[7325.00,7398.00] vol=2.7x ATR=26.27 |
| Stop hit — per-position SL triggered | 2026-04-27 11:00:00 | 7390.73 | 7358.26 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:15:00 | 7371.00 | 7416.77 | 0.00 | ORB-short ORB[7402.50,7490.00] vol=2.9x ATR=21.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:35:00 | 7338.59 | 7406.46 | 0.00 | T1 1.5R @ 7338.59 |
| Target hit | 2026-04-28 15:20:00 | 7285.00 | 7321.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — SELL (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 7236.50 | 7274.58 | 0.00 | ORB-short ORB[7260.50,7340.00] vol=2.1x ATR=22.00 |
| Stop hit — per-position SL triggered | 2026-04-29 11:30:00 | 7258.50 | 7258.11 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 7289.50 | 7275.51 | 0.00 | ORB-long ORB[7215.00,7278.50] vol=2.3x ATR=25.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:45:00 | 7327.91 | 7288.38 | 0.00 | T1 1.5R @ 7327.91 |
| Stop hit — per-position SL triggered | 2026-05-05 10:20:00 | 7289.50 | 7304.94 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 7086.00 | 7203.49 | 0.00 | ORB-short ORB[7183.00,7243.50] vol=5.8x ATR=32.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:15:00 | 7038.01 | 7173.41 | 0.00 | T1 1.5R @ 7038.01 |
| Stop hit — per-position SL triggered | 2026-05-07 11:20:00 | 7086.00 | 7166.93 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:30:00 | 5623.50 | 2025-05-13 09:35:00 | 5659.49 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-05-13 09:30:00 | 5623.50 | 2025-05-13 09:55:00 | 5623.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-15 10:45:00 | 5696.00 | 2025-05-15 11:55:00 | 5679.57 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-16 09:35:00 | 5809.00 | 2025-05-16 09:40:00 | 5834.67 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-05-16 09:35:00 | 5809.00 | 2025-05-16 09:50:00 | 5809.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-21 11:10:00 | 5845.00 | 2025-05-21 11:20:00 | 5827.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-23 09:30:00 | 5972.50 | 2025-05-23 09:35:00 | 5951.82 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-05-26 09:40:00 | 6031.50 | 2025-05-26 10:40:00 | 6016.42 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-27 09:30:00 | 5969.50 | 2025-05-27 09:35:00 | 5981.61 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-05-30 11:10:00 | 6031.00 | 2025-05-30 11:45:00 | 6008.06 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-30 11:10:00 | 6031.00 | 2025-05-30 12:05:00 | 6031.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-03 09:30:00 | 6045.00 | 2025-06-03 09:35:00 | 6069.59 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-06-03 09:30:00 | 6045.00 | 2025-06-03 09:45:00 | 6045.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 11:15:00 | 5981.00 | 2025-06-04 13:05:00 | 5991.80 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-09 09:45:00 | 6099.50 | 2025-06-09 09:55:00 | 6084.37 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-17 09:40:00 | 6064.00 | 2025-06-17 10:00:00 | 6050.73 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-18 10:05:00 | 6076.50 | 2025-06-18 10:25:00 | 6060.16 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-20 10:30:00 | 5965.00 | 2025-06-20 15:20:00 | 5967.00 | TARGET_HIT | 1.00 | 0.03% |
| BUY | retest1 | 2025-06-23 11:10:00 | 5950.00 | 2025-06-23 12:00:00 | 5934.30 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-24 09:50:00 | 6011.00 | 2025-06-24 10:10:00 | 5991.45 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-27 09:45:00 | 6110.00 | 2025-06-27 09:55:00 | 6136.23 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-06-27 09:45:00 | 6110.00 | 2025-06-27 10:35:00 | 6120.50 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-07-01 09:35:00 | 6142.00 | 2025-07-01 09:40:00 | 6128.68 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-02 10:50:00 | 5887.00 | 2025-07-02 11:00:00 | 5868.50 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-02 10:50:00 | 5887.00 | 2025-07-02 11:10:00 | 5887.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-07 11:05:00 | 5819.00 | 2025-07-07 12:20:00 | 5830.75 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-09 11:00:00 | 5874.50 | 2025-07-09 11:15:00 | 5892.35 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-07-09 11:00:00 | 5874.50 | 2025-07-09 11:25:00 | 5874.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 10:45:00 | 5855.00 | 2025-07-10 11:05:00 | 5836.04 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-10 10:45:00 | 5855.00 | 2025-07-10 13:05:00 | 5855.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 10:30:00 | 5785.50 | 2025-07-11 10:50:00 | 5765.60 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-11 10:30:00 | 5785.50 | 2025-07-11 10:55:00 | 5785.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-14 09:35:00 | 5714.00 | 2025-07-14 09:40:00 | 5733.68 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-15 11:10:00 | 5704.00 | 2025-07-15 12:30:00 | 5689.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-21 09:40:00 | 5727.50 | 2025-07-21 09:45:00 | 5754.24 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-07-21 09:40:00 | 5727.50 | 2025-07-21 11:20:00 | 5749.50 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-23 09:30:00 | 5715.50 | 2025-07-23 09:35:00 | 5694.07 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-23 09:30:00 | 5715.50 | 2025-07-23 12:20:00 | 5699.50 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2025-07-25 09:30:00 | 5648.50 | 2025-07-25 09:35:00 | 5633.15 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-07-25 09:30:00 | 5648.50 | 2025-07-25 10:45:00 | 5648.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-28 10:45:00 | 5593.50 | 2025-07-28 11:45:00 | 5570.74 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-28 10:45:00 | 5593.50 | 2025-07-28 15:20:00 | 5540.00 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2025-07-29 11:10:00 | 5497.50 | 2025-07-29 11:20:00 | 5509.92 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-06 11:15:00 | 5050.00 | 2025-08-06 12:00:00 | 5063.07 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-11 10:40:00 | 5090.00 | 2025-08-11 11:00:00 | 5117.33 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-08-11 10:40:00 | 5090.00 | 2025-08-11 11:30:00 | 5090.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-12 11:00:00 | 5025.00 | 2025-08-12 11:15:00 | 5036.36 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-14 10:35:00 | 5050.00 | 2025-08-14 10:55:00 | 5060.52 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-08-18 11:15:00 | 5029.50 | 2025-08-18 12:10:00 | 5016.64 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-08-18 11:15:00 | 5029.50 | 2025-08-18 14:50:00 | 5029.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 10:55:00 | 5064.00 | 2025-08-19 11:15:00 | 5078.84 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-08-19 10:55:00 | 5064.00 | 2025-08-19 11:45:00 | 5064.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 10:10:00 | 5120.00 | 2025-08-20 10:25:00 | 5135.49 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-08-20 10:10:00 | 5120.00 | 2025-08-20 10:40:00 | 5120.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-21 09:55:00 | 5154.00 | 2025-08-21 10:00:00 | 5169.27 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-08-21 09:55:00 | 5154.00 | 2025-08-21 11:20:00 | 5161.00 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-08-22 10:40:00 | 5097.00 | 2025-08-22 10:55:00 | 5107.56 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-08-25 11:05:00 | 5108.00 | 2025-08-25 11:20:00 | 5099.06 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-29 09:50:00 | 4947.00 | 2025-08-29 09:55:00 | 4958.45 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-01 10:40:00 | 5062.20 | 2025-09-01 10:50:00 | 5081.44 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-09-01 10:40:00 | 5062.20 | 2025-09-01 10:55:00 | 5062.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 09:45:00 | 5188.00 | 2025-09-03 10:10:00 | 5206.11 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-03 09:45:00 | 5188.00 | 2025-09-03 10:35:00 | 5188.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-05 10:10:00 | 5124.80 | 2025-09-05 10:15:00 | 5104.39 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-09-05 10:10:00 | 5124.80 | 2025-09-05 10:25:00 | 5124.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-11 11:05:00 | 5226.60 | 2025-09-11 11:25:00 | 5218.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-15 09:45:00 | 5301.00 | 2025-09-15 10:00:00 | 5316.38 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-09-15 09:45:00 | 5301.00 | 2025-09-15 10:05:00 | 5301.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-16 11:15:00 | 5371.90 | 2025-09-16 12:05:00 | 5361.30 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-17 09:30:00 | 5393.10 | 2025-09-17 10:15:00 | 5382.16 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-18 09:40:00 | 5450.80 | 2025-09-18 10:15:00 | 5437.09 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-23 09:55:00 | 5315.50 | 2025-09-23 10:00:00 | 5328.38 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-26 10:55:00 | 5151.40 | 2025-09-26 11:35:00 | 5165.27 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-01 11:00:00 | 5155.50 | 2025-10-01 11:15:00 | 5164.60 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-14 10:50:00 | 5103.50 | 2025-10-14 11:05:00 | 5088.04 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-14 10:50:00 | 5103.50 | 2025-10-14 11:15:00 | 5103.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 09:30:00 | 5156.50 | 2025-10-15 09:35:00 | 5146.73 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-16 09:30:00 | 5205.50 | 2025-10-16 09:35:00 | 5219.12 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-10-16 09:30:00 | 5205.50 | 2025-10-16 10:15:00 | 5214.00 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-10-20 09:30:00 | 5256.00 | 2025-10-20 09:35:00 | 5241.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-27 11:05:00 | 5205.50 | 2025-10-27 11:10:00 | 5197.86 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-10-28 10:55:00 | 5211.50 | 2025-10-28 11:05:00 | 5220.02 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-29 10:25:00 | 5271.00 | 2025-10-29 10:45:00 | 5291.49 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-10-29 10:25:00 | 5271.00 | 2025-10-29 11:10:00 | 5271.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-10 10:35:00 | 5186.00 | 2025-12-10 10:45:00 | 5172.49 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-11 09:40:00 | 5219.50 | 2025-12-11 09:45:00 | 5239.58 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-12-11 09:40:00 | 5219.50 | 2025-12-11 10:20:00 | 5219.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-15 10:25:00 | 5210.50 | 2025-12-15 10:55:00 | 5222.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-17 10:45:00 | 5216.50 | 2025-12-17 11:25:00 | 5202.59 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-17 10:45:00 | 5216.50 | 2025-12-17 15:20:00 | 5165.50 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2025-12-24 09:40:00 | 5262.50 | 2025-12-24 09:45:00 | 5252.07 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-26 10:30:00 | 5233.50 | 2025-12-26 10:55:00 | 5223.99 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-30 09:50:00 | 5126.50 | 2025-12-30 12:20:00 | 5110.41 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-30 09:50:00 | 5126.50 | 2025-12-30 12:40:00 | 5126.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-06 09:50:00 | 5223.50 | 2026-01-06 10:10:00 | 5242.37 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-01-06 09:50:00 | 5223.50 | 2026-01-06 10:20:00 | 5223.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 09:30:00 | 5301.50 | 2026-01-08 09:35:00 | 5318.60 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-19 10:45:00 | 4857.50 | 2026-01-19 10:55:00 | 4872.67 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-20 10:00:00 | 4747.50 | 2026-01-20 11:05:00 | 4725.31 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-01-20 10:00:00 | 4747.50 | 2026-01-20 15:20:00 | 4685.00 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2026-01-21 11:10:00 | 4715.50 | 2026-01-21 11:35:00 | 4696.44 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-01-23 10:55:00 | 4721.00 | 2026-01-23 11:05:00 | 4706.17 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-01-23 10:55:00 | 4721.00 | 2026-01-23 14:50:00 | 4714.00 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2026-02-05 11:10:00 | 5760.00 | 2026-02-05 11:20:00 | 5784.10 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-05 11:10:00 | 5760.00 | 2026-02-05 11:35:00 | 5760.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 11:05:00 | 5810.00 | 2026-02-11 11:15:00 | 5821.22 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-12 10:55:00 | 5791.00 | 2026-02-12 11:10:00 | 5802.51 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-13 10:55:00 | 5804.00 | 2026-02-13 11:00:00 | 5788.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-16 11:05:00 | 5815.00 | 2026-02-16 11:20:00 | 5834.45 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-16 11:05:00 | 5815.00 | 2026-02-16 15:20:00 | 5894.00 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2026-02-18 10:55:00 | 5789.50 | 2026-02-18 11:25:00 | 5771.69 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 10:55:00 | 5789.50 | 2026-02-18 11:35:00 | 5789.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:45:00 | 5922.00 | 2026-03-05 11:00:00 | 5950.10 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-05 10:45:00 | 5922.00 | 2026-03-05 11:10:00 | 5922.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 11:10:00 | 6016.50 | 2026-03-06 11:35:00 | 6046.27 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-06 11:10:00 | 6016.50 | 2026-03-06 15:20:00 | 6069.00 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2026-03-18 10:40:00 | 6392.00 | 2026-03-18 11:35:00 | 6370.64 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-08 10:25:00 | 6506.00 | 2026-04-08 10:35:00 | 6479.42 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-09 10:30:00 | 6510.00 | 2026-04-09 10:35:00 | 6527.20 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-15 09:30:00 | 6975.50 | 2026-04-15 10:50:00 | 6946.43 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-16 09:40:00 | 6806.50 | 2026-04-16 09:50:00 | 6828.75 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-21 11:10:00 | 7293.50 | 2026-04-21 11:35:00 | 7276.87 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-27 10:55:00 | 7417.00 | 2026-04-27 11:00:00 | 7390.73 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-28 10:15:00 | 7371.00 | 2026-04-28 10:35:00 | 7338.59 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-04-28 10:15:00 | 7371.00 | 2026-04-28 15:20:00 | 7285.00 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2026-04-29 10:00:00 | 7236.50 | 2026-04-29 11:30:00 | 7258.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 09:35:00 | 7289.50 | 2026-05-05 09:45:00 | 7327.91 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-05-05 09:35:00 | 7289.50 | 2026-05-05 10:20:00 | 7289.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 11:10:00 | 7086.00 | 2026-05-07 11:15:00 | 7038.01 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-05-07 11:10:00 | 7086.00 | 2026-05-07 11:20:00 | 7086.00 | STOP_HIT | 0.50 | 0.00% |
