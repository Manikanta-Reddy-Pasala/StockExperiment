# Apar Industries Ltd. (APARINDS)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-11-01 18:55:00 (27441 bars)
- **Last close:** 9982.00
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
| ENTRY1 | 46 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 10 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 36
- **Target hits / Stop hits / Partials:** 10 / 36 / 22
- **Avg / median % per leg:** 0.30% / 0.00%
- **Sum % (uncompounded):** 20.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 6 | 31.6% | 2 | 13 | 4 | 0.24% | 4.5% |
| BUY @ 2nd Alert (retest1) | 19 | 6 | 31.6% | 2 | 13 | 4 | 0.24% | 4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 26 | 53.1% | 8 | 23 | 18 | 0.33% | 16.0% |
| SELL @ 2nd Alert (retest1) | 49 | 26 | 53.1% | 8 | 23 | 18 | 0.33% | 16.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 68 | 32 | 47.1% | 10 | 36 | 22 | 0.30% | 20.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:50:00 | 2666.00 | 2722.61 | 0.00 | ORB-short ORB[2725.00,2760.05] vol=2.2x ATR=14.46 |
| Stop hit — per-position SL triggered | 2023-05-19 09:55:00 | 2680.46 | 2715.95 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 10:35:00 | 2733.05 | 2765.59 | 0.00 | ORB-short ORB[2762.55,2793.00] vol=2.4x ATR=12.49 |
| Stop hit — per-position SL triggered | 2023-05-29 10:50:00 | 2745.54 | 2762.70 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 09:35:00 | 2766.55 | 2776.67 | 0.00 | ORB-short ORB[2770.25,2784.05] vol=1.8x ATR=6.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 09:40:00 | 2757.22 | 2775.37 | 0.00 | T1 1.5R @ 2757.22 |
| Target hit | 2023-06-06 11:55:00 | 2759.45 | 2759.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — BUY (started 2023-06-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 10:40:00 | 2830.00 | 2822.33 | 0.00 | ORB-long ORB[2790.00,2828.00] vol=1.8x ATR=7.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 10:50:00 | 2840.53 | 2826.91 | 0.00 | T1 1.5R @ 2840.53 |
| Stop hit — per-position SL triggered | 2023-06-08 10:55:00 | 2830.00 | 2828.36 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 09:40:00 | 3006.95 | 3024.27 | 0.00 | ORB-short ORB[3010.35,3055.00] vol=1.6x ATR=14.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 10:05:00 | 2984.66 | 3015.31 | 0.00 | T1 1.5R @ 2984.66 |
| Target hit | 2023-06-14 15:20:00 | 2943.90 | 2971.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2023-06-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 10:10:00 | 3373.65 | 3415.90 | 0.00 | ORB-short ORB[3406.10,3456.00] vol=1.6x ATR=13.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 10:40:00 | 3352.73 | 3399.56 | 0.00 | T1 1.5R @ 3352.73 |
| Stop hit — per-position SL triggered | 2023-06-28 10:50:00 | 3373.65 | 3396.92 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-07-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 09:40:00 | 3759.35 | 3731.94 | 0.00 | ORB-long ORB[3701.05,3749.95] vol=1.6x ATR=19.85 |
| Stop hit — per-position SL triggered | 2023-07-20 10:35:00 | 3739.50 | 3746.16 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-08-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 11:00:00 | 3651.65 | 3679.29 | 0.00 | ORB-short ORB[3655.50,3709.75] vol=4.6x ATR=15.24 |
| Stop hit — per-position SL triggered | 2023-08-04 11:15:00 | 3666.89 | 3678.14 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-08-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 09:40:00 | 4020.80 | 3983.61 | 0.00 | ORB-long ORB[3921.95,3969.95] vol=2.0x ATR=19.34 |
| Stop hit — per-position SL triggered | 2023-08-10 09:45:00 | 4001.46 | 3991.20 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 10:15:00 | 5028.25 | 4992.27 | 0.00 | ORB-long ORB[4964.95,5013.65] vol=2.4x ATR=15.20 |
| Stop hit — per-position SL triggered | 2023-08-31 11:15:00 | 5013.05 | 5009.71 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-09-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 10:50:00 | 4935.05 | 4995.09 | 0.00 | ORB-short ORB[4972.00,5024.65] vol=4.4x ATR=15.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 10:55:00 | 4911.94 | 4978.86 | 0.00 | T1 1.5R @ 4911.94 |
| Stop hit — per-position SL triggered | 2023-09-01 13:05:00 | 4935.05 | 4946.41 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 09:40:00 | 4936.45 | 4951.94 | 0.00 | ORB-short ORB[4941.15,4984.00] vol=2.6x ATR=17.07 |
| Stop hit — per-position SL triggered | 2023-09-05 09:45:00 | 4953.52 | 4951.52 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 09:45:00 | 5578.65 | 5496.45 | 0.00 | ORB-long ORB[5445.00,5512.50] vol=4.6x ATR=33.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-26 09:50:00 | 5629.60 | 5531.23 | 0.00 | T1 1.5R @ 5629.60 |
| Target hit | 2023-09-26 15:20:00 | 5780.15 | 5697.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2023-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:30:00 | 5352.00 | 5318.08 | 0.00 | ORB-long ORB[5264.10,5342.65] vol=1.8x ATR=27.17 |
| Stop hit — per-position SL triggered | 2023-10-09 09:35:00 | 5324.83 | 5322.10 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-10-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 10:10:00 | 5447.00 | 5411.50 | 0.00 | ORB-long ORB[5362.05,5438.50] vol=1.5x ATR=22.39 |
| Stop hit — per-position SL triggered | 2023-10-12 10:20:00 | 5424.61 | 5413.56 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-10-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:50:00 | 5405.05 | 5444.52 | 0.00 | ORB-short ORB[5431.00,5507.65] vol=1.6x ATR=12.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 11:05:00 | 5386.16 | 5438.02 | 0.00 | T1 1.5R @ 5386.16 |
| Stop hit — per-position SL triggered | 2023-10-13 11:30:00 | 5405.05 | 5434.06 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-17 11:15:00 | 5466.35 | 5495.56 | 0.00 | ORB-short ORB[5480.25,5549.95] vol=2.0x ATR=10.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 11:25:00 | 5450.12 | 5488.55 | 0.00 | T1 1.5R @ 5450.12 |
| Stop hit — per-position SL triggered | 2023-10-17 11:55:00 | 5466.35 | 5479.79 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-10-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 11:10:00 | 5329.75 | 5387.61 | 0.00 | ORB-short ORB[5380.15,5449.00] vol=5.0x ATR=17.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:30:00 | 5303.79 | 5368.46 | 0.00 | T1 1.5R @ 5303.79 |
| Target hit | 2023-10-18 15:20:00 | 5290.00 | 5311.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2023-10-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 10:40:00 | 5228.50 | 5272.40 | 0.00 | ORB-short ORB[5260.00,5319.95] vol=2.2x ATR=18.70 |
| Stop hit — per-position SL triggered | 2023-10-20 11:00:00 | 5247.20 | 5269.62 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-10-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 11:00:00 | 5204.90 | 5145.42 | 0.00 | ORB-long ORB[5055.60,5130.00] vol=2.6x ATR=21.55 |
| Stop hit — per-position SL triggered | 2023-10-31 11:15:00 | 5183.35 | 5155.05 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-11-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 09:50:00 | 5067.05 | 5116.37 | 0.00 | ORB-short ORB[5099.05,5169.95] vol=2.3x ATR=22.14 |
| Stop hit — per-position SL triggered | 2023-11-06 11:35:00 | 5089.19 | 5095.18 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-11-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 09:40:00 | 5111.20 | 5133.72 | 0.00 | ORB-short ORB[5120.95,5156.00] vol=2.5x ATR=14.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 11:30:00 | 5089.77 | 5113.61 | 0.00 | T1 1.5R @ 5089.77 |
| Stop hit — per-position SL triggered | 2023-11-07 13:10:00 | 5111.20 | 5107.50 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-11-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 11:05:00 | 5600.00 | 5629.83 | 0.00 | ORB-short ORB[5608.05,5690.00] vol=1.5x ATR=16.87 |
| Stop hit — per-position SL triggered | 2023-11-24 11:30:00 | 5616.87 | 5628.17 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-11-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:55:00 | 5458.80 | 5487.51 | 0.00 | ORB-short ORB[5466.00,5539.00] vol=1.7x ATR=22.10 |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 5480.90 | 5485.42 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-12-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 11:00:00 | 5472.00 | 5511.02 | 0.00 | ORB-short ORB[5514.55,5580.00] vol=1.9x ATR=15.36 |
| Stop hit — per-position SL triggered | 2023-12-06 11:15:00 | 5487.36 | 5509.48 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-12-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:05:00 | 5370.00 | 5397.03 | 0.00 | ORB-short ORB[5385.00,5448.90] vol=2.0x ATR=14.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 10:50:00 | 5348.28 | 5384.32 | 0.00 | T1 1.5R @ 5348.28 |
| Stop hit — per-position SL triggered | 2023-12-08 11:50:00 | 5370.00 | 5365.08 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-12-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 10:05:00 | 5386.95 | 5350.57 | 0.00 | ORB-long ORB[5293.05,5357.90] vol=2.2x ATR=19.90 |
| Stop hit — per-position SL triggered | 2023-12-14 10:10:00 | 5367.05 | 5385.72 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-12-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:20:00 | 5427.60 | 5390.19 | 0.00 | ORB-long ORB[5360.00,5393.90] vol=5.3x ATR=18.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 10:40:00 | 5454.68 | 5411.70 | 0.00 | T1 1.5R @ 5454.68 |
| Stop hit — per-position SL triggered | 2023-12-20 11:05:00 | 5427.60 | 5417.01 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-12-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 10:15:00 | 5808.00 | 5828.57 | 0.00 | ORB-short ORB[5825.05,5903.85] vol=2.7x ATR=32.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 10:40:00 | 5758.55 | 5824.98 | 0.00 | T1 1.5R @ 5758.55 |
| Stop hit — per-position SL triggered | 2023-12-28 11:30:00 | 5808.00 | 5820.39 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 5993.40 | 6036.22 | 0.00 | ORB-short ORB[6005.00,6059.45] vol=2.5x ATR=24.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:00:00 | 5957.18 | 6026.48 | 0.00 | T1 1.5R @ 5957.18 |
| Stop hit — per-position SL triggered | 2024-01-02 10:05:00 | 5993.40 | 6024.62 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 09:30:00 | 5827.25 | 5865.18 | 0.00 | ORB-short ORB[5855.65,5917.00] vol=1.9x ATR=23.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 09:55:00 | 5791.48 | 5825.68 | 0.00 | T1 1.5R @ 5791.48 |
| Target hit | 2024-01-05 12:30:00 | 5800.00 | 5797.60 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2024-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 09:35:00 | 5452.00 | 5407.57 | 0.00 | ORB-long ORB[5353.60,5417.40] vol=2.3x ATR=18.96 |
| Stop hit — per-position SL triggered | 2024-01-16 09:40:00 | 5433.04 | 5411.76 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-01-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 10:30:00 | 5384.95 | 5417.98 | 0.00 | ORB-short ORB[5400.05,5458.00] vol=2.0x ATR=18.93 |
| Stop hit — per-position SL triggered | 2024-01-19 10:45:00 | 5403.88 | 5414.58 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-01-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 09:30:00 | 5459.40 | 5438.97 | 0.00 | ORB-long ORB[5389.05,5456.15] vol=1.8x ATR=23.75 |
| Stop hit — per-position SL triggered | 2024-01-23 09:35:00 | 5435.65 | 5436.28 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 10:55:00 | 5908.05 | 5842.00 | 0.00 | ORB-long ORB[5740.50,5829.00] vol=2.1x ATR=34.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 11:10:00 | 5959.47 | 5870.06 | 0.00 | T1 1.5R @ 5959.47 |
| Target hit | 2024-01-30 13:05:00 | 6066.00 | 6100.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — SELL (started 2024-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 10:15:00 | 6232.10 | 6276.98 | 0.00 | ORB-short ORB[6271.00,6340.00] vol=4.2x ATR=25.30 |
| Stop hit — per-position SL triggered | 2024-02-05 10:20:00 | 6257.40 | 6276.37 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 11:10:00 | 6300.05 | 6328.02 | 0.00 | ORB-short ORB[6339.50,6407.10] vol=1.8x ATR=15.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 12:25:00 | 6276.87 | 6315.42 | 0.00 | T1 1.5R @ 6276.87 |
| Stop hit — per-position SL triggered | 2024-02-23 15:00:00 | 6300.05 | 6281.05 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-02-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:55:00 | 6220.25 | 6263.04 | 0.00 | ORB-short ORB[6239.90,6300.00] vol=1.7x ATR=20.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 11:00:00 | 6189.43 | 6254.38 | 0.00 | T1 1.5R @ 6189.43 |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 6220.25 | 6248.72 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 09:55:00 | 6209.20 | 6236.66 | 0.00 | ORB-short ORB[6220.00,6294.95] vol=2.0x ATR=21.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 10:15:00 | 6177.16 | 6223.55 | 0.00 | T1 1.5R @ 6177.16 |
| Target hit | 2024-03-05 15:20:00 | 6094.80 | 6152.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2024-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 6088.25 | 6152.28 | 0.00 | ORB-short ORB[6130.00,6193.30] vol=2.5x ATR=31.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:35:00 | 6041.09 | 6114.75 | 0.00 | T1 1.5R @ 6041.09 |
| Target hit | 2024-03-06 12:20:00 | 5924.90 | 5921.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2024-03-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 10:20:00 | 6043.40 | 6090.30 | 0.00 | ORB-short ORB[6070.00,6144.00] vol=1.5x ATR=23.48 |
| Stop hit — per-position SL triggered | 2024-03-11 10:40:00 | 6066.88 | 6084.73 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-09 10:15:00 | 7292.50 | 7354.61 | 0.00 | ORB-short ORB[7325.00,7423.80] vol=1.5x ATR=32.67 |
| Stop hit — per-position SL triggered | 2024-04-09 10:40:00 | 7325.17 | 7351.20 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 09:40:00 | 7178.95 | 7217.42 | 0.00 | ORB-short ORB[7184.00,7275.20] vol=2.2x ATR=28.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 10:10:00 | 7135.61 | 7196.73 | 0.00 | T1 1.5R @ 7135.61 |
| Target hit | 2024-04-10 12:50:00 | 7070.00 | 7061.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — BUY (started 2024-04-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 09:30:00 | 7088.95 | 7011.43 | 0.00 | ORB-long ORB[6962.25,7018.00] vol=2.8x ATR=26.46 |
| Stop hit — per-position SL triggered | 2024-04-18 09:35:00 | 7062.49 | 7019.55 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-05-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 10:50:00 | 7932.50 | 7899.91 | 0.00 | ORB-long ORB[7830.05,7930.00] vol=1.6x ATR=19.49 |
| Stop hit — per-position SL triggered | 2024-05-02 11:00:00 | 7913.01 | 7901.19 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-05-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:10:00 | 7735.40 | 7794.78 | 0.00 | ORB-short ORB[7800.20,7869.95] vol=3.4x ATR=29.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:20:00 | 7690.47 | 7768.46 | 0.00 | T1 1.5R @ 7690.47 |
| Target hit | 2024-05-09 15:00:00 | 7582.95 | 7577.82 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-19 09:50:00 | 2666.00 | 2023-05-19 09:55:00 | 2680.46 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2023-05-29 10:35:00 | 2733.05 | 2023-05-29 10:50:00 | 2745.54 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-06-06 09:35:00 | 2766.55 | 2023-06-06 09:40:00 | 2757.22 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-06-06 09:35:00 | 2766.55 | 2023-06-06 11:55:00 | 2759.45 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2023-06-08 10:40:00 | 2830.00 | 2023-06-08 10:50:00 | 2840.53 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-06-08 10:40:00 | 2830.00 | 2023-06-08 10:55:00 | 2830.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-14 09:40:00 | 3006.95 | 2023-06-14 10:05:00 | 2984.66 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2023-06-14 09:40:00 | 3006.95 | 2023-06-14 15:20:00 | 2943.90 | TARGET_HIT | 0.50 | 2.10% |
| SELL | retest1 | 2023-06-28 10:10:00 | 3373.65 | 2023-06-28 10:40:00 | 3352.73 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2023-06-28 10:10:00 | 3373.65 | 2023-06-28 10:50:00 | 3373.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-20 09:40:00 | 3759.35 | 2023-07-20 10:35:00 | 3739.50 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2023-08-04 11:00:00 | 3651.65 | 2023-08-04 11:15:00 | 3666.89 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-08-10 09:40:00 | 4020.80 | 2023-08-10 09:45:00 | 4001.46 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2023-08-31 10:15:00 | 5028.25 | 2023-08-31 11:15:00 | 5013.05 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-09-01 10:50:00 | 4935.05 | 2023-09-01 10:55:00 | 4911.94 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-09-01 10:50:00 | 4935.05 | 2023-09-01 13:05:00 | 4935.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-05 09:40:00 | 4936.45 | 2023-09-05 09:45:00 | 4953.52 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-09-26 09:45:00 | 5578.65 | 2023-09-26 09:50:00 | 5629.60 | PARTIAL | 0.50 | 0.91% |
| BUY | retest1 | 2023-09-26 09:45:00 | 5578.65 | 2023-09-26 15:20:00 | 5780.15 | TARGET_HIT | 0.50 | 3.61% |
| BUY | retest1 | 2023-10-09 09:30:00 | 5352.00 | 2023-10-09 09:35:00 | 5324.83 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2023-10-12 10:10:00 | 5447.00 | 2023-10-12 10:20:00 | 5424.61 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-10-13 10:50:00 | 5405.05 | 2023-10-13 11:05:00 | 5386.16 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-10-13 10:50:00 | 5405.05 | 2023-10-13 11:30:00 | 5405.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-17 11:15:00 | 5466.35 | 2023-10-17 11:25:00 | 5450.12 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-10-17 11:15:00 | 5466.35 | 2023-10-17 11:55:00 | 5466.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-18 11:10:00 | 5329.75 | 2023-10-18 11:30:00 | 5303.79 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-10-18 11:10:00 | 5329.75 | 2023-10-18 15:20:00 | 5290.00 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2023-10-20 10:40:00 | 5228.50 | 2023-10-20 11:00:00 | 5247.20 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-10-31 11:00:00 | 5204.90 | 2023-10-31 11:15:00 | 5183.35 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-11-06 09:50:00 | 5067.05 | 2023-11-06 11:35:00 | 5089.19 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-11-07 09:40:00 | 5111.20 | 2023-11-07 11:30:00 | 5089.77 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-11-07 09:40:00 | 5111.20 | 2023-11-07 13:10:00 | 5111.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-24 11:05:00 | 5600.00 | 2023-11-24 11:30:00 | 5616.87 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-11-30 09:55:00 | 5458.80 | 2023-11-30 10:15:00 | 5480.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-12-06 11:00:00 | 5472.00 | 2023-12-06 11:15:00 | 5487.36 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-08 10:05:00 | 5370.00 | 2023-12-08 10:50:00 | 5348.28 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-12-08 10:05:00 | 5370.00 | 2023-12-08 11:50:00 | 5370.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-14 10:05:00 | 5386.95 | 2023-12-14 10:10:00 | 5367.05 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-12-20 10:20:00 | 5427.60 | 2023-12-20 10:40:00 | 5454.68 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-12-20 10:20:00 | 5427.60 | 2023-12-20 11:05:00 | 5427.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-28 10:15:00 | 5808.00 | 2023-12-28 10:40:00 | 5758.55 | PARTIAL | 0.50 | 0.85% |
| SELL | retest1 | 2023-12-28 10:15:00 | 5808.00 | 2023-12-28 11:30:00 | 5808.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 09:55:00 | 5993.40 | 2024-01-02 10:00:00 | 5957.18 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-01-02 09:55:00 | 5993.40 | 2024-01-02 10:05:00 | 5993.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-05 09:30:00 | 5827.25 | 2024-01-05 09:55:00 | 5791.48 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-01-05 09:30:00 | 5827.25 | 2024-01-05 12:30:00 | 5800.00 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2024-01-16 09:35:00 | 5452.00 | 2024-01-16 09:40:00 | 5433.04 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-01-19 10:30:00 | 5384.95 | 2024-01-19 10:45:00 | 5403.88 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-01-23 09:30:00 | 5459.40 | 2024-01-23 09:35:00 | 5435.65 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-01-30 10:55:00 | 5908.05 | 2024-01-30 11:10:00 | 5959.47 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-01-30 10:55:00 | 5908.05 | 2024-01-30 13:05:00 | 6066.00 | TARGET_HIT | 0.50 | 2.67% |
| SELL | retest1 | 2024-02-05 10:15:00 | 6232.10 | 2024-02-05 10:20:00 | 6257.40 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-02-23 11:10:00 | 6300.05 | 2024-02-23 12:25:00 | 6276.87 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-02-23 11:10:00 | 6300.05 | 2024-02-23 15:00:00 | 6300.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-28 10:55:00 | 6220.25 | 2024-02-28 11:00:00 | 6189.43 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-02-28 10:55:00 | 6220.25 | 2024-02-28 11:15:00 | 6220.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-05 09:55:00 | 6209.20 | 2024-03-05 10:15:00 | 6177.16 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-03-05 09:55:00 | 6209.20 | 2024-03-05 15:20:00 | 6094.80 | TARGET_HIT | 0.50 | 1.84% |
| SELL | retest1 | 2024-03-06 09:30:00 | 6088.25 | 2024-03-06 09:35:00 | 6041.09 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-03-06 09:30:00 | 6088.25 | 2024-03-06 12:20:00 | 5924.90 | TARGET_HIT | 0.50 | 2.68% |
| SELL | retest1 | 2024-03-11 10:20:00 | 6043.40 | 2024-03-11 10:40:00 | 6066.88 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-04-09 10:15:00 | 7292.50 | 2024-04-09 10:40:00 | 7325.17 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-04-10 09:40:00 | 7178.95 | 2024-04-10 10:10:00 | 7135.61 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-04-10 09:40:00 | 7178.95 | 2024-04-10 12:50:00 | 7070.00 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2024-04-18 09:30:00 | 7088.95 | 2024-04-18 09:35:00 | 7062.49 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-05-02 10:50:00 | 7932.50 | 2024-05-02 11:00:00 | 7913.01 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-09 10:10:00 | 7735.40 | 2024-05-09 10:20:00 | 7690.47 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-05-09 10:10:00 | 7735.40 | 2024-05-09 15:00:00 | 7582.95 | TARGET_HIT | 0.50 | 1.97% |
