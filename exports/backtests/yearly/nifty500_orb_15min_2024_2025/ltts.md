# L&T Technology Services Ltd. (LTTS)

## Backtest Summary

- **Window:** 2024-07-10 09:15:00 → 2026-05-08 15:25:00 (32350 bars)
- **Last close:** 3801.60
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
| ENTRY1 | 66 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 11 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 96 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 55
- **Target hits / Stop hits / Partials:** 11 / 55 / 30
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 14.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 19 | 38.8% | 5 | 30 | 14 | 0.03% | 1.4% |
| BUY @ 2nd Alert (retest1) | 49 | 19 | 38.8% | 5 | 30 | 14 | 0.03% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 22 | 46.8% | 6 | 25 | 16 | 0.27% | 12.6% |
| SELL @ 2nd Alert (retest1) | 47 | 22 | 46.8% | 6 | 25 | 16 | 0.27% | 12.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 96 | 41 | 42.7% | 11 | 55 | 30 | 0.15% | 14.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:25:00 | 5011.65 | 5063.30 | 0.00 | ORB-short ORB[5050.30,5098.50] vol=1.7x ATR=25.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 4973.60 | 5046.65 | 0.00 | T1 1.5R @ 4973.60 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 5011.65 | 5041.90 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 09:35:00 | 5020.00 | 5039.26 | 0.00 | ORB-short ORB[5030.95,5087.75] vol=2.3x ATR=17.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 11:25:00 | 4994.10 | 5027.98 | 0.00 | T1 1.5R @ 4994.10 |
| Target hit | 2024-07-11 14:35:00 | 4976.00 | 4971.47 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2024-07-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:55:00 | 5038.95 | 5010.21 | 0.00 | ORB-long ORB[4967.20,5024.95] vol=1.7x ATR=16.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:20:00 | 5064.00 | 5026.41 | 0.00 | T1 1.5R @ 5064.00 |
| Stop hit — per-position SL triggered | 2024-07-12 10:30:00 | 5038.95 | 5028.48 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:30:00 | 5187.45 | 5160.85 | 0.00 | ORB-long ORB[5111.00,5149.80] vol=4.7x ATR=21.57 |
| Stop hit — per-position SL triggered | 2024-07-24 09:40:00 | 5165.88 | 5171.66 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:35:00 | 5202.65 | 5222.18 | 0.00 | ORB-short ORB[5217.40,5259.45] vol=1.5x ATR=12.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 09:45:00 | 5184.39 | 5216.26 | 0.00 | T1 1.5R @ 5184.39 |
| Stop hit — per-position SL triggered | 2024-07-29 11:15:00 | 5202.65 | 5197.18 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-07-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:05:00 | 5225.55 | 5205.64 | 0.00 | ORB-long ORB[5175.25,5215.45] vol=2.4x ATR=10.43 |
| Stop hit — per-position SL triggered | 2024-07-31 10:10:00 | 5215.12 | 5209.57 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-08-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:35:00 | 5201.10 | 5228.10 | 0.00 | ORB-short ORB[5229.00,5267.85] vol=3.8x ATR=13.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:50:00 | 5181.10 | 5222.03 | 0.00 | T1 1.5R @ 5181.10 |
| Stop hit — per-position SL triggered | 2024-08-01 11:10:00 | 5201.10 | 5220.97 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-08-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 09:35:00 | 5045.00 | 5072.78 | 0.00 | ORB-short ORB[5051.05,5100.00] vol=2.0x ATR=16.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 10:00:00 | 5019.86 | 5056.63 | 0.00 | T1 1.5R @ 5019.86 |
| Stop hit — per-position SL triggered | 2024-08-02 10:10:00 | 5045.00 | 5054.79 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:05:00 | 4900.25 | 4927.13 | 0.00 | ORB-short ORB[4914.00,4961.00] vol=2.3x ATR=13.08 |
| Stop hit — per-position SL triggered | 2024-08-08 10:10:00 | 4913.33 | 4925.93 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-08-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:40:00 | 4883.65 | 4842.68 | 0.00 | ORB-long ORB[4821.55,4860.00] vol=1.8x ATR=13.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 12:05:00 | 4903.29 | 4860.24 | 0.00 | T1 1.5R @ 4903.29 |
| Target hit | 2024-08-14 15:20:00 | 4913.45 | 4896.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:50:00 | 5255.80 | 5244.47 | 0.00 | ORB-long ORB[5210.15,5250.65] vol=2.1x ATR=17.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:05:00 | 5282.48 | 5248.25 | 0.00 | T1 1.5R @ 5282.48 |
| Stop hit — per-position SL triggered | 2024-08-19 10:20:00 | 5255.80 | 5249.47 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-08-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:10:00 | 5503.95 | 5483.30 | 0.00 | ORB-long ORB[5440.00,5500.00] vol=2.4x ATR=14.60 |
| Stop hit — per-position SL triggered | 2024-08-22 10:20:00 | 5489.35 | 5486.09 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:05:00 | 5511.00 | 5538.86 | 0.00 | ORB-short ORB[5520.00,5574.65] vol=2.8x ATR=17.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 12:00:00 | 5484.11 | 5523.78 | 0.00 | T1 1.5R @ 5484.11 |
| Stop hit — per-position SL triggered | 2024-08-26 14:25:00 | 5511.00 | 5515.31 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:45:00 | 5492.20 | 5505.44 | 0.00 | ORB-short ORB[5495.65,5530.00] vol=1.6x ATR=11.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:55:00 | 5475.36 | 5501.68 | 0.00 | T1 1.5R @ 5475.36 |
| Stop hit — per-position SL triggered | 2024-08-27 13:05:00 | 5492.20 | 5491.95 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:45:00 | 5716.10 | 5670.22 | 0.00 | ORB-long ORB[5628.45,5688.00] vol=1.7x ATR=20.97 |
| Stop hit — per-position SL triggered | 2024-08-29 10:25:00 | 5695.13 | 5686.94 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-09-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 09:35:00 | 5783.00 | 5753.15 | 0.00 | ORB-long ORB[5724.10,5760.00] vol=1.9x ATR=16.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 10:00:00 | 5807.18 | 5766.68 | 0.00 | T1 1.5R @ 5807.18 |
| Stop hit — per-position SL triggered | 2024-09-02 10:05:00 | 5783.00 | 5767.93 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:45:00 | 5767.75 | 5749.72 | 0.00 | ORB-long ORB[5707.55,5760.00] vol=2.7x ATR=14.98 |
| Stop hit — per-position SL triggered | 2024-09-03 10:10:00 | 5752.77 | 5754.34 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 09:35:00 | 5655.55 | 5687.55 | 0.00 | ORB-short ORB[5675.00,5740.00] vol=1.9x ATR=13.64 |
| Stop hit — per-position SL triggered | 2024-09-05 09:50:00 | 5669.19 | 5679.84 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:20:00 | 5601.55 | 5635.95 | 0.00 | ORB-short ORB[5610.00,5674.40] vol=2.0x ATR=17.30 |
| Stop hit — per-position SL triggered | 2024-09-10 10:25:00 | 5618.85 | 5635.01 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-09-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:45:00 | 5740.00 | 5706.43 | 0.00 | ORB-long ORB[5691.50,5739.95] vol=2.6x ATR=14.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 11:00:00 | 5761.93 | 5720.53 | 0.00 | T1 1.5R @ 5761.93 |
| Stop hit — per-position SL triggered | 2024-09-13 11:20:00 | 5740.00 | 5728.59 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 09:55:00 | 5727.30 | 5762.97 | 0.00 | ORB-short ORB[5748.05,5822.90] vol=1.8x ATR=14.76 |
| Stop hit — per-position SL triggered | 2024-09-16 10:50:00 | 5742.06 | 5754.36 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:35:00 | 5756.95 | 5710.04 | 0.00 | ORB-long ORB[5673.55,5717.00] vol=2.2x ATR=15.01 |
| Stop hit — per-position SL triggered | 2024-09-17 09:40:00 | 5741.94 | 5715.33 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:55:00 | 5420.95 | 5451.72 | 0.00 | ORB-short ORB[5451.85,5489.95] vol=2.6x ATR=12.96 |
| Stop hit — per-position SL triggered | 2024-09-23 11:15:00 | 5433.91 | 5446.83 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:30:00 | 5456.00 | 5475.18 | 0.00 | ORB-short ORB[5460.00,5501.60] vol=1.6x ATR=10.90 |
| Stop hit — per-position SL triggered | 2024-09-24 09:45:00 | 5466.90 | 5468.93 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:35:00 | 5277.70 | 5293.70 | 0.00 | ORB-short ORB[5286.00,5321.35] vol=3.5x ATR=16.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:40:00 | 5253.52 | 5285.56 | 0.00 | T1 1.5R @ 5253.52 |
| Target hit | 2024-10-03 15:20:00 | 5099.55 | 5151.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2024-10-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:35:00 | 5184.05 | 5137.42 | 0.00 | ORB-long ORB[5081.45,5140.00] vol=1.7x ATR=19.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 10:55:00 | 5213.69 | 5150.21 | 0.00 | T1 1.5R @ 5213.69 |
| Stop hit — per-position SL triggered | 2024-10-04 11:55:00 | 5184.05 | 5176.64 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:15:00 | 5358.80 | 5326.49 | 0.00 | ORB-long ORB[5290.00,5335.00] vol=3.8x ATR=15.44 |
| Stop hit — per-position SL triggered | 2024-10-16 10:25:00 | 5343.36 | 5330.91 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:30:00 | 5265.65 | 5233.23 | 0.00 | ORB-long ORB[5197.55,5247.80] vol=1.7x ATR=15.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:50:00 | 5288.96 | 5267.24 | 0.00 | T1 1.5R @ 5288.96 |
| Target hit | 2024-10-30 10:20:00 | 5272.00 | 5273.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2024-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:30:00 | 5062.30 | 5041.30 | 0.00 | ORB-long ORB[5009.00,5058.40] vol=4.4x ATR=18.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:45:00 | 5090.14 | 5047.91 | 0.00 | T1 1.5R @ 5090.14 |
| Stop hit — per-position SL triggered | 2024-11-06 10:10:00 | 5062.30 | 5057.65 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-11-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:30:00 | 5193.60 | 5167.83 | 0.00 | ORB-long ORB[5140.25,5191.90] vol=2.1x ATR=16.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 09:40:00 | 5218.61 | 5183.80 | 0.00 | T1 1.5R @ 5218.61 |
| Target hit | 2024-11-08 10:50:00 | 5203.80 | 5208.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2024-11-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 10:30:00 | 5225.40 | 5244.69 | 0.00 | ORB-short ORB[5261.90,5331.50] vol=1.7x ATR=25.20 |
| Stop hit — per-position SL triggered | 2024-11-13 10:50:00 | 5250.60 | 5244.82 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-11-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:05:00 | 5180.75 | 5137.87 | 0.00 | ORB-long ORB[5088.10,5130.35] vol=1.6x ATR=13.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:10:00 | 5201.18 | 5143.26 | 0.00 | T1 1.5R @ 5201.18 |
| Target hit | 2024-11-19 12:05:00 | 5185.30 | 5188.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2024-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:35:00 | 5082.00 | 5112.47 | 0.00 | ORB-short ORB[5086.30,5160.00] vol=2.2x ATR=19.85 |
| Stop hit — per-position SL triggered | 2024-11-21 09:45:00 | 5101.85 | 5108.49 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:00:00 | 5238.00 | 5209.61 | 0.00 | ORB-long ORB[5165.70,5213.40] vol=1.7x ATR=13.68 |
| Stop hit — per-position SL triggered | 2024-11-22 10:10:00 | 5224.32 | 5214.05 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-11-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 11:00:00 | 5431.00 | 5370.88 | 0.00 | ORB-long ORB[5323.05,5400.00] vol=4.8x ATR=15.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 11:20:00 | 5454.44 | 5385.82 | 0.00 | T1 1.5R @ 5454.44 |
| Stop hit — per-position SL triggered | 2024-11-25 11:40:00 | 5431.00 | 5394.48 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 09:35:00 | 5539.00 | 5512.98 | 0.00 | ORB-long ORB[5469.15,5525.00] vol=1.7x ATR=17.63 |
| Stop hit — per-position SL triggered | 2024-11-26 09:55:00 | 5521.37 | 5520.87 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 5354.35 | 5378.35 | 0.00 | ORB-short ORB[5361.75,5440.00] vol=1.9x ATR=15.86 |
| Stop hit — per-position SL triggered | 2024-11-28 09:35:00 | 5370.21 | 5376.31 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 5369.05 | 5354.86 | 0.00 | ORB-long ORB[5325.00,5365.35] vol=2.5x ATR=10.88 |
| Stop hit — per-position SL triggered | 2024-12-04 09:40:00 | 5358.17 | 5359.29 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:10:00 | 5400.00 | 5378.00 | 0.00 | ORB-long ORB[5317.95,5392.15] vol=5.2x ATR=15.25 |
| Stop hit — per-position SL triggered | 2024-12-10 10:35:00 | 5384.75 | 5381.70 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 11:15:00 | 5372.00 | 5388.26 | 0.00 | ORB-short ORB[5375.15,5407.35] vol=2.6x ATR=9.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:25:00 | 5357.05 | 5384.44 | 0.00 | T1 1.5R @ 5357.05 |
| Stop hit — per-position SL triggered | 2024-12-11 15:00:00 | 5372.00 | 5369.31 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:15:00 | 5424.80 | 5407.56 | 0.00 | ORB-long ORB[5380.45,5423.70] vol=1.6x ATR=13.04 |
| Stop hit — per-position SL triggered | 2024-12-12 10:25:00 | 5411.76 | 5410.41 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:30:00 | 5363.70 | 5378.14 | 0.00 | ORB-short ORB[5365.00,5407.65] vol=1.6x ATR=13.07 |
| Stop hit — per-position SL triggered | 2024-12-13 10:05:00 | 5376.77 | 5373.01 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:35:00 | 5335.50 | 5354.90 | 0.00 | ORB-short ORB[5340.05,5389.00] vol=2.4x ATR=11.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:00:00 | 5317.96 | 5341.46 | 0.00 | T1 1.5R @ 5317.96 |
| Target hit | 2024-12-17 15:20:00 | 5223.10 | 5275.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-12-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:50:00 | 5196.00 | 5222.60 | 0.00 | ORB-short ORB[5207.80,5247.50] vol=5.6x ATR=16.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:50:00 | 5171.74 | 5210.83 | 0.00 | T1 1.5R @ 5171.74 |
| Stop hit — per-position SL triggered | 2024-12-18 13:05:00 | 5196.00 | 5210.44 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 4702.35 | 4717.38 | 0.00 | ORB-short ORB[4702.75,4751.00] vol=1.8x ATR=11.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:45:00 | 4685.11 | 4706.04 | 0.00 | T1 1.5R @ 4685.11 |
| Stop hit — per-position SL triggered | 2024-12-26 10:35:00 | 4702.35 | 4693.34 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:45:00 | 4739.75 | 4722.56 | 0.00 | ORB-long ORB[4697.20,4729.95] vol=1.5x ATR=13.65 |
| Stop hit — per-position SL triggered | 2024-12-30 11:00:00 | 4726.10 | 4723.59 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 11:10:00 | 4729.85 | 4747.68 | 0.00 | ORB-short ORB[4733.35,4762.45] vol=1.7x ATR=11.70 |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 4741.55 | 4746.89 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:40:00 | 4730.95 | 4710.40 | 0.00 | ORB-long ORB[4680.00,4730.00] vol=1.8x ATR=12.33 |
| Stop hit — per-position SL triggered | 2025-01-02 10:20:00 | 4718.62 | 4717.90 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 4750.75 | 4796.06 | 0.00 | ORB-short ORB[4783.20,4811.00] vol=3.0x ATR=13.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:45:00 | 4730.91 | 4787.21 | 0.00 | T1 1.5R @ 4730.91 |
| Stop hit — per-position SL triggered | 2025-01-06 12:10:00 | 4750.75 | 4782.83 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:15:00 | 4950.00 | 4923.69 | 0.00 | ORB-long ORB[4890.05,4920.50] vol=1.6x ATR=14.00 |
| Stop hit — per-position SL triggered | 2025-01-09 10:20:00 | 4936.00 | 4926.13 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:30:00 | 5435.00 | 5418.10 | 0.00 | ORB-long ORB[5373.30,5428.25] vol=4.2x ATR=13.72 |
| Stop hit — per-position SL triggered | 2025-01-21 09:35:00 | 5421.28 | 5416.74 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:40:00 | 5466.10 | 5421.32 | 0.00 | ORB-long ORB[5340.55,5418.25] vol=3.7x ATR=19.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:00:00 | 5495.33 | 5452.47 | 0.00 | T1 1.5R @ 5495.33 |
| Stop hit — per-position SL triggered | 2025-01-23 10:05:00 | 5466.10 | 5453.63 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:45:00 | 5382.30 | 5349.76 | 0.00 | ORB-long ORB[5300.00,5369.65] vol=2.0x ATR=15.50 |
| Stop hit — per-position SL triggered | 2025-01-30 10:50:00 | 5366.80 | 5352.34 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 11:00:00 | 5368.85 | 5373.67 | 0.00 | ORB-short ORB[5382.00,5443.00] vol=1.7x ATR=16.71 |
| Stop hit — per-position SL triggered | 2025-01-31 12:00:00 | 5385.56 | 5372.92 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-02-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:35:00 | 5570.05 | 5547.39 | 0.00 | ORB-long ORB[5523.85,5568.80] vol=2.1x ATR=17.12 |
| Stop hit — per-position SL triggered | 2025-02-05 09:50:00 | 5552.93 | 5558.34 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:35:00 | 5560.55 | 5523.73 | 0.00 | ORB-long ORB[5502.65,5549.00] vol=3.6x ATR=20.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 12:25:00 | 5590.65 | 5547.03 | 0.00 | T1 1.5R @ 5590.65 |
| Target hit | 2025-02-07 14:05:00 | 5561.10 | 5563.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 57 — SELL (started 2025-02-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:50:00 | 5194.35 | 5232.47 | 0.00 | ORB-short ORB[5262.60,5322.00] vol=1.6x ATR=28.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 10:05:00 | 5152.17 | 5208.18 | 0.00 | T1 1.5R @ 5152.17 |
| Target hit | 2025-02-12 15:20:00 | 5106.00 | 5126.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2025-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:30:00 | 4810.00 | 4835.39 | 0.00 | ORB-short ORB[4824.00,4868.35] vol=1.7x ATR=16.76 |
| Stop hit — per-position SL triggered | 2025-02-18 09:35:00 | 4826.76 | 4833.53 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-19 10:50:00 | 4887.00 | 4915.96 | 0.00 | ORB-short ORB[4887.50,4951.30] vol=3.5x ATR=15.81 |
| Stop hit — per-position SL triggered | 2025-02-19 11:05:00 | 4902.81 | 4915.05 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:20:00 | 4492.20 | 4555.44 | 0.00 | ORB-short ORB[4570.15,4634.05] vol=1.7x ATR=21.00 |
| Stop hit — per-position SL triggered | 2025-03-12 10:35:00 | 4513.20 | 4550.06 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:40:00 | 4275.10 | 4221.74 | 0.00 | ORB-long ORB[4175.00,4238.00] vol=1.9x ATR=23.42 |
| Stop hit — per-position SL triggered | 2025-04-15 09:50:00 | 4251.68 | 4226.95 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-04-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 10:45:00 | 4307.00 | 4344.61 | 0.00 | ORB-short ORB[4319.40,4360.00] vol=2.1x ATR=17.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:00:00 | 4281.30 | 4336.44 | 0.00 | T1 1.5R @ 4281.30 |
| Target hit | 2025-04-16 15:20:00 | 4230.00 | 4271.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 4267.30 | 4251.73 | 0.00 | ORB-long ORB[4216.10,4264.30] vol=4.5x ATR=13.72 |
| Stop hit — per-position SL triggered | 2025-04-21 09:35:00 | 4253.58 | 4252.20 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 11:10:00 | 4167.30 | 4167.83 | 0.00 | ORB-short ORB[4170.00,4219.00] vol=1.8x ATR=9.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 11:30:00 | 4153.22 | 4167.29 | 0.00 | T1 1.5R @ 4153.22 |
| Target hit | 2025-04-29 12:35:00 | 4164.80 | 4162.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:15:00 | 4093.10 | 4076.00 | 0.00 | ORB-long ORB[4050.00,4090.00] vol=4.8x ATR=11.05 |
| Stop hit — per-position SL triggered | 2025-05-07 13:05:00 | 4082.05 | 4079.91 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 11:00:00 | 4139.90 | 4110.32 | 0.00 | ORB-long ORB[4080.00,4119.50] vol=2.7x ATR=11.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 11:25:00 | 4156.71 | 4116.46 | 0.00 | T1 1.5R @ 4156.71 |
| Stop hit — per-position SL triggered | 2025-05-08 12:50:00 | 4139.90 | 4133.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-07-10 10:25:00 | 5011.65 | 2024-07-10 10:35:00 | 4973.60 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2024-07-10 10:25:00 | 5011.65 | 2024-07-10 10:45:00 | 5011.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 09:35:00 | 5020.00 | 2024-07-11 11:25:00 | 4994.10 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-11 09:35:00 | 5020.00 | 2024-07-11 14:35:00 | 4976.00 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2024-07-12 09:55:00 | 5038.95 | 2024-07-12 10:20:00 | 5064.00 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-12 09:55:00 | 5038.95 | 2024-07-12 10:30:00 | 5038.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 09:30:00 | 5187.45 | 2024-07-24 09:40:00 | 5165.88 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-29 09:35:00 | 5202.65 | 2024-07-29 09:45:00 | 5184.39 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-07-29 09:35:00 | 5202.65 | 2024-07-29 11:15:00 | 5202.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 10:05:00 | 5225.55 | 2024-07-31 10:10:00 | 5215.12 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-01 10:35:00 | 5201.10 | 2024-08-01 10:50:00 | 5181.10 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-01 10:35:00 | 5201.10 | 2024-08-01 11:10:00 | 5201.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-02 09:35:00 | 5045.00 | 2024-08-02 10:00:00 | 5019.86 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-08-02 09:35:00 | 5045.00 | 2024-08-02 10:10:00 | 5045.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 10:05:00 | 4900.25 | 2024-08-08 10:10:00 | 4913.33 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-14 10:40:00 | 4883.65 | 2024-08-14 12:05:00 | 4903.29 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-14 10:40:00 | 4883.65 | 2024-08-14 15:20:00 | 4913.45 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-08-19 09:50:00 | 5255.80 | 2024-08-19 10:05:00 | 5282.48 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-19 09:50:00 | 5255.80 | 2024-08-19 10:20:00 | 5255.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 10:10:00 | 5503.95 | 2024-08-22 10:20:00 | 5489.35 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-26 10:05:00 | 5511.00 | 2024-08-26 12:00:00 | 5484.11 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-08-26 10:05:00 | 5511.00 | 2024-08-26 14:25:00 | 5511.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-27 10:45:00 | 5492.20 | 2024-08-27 10:55:00 | 5475.36 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-08-27 10:45:00 | 5492.20 | 2024-08-27 13:05:00 | 5492.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 09:45:00 | 5716.10 | 2024-08-29 10:25:00 | 5695.13 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-02 09:35:00 | 5783.00 | 2024-09-02 10:00:00 | 5807.18 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-09-02 09:35:00 | 5783.00 | 2024-09-02 10:05:00 | 5783.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:45:00 | 5767.75 | 2024-09-03 10:10:00 | 5752.77 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-05 09:35:00 | 5655.55 | 2024-09-05 09:50:00 | 5669.19 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-10 10:20:00 | 5601.55 | 2024-09-10 10:25:00 | 5618.85 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-13 10:45:00 | 5740.00 | 2024-09-13 11:00:00 | 5761.93 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-09-13 10:45:00 | 5740.00 | 2024-09-13 11:20:00 | 5740.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-16 09:55:00 | 5727.30 | 2024-09-16 10:50:00 | 5742.06 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-17 09:35:00 | 5756.95 | 2024-09-17 09:40:00 | 5741.94 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-23 10:55:00 | 5420.95 | 2024-09-23 11:15:00 | 5433.91 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-24 09:30:00 | 5456.00 | 2024-09-24 09:45:00 | 5466.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-10-03 09:35:00 | 5277.70 | 2024-10-03 09:40:00 | 5253.52 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-03 09:35:00 | 5277.70 | 2024-10-03 15:20:00 | 5099.55 | TARGET_HIT | 0.50 | 3.38% |
| BUY | retest1 | 2024-10-04 10:35:00 | 5184.05 | 2024-10-04 10:55:00 | 5213.69 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-10-04 10:35:00 | 5184.05 | 2024-10-04 11:55:00 | 5184.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 10:15:00 | 5358.80 | 2024-10-16 10:25:00 | 5343.36 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-30 09:30:00 | 5265.65 | 2024-10-30 09:50:00 | 5288.96 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-10-30 09:30:00 | 5265.65 | 2024-10-30 10:20:00 | 5272.00 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-11-06 09:30:00 | 5062.30 | 2024-11-06 09:45:00 | 5090.14 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-11-06 09:30:00 | 5062.30 | 2024-11-06 10:10:00 | 5062.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-08 09:30:00 | 5193.60 | 2024-11-08 09:40:00 | 5218.61 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-11-08 09:30:00 | 5193.60 | 2024-11-08 10:50:00 | 5203.80 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-11-13 10:30:00 | 5225.40 | 2024-11-13 10:50:00 | 5250.60 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-11-19 10:05:00 | 5180.75 | 2024-11-19 10:10:00 | 5201.18 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-11-19 10:05:00 | 5180.75 | 2024-11-19 12:05:00 | 5185.30 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2024-11-21 09:35:00 | 5082.00 | 2024-11-21 09:45:00 | 5101.85 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-11-22 10:00:00 | 5238.00 | 2024-11-22 10:10:00 | 5224.32 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-11-25 11:00:00 | 5431.00 | 2024-11-25 11:20:00 | 5454.44 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-11-25 11:00:00 | 5431.00 | 2024-11-25 11:40:00 | 5431.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-26 09:35:00 | 5539.00 | 2024-11-26 09:55:00 | 5521.37 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-28 09:30:00 | 5354.35 | 2024-11-28 09:35:00 | 5370.21 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-04 09:30:00 | 5369.05 | 2024-12-04 09:40:00 | 5358.17 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-12-10 10:10:00 | 5400.00 | 2024-12-10 10:35:00 | 5384.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-11 11:15:00 | 5372.00 | 2024-12-11 11:25:00 | 5357.05 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-12-11 11:15:00 | 5372.00 | 2024-12-11 15:00:00 | 5372.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-12 10:15:00 | 5424.80 | 2024-12-12 10:25:00 | 5411.76 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-13 09:30:00 | 5363.70 | 2024-12-13 10:05:00 | 5376.77 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-17 09:35:00 | 5335.50 | 2024-12-17 10:00:00 | 5317.96 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-12-17 09:35:00 | 5335.50 | 2024-12-17 15:20:00 | 5223.10 | TARGET_HIT | 0.50 | 2.11% |
| SELL | retest1 | 2024-12-18 10:50:00 | 5196.00 | 2024-12-18 12:50:00 | 5171.74 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-12-18 10:50:00 | 5196.00 | 2024-12-18 13:05:00 | 5196.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 09:30:00 | 4702.35 | 2024-12-26 09:45:00 | 4685.11 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-26 09:30:00 | 4702.35 | 2024-12-26 10:35:00 | 4702.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 10:45:00 | 4739.75 | 2024-12-30 11:00:00 | 4726.10 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-01 11:10:00 | 4729.85 | 2025-01-01 11:15:00 | 4741.55 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-02 09:40:00 | 4730.95 | 2025-01-02 10:20:00 | 4718.62 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-06 11:10:00 | 4750.75 | 2025-01-06 11:45:00 | 4730.91 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-06 11:10:00 | 4750.75 | 2025-01-06 12:10:00 | 4750.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 10:15:00 | 4950.00 | 2025-01-09 10:20:00 | 4936.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-21 09:30:00 | 5435.00 | 2025-01-21 09:35:00 | 5421.28 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-23 09:40:00 | 5466.10 | 2025-01-23 10:00:00 | 5495.33 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-01-23 09:40:00 | 5466.10 | 2025-01-23 10:05:00 | 5466.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-30 10:45:00 | 5382.30 | 2025-01-30 10:50:00 | 5366.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-31 11:00:00 | 5368.85 | 2025-01-31 12:00:00 | 5385.56 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-02-05 09:35:00 | 5570.05 | 2025-02-05 09:50:00 | 5552.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-02-07 10:35:00 | 5560.55 | 2025-02-07 12:25:00 | 5590.65 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-02-07 10:35:00 | 5560.55 | 2025-02-07 14:05:00 | 5561.10 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2025-02-12 09:50:00 | 5194.35 | 2025-02-12 10:05:00 | 5152.17 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2025-02-12 09:50:00 | 5194.35 | 2025-02-12 15:20:00 | 5106.00 | TARGET_HIT | 0.50 | 1.70% |
| SELL | retest1 | 2025-02-18 09:30:00 | 4810.00 | 2025-02-18 09:35:00 | 4826.76 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-02-19 10:50:00 | 4887.00 | 2025-02-19 11:05:00 | 4902.81 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-12 10:20:00 | 4492.20 | 2025-03-12 10:35:00 | 4513.20 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-04-15 09:40:00 | 4275.10 | 2025-04-15 09:50:00 | 4251.68 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-04-16 10:45:00 | 4307.00 | 2025-04-16 11:00:00 | 4281.30 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-04-16 10:45:00 | 4307.00 | 2025-04-16 15:20:00 | 4230.00 | TARGET_HIT | 0.50 | 1.79% |
| BUY | retest1 | 2025-04-21 09:30:00 | 4267.30 | 2025-04-21 09:35:00 | 4253.58 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-29 11:10:00 | 4167.30 | 2025-04-29 11:30:00 | 4153.22 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-04-29 11:10:00 | 4167.30 | 2025-04-29 12:35:00 | 4164.80 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2025-05-07 11:15:00 | 4093.10 | 2025-05-07 13:05:00 | 4082.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-08 11:00:00 | 4139.90 | 2025-05-08 11:25:00 | 4156.71 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-05-08 11:00:00 | 4139.90 | 2025-05-08 12:50:00 | 4139.90 | STOP_HIT | 0.50 | 0.00% |
