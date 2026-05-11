# LTM Ltd. (LTM)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15838 bars)
- **Last close:** 4360.00
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
| ENTRY1 | 56 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 10 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 46
- **Target hits / Stop hits / Partials:** 10 / 46 / 19
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 7.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 21 | 46.7% | 7 | 24 | 14 | 0.14% | 6.4% |
| BUY @ 2nd Alert (retest1) | 45 | 21 | 46.7% | 7 | 24 | 14 | 0.14% | 6.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 30 | 8 | 26.7% | 3 | 22 | 5 | 0.03% | 0.9% |
| SELL @ 2nd Alert (retest1) | 30 | 8 | 26.7% | 3 | 22 | 5 | 0.03% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 75 | 29 | 38.7% | 10 | 46 | 19 | 0.10% | 7.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:40:00 | 5029.80 | 4995.12 | 0.00 | ORB-long ORB[4940.00,5013.60] vol=1.5x ATR=16.24 |
| Stop hit — per-position SL triggered | 2025-05-14 09:45:00 | 5013.56 | 4996.80 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 10:40:00 | 4985.70 | 4998.70 | 0.00 | ORB-short ORB[5002.50,5055.00] vol=1.7x ATR=14.66 |
| Stop hit — per-position SL triggered | 2025-05-15 10:55:00 | 5000.36 | 4998.15 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 09:55:00 | 5065.00 | 5006.81 | 0.00 | ORB-long ORB[4970.20,5008.70] vol=2.0x ATR=17.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 10:10:00 | 5091.49 | 5030.17 | 0.00 | T1 1.5R @ 5091.49 |
| Stop hit — per-position SL triggered | 2025-05-22 11:00:00 | 5065.00 | 5053.74 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:35:00 | 5066.20 | 5033.89 | 0.00 | ORB-long ORB[4982.90,5058.30] vol=1.5x ATR=14.28 |
| Stop hit — per-position SL triggered | 2025-05-23 13:45:00 | 5051.92 | 5063.63 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:55:00 | 5093.40 | 5078.50 | 0.00 | ORB-long ORB[5041.20,5074.90] vol=3.2x ATR=12.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 11:15:00 | 5112.51 | 5082.37 | 0.00 | T1 1.5R @ 5112.51 |
| Target hit | 2025-05-26 14:45:00 | 5103.30 | 5103.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2025-05-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 11:00:00 | 5072.90 | 5083.83 | 0.00 | ORB-short ORB[5076.10,5109.50] vol=4.8x ATR=8.69 |
| Stop hit — per-position SL triggered | 2025-05-27 11:05:00 | 5081.59 | 5083.70 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:55:00 | 5071.20 | 5085.17 | 0.00 | ORB-short ORB[5085.20,5159.70] vol=3.0x ATR=12.11 |
| Stop hit — per-position SL triggered | 2025-05-30 11:10:00 | 5083.31 | 5084.60 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:10:00 | 5069.00 | 5042.56 | 0.00 | ORB-long ORB[5001.00,5060.00] vol=2.5x ATR=14.52 |
| Stop hit — per-position SL triggered | 2025-06-02 11:40:00 | 5054.48 | 5044.55 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 11:00:00 | 5426.50 | 5398.20 | 0.00 | ORB-long ORB[5360.00,5414.00] vol=3.8x ATR=13.05 |
| Stop hit — per-position SL triggered | 2025-06-12 11:10:00 | 5413.45 | 5400.70 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 5490.50 | 5472.99 | 0.00 | ORB-long ORB[5450.00,5480.50] vol=2.0x ATR=14.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 09:35:00 | 5511.96 | 5487.76 | 0.00 | T1 1.5R @ 5511.96 |
| Target hit | 2025-06-17 10:00:00 | 5501.00 | 5501.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2025-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 09:30:00 | 5364.00 | 5390.84 | 0.00 | ORB-short ORB[5379.50,5455.00] vol=1.9x ATR=15.35 |
| Stop hit — per-position SL triggered | 2025-06-19 09:45:00 | 5379.35 | 5378.75 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:00:00 | 5406.50 | 5395.77 | 0.00 | ORB-long ORB[5341.00,5403.50] vol=2.2x ATR=15.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:40:00 | 5430.09 | 5404.67 | 0.00 | T1 1.5R @ 5430.09 |
| Target hit | 2025-06-25 15:20:00 | 5429.50 | 5431.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2025-06-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 10:45:00 | 5359.00 | 5379.86 | 0.00 | ORB-short ORB[5376.50,5412.00] vol=4.3x ATR=12.74 |
| Stop hit — per-position SL triggered | 2025-06-27 12:55:00 | 5371.74 | 5371.65 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:30:00 | 5375.00 | 5344.65 | 0.00 | ORB-long ORB[5300.50,5343.00] vol=4.4x ATR=13.99 |
| Stop hit — per-position SL triggered | 2025-07-01 09:35:00 | 5361.01 | 5348.76 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:00:00 | 5301.50 | 5330.08 | 0.00 | ORB-short ORB[5320.00,5382.50] vol=4.6x ATR=14.61 |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 5316.11 | 5327.43 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 09:40:00 | 5313.00 | 5319.21 | 0.00 | ORB-short ORB[5316.00,5350.50] vol=4.4x ATR=11.00 |
| Stop hit — per-position SL triggered | 2025-07-04 10:05:00 | 5324.00 | 5317.31 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:00:00 | 5368.50 | 5342.39 | 0.00 | ORB-long ORB[5305.50,5358.50] vol=2.5x ATR=13.45 |
| Stop hit — per-position SL triggered | 2025-07-07 10:25:00 | 5355.05 | 5348.99 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:00:00 | 5290.00 | 5302.72 | 0.00 | ORB-short ORB[5299.00,5348.50] vol=1.5x ATR=9.19 |
| Stop hit — per-position SL triggered | 2025-07-10 11:15:00 | 5299.19 | 5301.85 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:40:00 | 5232.00 | 5207.51 | 0.00 | ORB-long ORB[5158.50,5225.00] vol=1.9x ATR=14.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 10:55:00 | 5254.16 | 5221.44 | 0.00 | T1 1.5R @ 5254.16 |
| Stop hit — per-position SL triggered | 2025-07-15 13:50:00 | 5232.00 | 5235.58 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:30:00 | 5291.50 | 5277.91 | 0.00 | ORB-long ORB[5241.50,5290.00] vol=1.8x ATR=10.36 |
| Stop hit — per-position SL triggered | 2025-07-16 09:40:00 | 5281.14 | 5280.75 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:35:00 | 5281.00 | 5297.66 | 0.00 | ORB-short ORB[5285.00,5335.00] vol=1.6x ATR=12.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 09:40:00 | 5262.27 | 5292.25 | 0.00 | T1 1.5R @ 5262.27 |
| Target hit | 2025-07-17 15:20:00 | 5179.00 | 5223.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:15:00 | 5175.00 | 5176.35 | 0.00 | ORB-short ORB[5181.00,5205.00] vol=5.1x ATR=10.02 |
| Stop hit — per-position SL triggered | 2025-07-23 10:30:00 | 5185.02 | 5177.32 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:55:00 | 5097.00 | 5146.90 | 0.00 | ORB-short ORB[5161.50,5219.50] vol=2.2x ATR=14.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 5075.71 | 5134.81 | 0.00 | T1 1.5R @ 5075.71 |
| Stop hit — per-position SL triggered | 2025-07-25 11:25:00 | 5097.00 | 5129.60 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:50:00 | 5074.00 | 5068.50 | 0.00 | ORB-long ORB[5045.50,5069.00] vol=2.8x ATR=10.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 11:15:00 | 5090.49 | 5073.65 | 0.00 | T1 1.5R @ 5090.49 |
| Target hit | 2025-07-30 15:20:00 | 5141.50 | 5104.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2025-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:00:00 | 5071.50 | 5093.97 | 0.00 | ORB-short ORB[5106.00,5132.00] vol=1.5x ATR=11.33 |
| Stop hit — per-position SL triggered | 2025-08-06 11:50:00 | 5082.83 | 5088.41 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:50:00 | 5372.50 | 5356.02 | 0.00 | ORB-long ORB[5325.50,5355.00] vol=2.0x ATR=9.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 11:25:00 | 5386.45 | 5362.56 | 0.00 | T1 1.5R @ 5386.45 |
| Stop hit — per-position SL triggered | 2025-09-16 11:35:00 | 5372.50 | 5364.12 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:50:00 | 5234.00 | 5251.62 | 0.00 | ORB-short ORB[5255.50,5294.50] vol=2.1x ATR=14.60 |
| Stop hit — per-position SL triggered | 2025-09-23 09:55:00 | 5248.60 | 5249.92 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:05:00 | 5202.50 | 5229.82 | 0.00 | ORB-short ORB[5235.50,5275.00] vol=1.6x ATR=12.74 |
| Stop hit — per-position SL triggered | 2025-09-24 11:30:00 | 5215.24 | 5223.85 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 09:50:00 | 5188.50 | 5159.78 | 0.00 | ORB-long ORB[5121.50,5158.00] vol=1.6x ATR=14.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:10:00 | 5209.61 | 5176.54 | 0.00 | T1 1.5R @ 5209.61 |
| Target hit | 2025-10-06 15:20:00 | 5270.00 | 5239.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2025-10-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:50:00 | 5610.00 | 5628.69 | 0.00 | ORB-short ORB[5615.50,5658.00] vol=1.6x ATR=10.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:30:00 | 5593.95 | 5622.10 | 0.00 | T1 1.5R @ 5593.95 |
| Stop hit — per-position SL triggered | 2025-10-28 13:20:00 | 5610.00 | 5612.80 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:30:00 | 5651.50 | 5601.06 | 0.00 | ORB-long ORB[5571.00,5605.00] vol=2.0x ATR=13.85 |
| Stop hit — per-position SL triggered | 2025-10-29 10:40:00 | 5637.65 | 5605.90 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 10:00:00 | 5741.50 | 5725.38 | 0.00 | ORB-long ORB[5702.00,5740.00] vol=2.4x ATR=14.45 |
| Stop hit — per-position SL triggered | 2025-10-31 10:05:00 | 5727.05 | 5725.30 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-11-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:20:00 | 5659.00 | 5616.04 | 0.00 | ORB-long ORB[5575.00,5622.00] vol=1.6x ATR=14.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:25:00 | 5680.88 | 5624.73 | 0.00 | T1 1.5R @ 5680.88 |
| Stop hit — per-position SL triggered | 2025-11-10 12:05:00 | 5659.00 | 5651.68 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-11-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 09:35:00 | 5865.00 | 5840.95 | 0.00 | ORB-long ORB[5789.00,5861.50] vol=3.1x ATR=15.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:40:00 | 5888.20 | 5847.84 | 0.00 | T1 1.5R @ 5888.20 |
| Target hit | 2025-11-19 15:20:00 | 5959.50 | 5955.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2025-11-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:45:00 | 5934.00 | 5963.63 | 0.00 | ORB-short ORB[5959.00,6014.00] vol=1.9x ATR=16.40 |
| Target hit | 2025-11-21 15:20:00 | 5925.00 | 5941.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2025-11-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 09:35:00 | 6007.50 | 5973.46 | 0.00 | ORB-long ORB[5908.00,5994.00] vol=1.7x ATR=14.88 |
| Stop hit — per-position SL triggered | 2025-11-24 09:45:00 | 5992.62 | 5982.98 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-11-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:10:00 | 5931.50 | 5907.90 | 0.00 | ORB-long ORB[5873.50,5924.00] vol=2.1x ATR=15.99 |
| Stop hit — per-position SL triggered | 2025-11-27 10:25:00 | 5915.51 | 5909.81 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-12-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:05:00 | 6252.00 | 6207.29 | 0.00 | ORB-long ORB[6159.00,6228.00] vol=2.6x ATR=13.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 11:15:00 | 6271.72 | 6211.34 | 0.00 | T1 1.5R @ 6271.72 |
| Stop hit — per-position SL triggered | 2025-12-04 11:40:00 | 6252.00 | 6218.93 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-12-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:45:00 | 6315.50 | 6277.47 | 0.00 | ORB-long ORB[6245.50,6272.00] vol=7.0x ATR=13.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:50:00 | 6336.09 | 6284.83 | 0.00 | T1 1.5R @ 6336.09 |
| Target hit | 2025-12-05 12:50:00 | 6326.00 | 6329.41 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2025-12-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 11:10:00 | 6340.50 | 6323.11 | 0.00 | ORB-long ORB[6290.00,6337.00] vol=1.7x ATR=11.02 |
| Stop hit — per-position SL triggered | 2025-12-08 11:20:00 | 6329.48 | 6324.31 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-12-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:35:00 | 6178.00 | 6197.32 | 0.00 | ORB-short ORB[6190.50,6255.50] vol=1.6x ATR=16.62 |
| Stop hit — per-position SL triggered | 2025-12-09 11:00:00 | 6194.62 | 6196.72 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-12-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:05:00 | 6204.00 | 6205.11 | 0.00 | ORB-short ORB[6209.00,6259.00] vol=2.2x ATR=12.36 |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 6216.36 | 6208.08 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:40:00 | 6225.00 | 6251.63 | 0.00 | ORB-short ORB[6250.50,6308.00] vol=7.6x ATR=13.36 |
| Stop hit — per-position SL triggered | 2025-12-18 11:00:00 | 6238.36 | 6240.57 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 6222.50 | 6214.40 | 0.00 | ORB-long ORB[6187.00,6209.50] vol=2.1x ATR=10.36 |
| Stop hit — per-position SL triggered | 2025-12-24 11:20:00 | 6212.14 | 6216.41 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 10:10:00 | 6114.00 | 6134.00 | 0.00 | ORB-short ORB[6136.00,6174.50] vol=2.3x ATR=12.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:05:00 | 6094.93 | 6122.50 | 0.00 | T1 1.5R @ 6094.93 |
| Target hit | 2025-12-26 15:20:00 | 6035.00 | 6082.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 6100.00 | 6066.30 | 0.00 | ORB-long ORB[6019.00,6078.00] vol=1.5x ATR=17.90 |
| Stop hit — per-position SL triggered | 2025-12-29 09:40:00 | 6082.10 | 6071.22 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:20:00 | 6088.00 | 6050.42 | 0.00 | ORB-long ORB[6003.00,6054.00] vol=1.9x ATR=11.38 |
| Stop hit — per-position SL triggered | 2025-12-30 10:30:00 | 6076.62 | 6055.32 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 6055.50 | 6065.49 | 0.00 | ORB-short ORB[6063.50,6104.50] vol=1.7x ATR=9.12 |
| Stop hit — per-position SL triggered | 2026-01-01 11:20:00 | 6064.62 | 6065.41 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-01-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:10:00 | 6082.50 | 6048.25 | 0.00 | ORB-long ORB[5977.50,6065.00] vol=1.6x ATR=18.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:35:00 | 6110.30 | 6057.34 | 0.00 | T1 1.5R @ 6110.30 |
| Stop hit — per-position SL triggered | 2026-01-07 12:40:00 | 6082.50 | 6078.31 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-01-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:40:00 | 6015.00 | 6053.73 | 0.00 | ORB-short ORB[6058.00,6134.00] vol=1.6x ATR=15.86 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 6030.86 | 6045.59 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-01-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 09:45:00 | 6093.00 | 6048.96 | 0.00 | ORB-long ORB[5999.50,6070.50] vol=2.3x ATR=19.04 |
| Stop hit — per-position SL triggered | 2026-01-09 09:50:00 | 6073.96 | 6051.79 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 11:05:00 | 6028.00 | 6045.14 | 0.00 | ORB-short ORB[6031.00,6105.00] vol=1.5x ATR=14.14 |
| Stop hit — per-position SL triggered | 2026-01-14 11:25:00 | 6042.14 | 6042.02 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 5657.00 | 5623.16 | 0.00 | ORB-long ORB[5586.00,5641.50] vol=1.6x ATR=14.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:55:00 | 5678.22 | 5651.51 | 0.00 | T1 1.5R @ 5678.22 |
| Stop hit — per-position SL triggered | 2026-02-10 13:20:00 | 5657.00 | 5653.81 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 5635.00 | 5640.57 | 0.00 | ORB-short ORB[5649.50,5692.00] vol=1.9x ATR=11.64 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 5646.64 | 5639.84 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 4805.50 | 4841.71 | 0.00 | ORB-short ORB[4814.50,4876.00] vol=1.6x ATR=18.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:55:00 | 4778.43 | 4828.39 | 0.00 | T1 1.5R @ 4778.43 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 4805.50 | 4812.50 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 4689.70 | 4671.07 | 0.00 | ORB-long ORB[4632.20,4688.00] vol=2.3x ATR=23.32 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 4666.38 | 4671.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:40:00 | 5029.80 | 2025-05-14 09:45:00 | 5013.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-05-15 10:40:00 | 4985.70 | 2025-05-15 10:55:00 | 5000.36 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-22 09:55:00 | 5065.00 | 2025-05-22 10:10:00 | 5091.49 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-05-22 09:55:00 | 5065.00 | 2025-05-22 11:00:00 | 5065.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-23 09:35:00 | 5066.20 | 2025-05-23 13:45:00 | 5051.92 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-05-26 10:55:00 | 5093.40 | 2025-05-26 11:15:00 | 5112.51 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-05-26 10:55:00 | 5093.40 | 2025-05-26 14:45:00 | 5103.30 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2025-05-27 11:00:00 | 5072.90 | 2025-05-27 11:05:00 | 5081.59 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-05-30 10:55:00 | 5071.20 | 2025-05-30 11:10:00 | 5083.31 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-02 11:10:00 | 5069.00 | 2025-06-02 11:40:00 | 5054.48 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-12 11:00:00 | 5426.50 | 2025-06-12 11:10:00 | 5413.45 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-17 09:30:00 | 5490.50 | 2025-06-17 09:35:00 | 5511.96 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-17 09:30:00 | 5490.50 | 2025-06-17 10:00:00 | 5501.00 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2025-06-19 09:30:00 | 5364.00 | 2025-06-19 09:45:00 | 5379.35 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-25 10:00:00 | 5406.50 | 2025-06-25 10:40:00 | 5430.09 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-25 10:00:00 | 5406.50 | 2025-06-25 15:20:00 | 5429.50 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-06-27 10:45:00 | 5359.00 | 2025-06-27 12:55:00 | 5371.74 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-01 09:30:00 | 5375.00 | 2025-07-01 09:35:00 | 5361.01 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-02 11:00:00 | 5301.50 | 2025-07-02 11:15:00 | 5316.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-04 09:40:00 | 5313.00 | 2025-07-04 10:05:00 | 5324.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-07 10:00:00 | 5368.50 | 2025-07-07 10:25:00 | 5355.05 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-10 11:00:00 | 5290.00 | 2025-07-10 11:15:00 | 5299.19 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-15 10:40:00 | 5232.00 | 2025-07-15 10:55:00 | 5254.16 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-07-15 10:40:00 | 5232.00 | 2025-07-15 13:50:00 | 5232.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-16 09:30:00 | 5291.50 | 2025-07-16 09:40:00 | 5281.14 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-17 09:35:00 | 5281.00 | 2025-07-17 09:40:00 | 5262.27 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-17 09:35:00 | 5281.00 | 2025-07-17 15:20:00 | 5179.00 | TARGET_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2025-07-23 10:15:00 | 5175.00 | 2025-07-23 10:30:00 | 5185.02 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-25 10:55:00 | 5097.00 | 2025-07-25 11:15:00 | 5075.71 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-25 10:55:00 | 5097.00 | 2025-07-25 11:25:00 | 5097.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-30 10:50:00 | 5074.00 | 2025-07-30 11:15:00 | 5090.49 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-07-30 10:50:00 | 5074.00 | 2025-07-30 15:20:00 | 5141.50 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2025-08-06 11:00:00 | 5071.50 | 2025-08-06 11:50:00 | 5082.83 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-16 10:50:00 | 5372.50 | 2025-09-16 11:25:00 | 5386.45 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-09-16 10:50:00 | 5372.50 | 2025-09-16 11:35:00 | 5372.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 09:50:00 | 5234.00 | 2025-09-23 09:55:00 | 5248.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-24 11:05:00 | 5202.50 | 2025-09-24 11:30:00 | 5215.24 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-06 09:50:00 | 5188.50 | 2025-10-06 10:10:00 | 5209.61 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-10-06 09:50:00 | 5188.50 | 2025-10-06 15:20:00 | 5270.00 | TARGET_HIT | 0.50 | 1.57% |
| SELL | retest1 | 2025-10-28 10:50:00 | 5610.00 | 2025-10-28 11:30:00 | 5593.95 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-10-28 10:50:00 | 5610.00 | 2025-10-28 13:20:00 | 5610.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 10:30:00 | 5651.50 | 2025-10-29 10:40:00 | 5637.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-31 10:00:00 | 5741.50 | 2025-10-31 10:05:00 | 5727.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-10 10:20:00 | 5659.00 | 2025-11-10 10:25:00 | 5680.88 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-11-10 10:20:00 | 5659.00 | 2025-11-10 12:05:00 | 5659.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-19 09:35:00 | 5865.00 | 2025-11-19 09:40:00 | 5888.20 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-11-19 09:35:00 | 5865.00 | 2025-11-19 15:20:00 | 5959.50 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2025-11-21 10:45:00 | 5934.00 | 2025-11-21 15:20:00 | 5925.00 | TARGET_HIT | 1.00 | 0.15% |
| BUY | retest1 | 2025-11-24 09:35:00 | 6007.50 | 2025-11-24 09:45:00 | 5992.62 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-27 10:10:00 | 5931.50 | 2025-11-27 10:25:00 | 5915.51 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-04 11:05:00 | 6252.00 | 2025-12-04 11:15:00 | 6271.72 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-12-04 11:05:00 | 6252.00 | 2025-12-04 11:40:00 | 6252.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:45:00 | 6315.50 | 2025-12-05 10:50:00 | 6336.09 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-12-05 10:45:00 | 6315.50 | 2025-12-05 12:50:00 | 6326.00 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-12-08 11:10:00 | 6340.50 | 2025-12-08 11:20:00 | 6329.48 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-09 10:35:00 | 6178.00 | 2025-12-09 11:00:00 | 6194.62 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-16 11:05:00 | 6204.00 | 2025-12-16 11:15:00 | 6216.36 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-18 10:40:00 | 6225.00 | 2025-12-18 11:00:00 | 6238.36 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-24 10:55:00 | 6222.50 | 2025-12-24 11:20:00 | 6212.14 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-26 10:10:00 | 6114.00 | 2025-12-26 11:05:00 | 6094.93 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-26 10:10:00 | 6114.00 | 2025-12-26 15:20:00 | 6035.00 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-12-29 09:30:00 | 6100.00 | 2025-12-29 09:40:00 | 6082.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-30 10:20:00 | 6088.00 | 2025-12-30 10:30:00 | 6076.62 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-01 11:15:00 | 6055.50 | 2026-01-01 11:20:00 | 6064.62 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2026-01-07 10:10:00 | 6082.50 | 2026-01-07 10:35:00 | 6110.30 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-01-07 10:10:00 | 6082.50 | 2026-01-07 12:40:00 | 6082.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 10:40:00 | 6015.00 | 2026-01-08 11:35:00 | 6030.86 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-09 09:45:00 | 6093.00 | 2026-01-09 09:50:00 | 6073.96 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-14 11:05:00 | 6028.00 | 2026-01-14 11:25:00 | 6042.14 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-10 09:45:00 | 5657.00 | 2026-02-10 12:55:00 | 5678.22 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-10 09:45:00 | 5657.00 | 2026-02-10 13:20:00 | 5657.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 11:10:00 | 5635.00 | 2026-02-11 11:30:00 | 5646.64 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-23 09:40:00 | 4805.50 | 2026-02-23 09:55:00 | 4778.43 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-23 09:40:00 | 4805.50 | 2026-02-23 11:00:00 | 4805.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:35:00 | 4689.70 | 2026-04-22 09:40:00 | 4666.38 | STOP_HIT | 1.00 | -0.50% |
