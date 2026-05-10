# Britannia Industries Ltd. (BRITANNIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 5516.00
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 4
- **Avg / median % per leg:** -0.01% / -0.22%
- **Sum % (uncompounded):** -0.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.09% | 1.2% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.09% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 2 | 20.0% | 1 | 8 | 1 | -0.16% | -1.6% |
| SELL @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 1 | 8 | 1 | -0.16% | -1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 7 | 30.4% | 3 | 16 | 4 | -0.01% | -0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:50:00 | 5825.00 | 5846.21 | 0.00 | ORB-short ORB[5856.00,5915.50] vol=2.9x ATR=22.72 |
| Stop hit — per-position SL triggered | 2026-02-09 13:50:00 | 5847.72 | 5831.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 6083.00 | 6042.18 | 0.00 | ORB-long ORB[6020.00,6057.00] vol=1.5x ATR=18.96 |
| Stop hit — per-position SL triggered | 2026-02-12 10:25:00 | 6064.04 | 6054.24 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 6010.50 | 5996.49 | 0.00 | ORB-long ORB[5965.00,5994.50] vol=3.2x ATR=13.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:05:00 | 6031.32 | 6001.90 | 0.00 | T1 1.5R @ 6031.32 |
| Target hit | 2026-02-16 15:20:00 | 6104.50 | 6075.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 6177.00 | 6148.64 | 0.00 | ORB-long ORB[6123.00,6163.00] vol=2.5x ATR=12.44 |
| Stop hit — per-position SL triggered | 2026-02-18 11:25:00 | 6164.56 | 6151.03 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 6138.50 | 6167.90 | 0.00 | ORB-short ORB[6148.00,6186.50] vol=3.2x ATR=11.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:05:00 | 6121.35 | 6153.67 | 0.00 | T1 1.5R @ 6121.35 |
| Target hit | 2026-02-19 15:20:00 | 6101.50 | 6128.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-02-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:40:00 | 6126.00 | 6111.74 | 0.00 | ORB-long ORB[6065.50,6103.00] vol=2.0x ATR=12.31 |
| Stop hit — per-position SL triggered | 2026-02-24 10:55:00 | 6113.69 | 6113.75 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 6032.00 | 6065.87 | 0.00 | ORB-short ORB[6065.50,6142.50] vol=1.9x ATR=14.50 |
| Stop hit — per-position SL triggered | 2026-02-27 11:55:00 | 6046.50 | 6057.09 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 5854.00 | 5896.97 | 0.00 | ORB-short ORB[5899.00,5952.50] vol=3.0x ATR=12.76 |
| Stop hit — per-position SL triggered | 2026-03-05 10:55:00 | 5866.76 | 5887.65 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 10:15:00 | 5851.50 | 5860.93 | 0.00 | ORB-short ORB[5852.00,5932.50] vol=1.6x ATR=21.80 |
| Stop hit — per-position SL triggered | 2026-03-09 10:55:00 | 5873.30 | 5859.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:15:00 | 5799.50 | 5845.41 | 0.00 | ORB-short ORB[5830.00,5910.00] vol=1.6x ATR=20.72 |
| Stop hit — per-position SL triggered | 2026-03-12 10:35:00 | 5820.22 | 5840.09 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:05:00 | 5630.00 | 5607.53 | 0.00 | ORB-long ORB[5536.50,5610.00] vol=1.7x ATR=12.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:50:00 | 5648.93 | 5615.93 | 0.00 | T1 1.5R @ 5648.93 |
| Target hit | 2026-03-25 15:00:00 | 5638.50 | 5679.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2026-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:00:00 | 5450.00 | 5495.89 | 0.00 | ORB-short ORB[5465.00,5531.50] vol=3.1x ATR=18.35 |
| Stop hit — per-position SL triggered | 2026-04-01 11:05:00 | 5468.35 | 5495.21 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:30:00 | 5705.00 | 5650.34 | 0.00 | ORB-long ORB[5568.50,5630.00] vol=2.2x ATR=16.11 |
| Stop hit — per-position SL triggered | 2026-04-17 10:35:00 | 5688.89 | 5651.24 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:05:00 | 5740.00 | 5737.80 | 0.00 | ORB-long ORB[5685.00,5725.00] vol=3.1x ATR=10.86 |
| Stop hit — per-position SL triggered | 2026-04-21 11:30:00 | 5729.14 | 5737.80 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:35:00 | 5734.00 | 5700.14 | 0.00 | ORB-long ORB[5669.00,5720.00] vol=2.1x ATR=13.62 |
| Stop hit — per-position SL triggered | 2026-04-30 10:40:00 | 5720.38 | 5698.31 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:50:00 | 5788.00 | 5769.54 | 0.00 | ORB-long ORB[5715.00,5779.00] vol=2.5x ATR=12.84 |
| Stop hit — per-position SL triggered | 2026-05-04 10:55:00 | 5775.16 | 5770.03 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 5739.00 | 5770.61 | 0.00 | ORB-short ORB[5754.50,5809.00] vol=1.5x ATR=16.80 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 5755.80 | 5767.02 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 5805.00 | 5815.32 | 0.00 | ORB-short ORB[5831.50,5880.00] vol=3.1x ATR=14.05 |
| Stop hit — per-position SL triggered | 2026-05-06 14:50:00 | 5819.05 | 5802.91 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:15:00 | 5800.00 | 5776.10 | 0.00 | ORB-long ORB[5741.00,5798.00] vol=1.9x ATR=18.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:35:00 | 5827.72 | 5778.21 | 0.00 | T1 1.5R @ 5827.72 |
| Stop hit — per-position SL triggered | 2026-05-07 14:00:00 | 5800.00 | 5787.65 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-09 10:50:00 | 5825.00 | 2026-02-09 13:50:00 | 5847.72 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-12 09:35:00 | 6083.00 | 2026-02-12 10:25:00 | 6064.04 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-16 10:45:00 | 6010.50 | 2026-02-16 11:05:00 | 6031.32 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-16 10:45:00 | 6010.50 | 2026-02-16 15:20:00 | 6104.50 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2026-02-18 11:05:00 | 6177.00 | 2026-02-18 11:25:00 | 6164.56 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-19 11:10:00 | 6138.50 | 2026-02-19 12:05:00 | 6121.35 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-19 11:10:00 | 6138.50 | 2026-02-19 15:20:00 | 6101.50 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-24 10:40:00 | 6126.00 | 2026-02-24 10:55:00 | 6113.69 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-27 10:40:00 | 6032.00 | 2026-02-27 11:55:00 | 6046.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-05 10:45:00 | 5854.00 | 2026-03-05 10:55:00 | 5866.76 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-09 10:15:00 | 5851.50 | 2026-03-09 10:55:00 | 5873.30 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-12 10:15:00 | 5799.50 | 2026-03-12 10:35:00 | 5820.22 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-25 11:05:00 | 5630.00 | 2026-03-25 11:50:00 | 5648.93 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-03-25 11:05:00 | 5630.00 | 2026-03-25 15:00:00 | 5638.50 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-04-01 11:00:00 | 5450.00 | 2026-04-01 11:05:00 | 5468.35 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-17 10:30:00 | 5705.00 | 2026-04-17 10:35:00 | 5688.89 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-21 11:05:00 | 5740.00 | 2026-04-21 11:30:00 | 5729.14 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-04-30 10:35:00 | 5734.00 | 2026-04-30 10:40:00 | 5720.38 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-05-04 10:50:00 | 5788.00 | 2026-05-04 10:55:00 | 5775.16 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-05 09:35:00 | 5739.00 | 2026-05-05 09:40:00 | 5755.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-06 10:55:00 | 5805.00 | 2026-05-06 14:50:00 | 5819.05 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-05-07 11:15:00 | 5800.00 | 2026-05-07 11:35:00 | 5827.72 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-05-07 11:15:00 | 5800.00 | 2026-05-07 14:00:00 | 5800.00 | STOP_HIT | 0.50 | 0.00% |
