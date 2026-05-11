# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16888 bars)
- **Last close:** 7309.00
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
| ENTRY1 | 72 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 13 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 104 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 59
- **Target hits / Stop hits / Partials:** 13 / 59 / 32
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 11.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 25 | 45.5% | 8 | 30 | 17 | 0.16% | 8.7% |
| BUY @ 2nd Alert (retest1) | 55 | 25 | 45.5% | 8 | 30 | 17 | 0.16% | 8.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 20 | 40.8% | 5 | 29 | 15 | 0.07% | 3.2% |
| SELL @ 2nd Alert (retest1) | 49 | 20 | 40.8% | 5 | 29 | 15 | 0.07% | 3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 104 | 45 | 43.3% | 13 | 59 | 32 | 0.11% | 12.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-13 11:10:00 | 5439.50 | 5474.25 | 0.00 | ORB-short ORB[5500.00,5553.50] vol=4.3x ATR=12.97 |
| Stop hit — per-position SL triggered | 2025-05-13 11:50:00 | 5452.47 | 5463.34 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 5577.50 | 5561.91 | 0.00 | ORB-long ORB[5509.50,5568.00] vol=2.7x ATR=12.44 |
| Stop hit — per-position SL triggered | 2025-05-19 09:35:00 | 5565.06 | 5563.12 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 11:05:00 | 5431.50 | 5415.68 | 0.00 | ORB-long ORB[5376.00,5424.00] vol=2.5x ATR=14.36 |
| Stop hit — per-position SL triggered | 2025-05-23 11:10:00 | 5417.14 | 5415.85 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:50:00 | 5382.50 | 5404.43 | 0.00 | ORB-short ORB[5394.00,5425.00] vol=1.9x ATR=10.83 |
| Stop hit — per-position SL triggered | 2025-05-27 10:10:00 | 5393.33 | 5399.47 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 10:25:00 | 5389.00 | 5407.61 | 0.00 | ORB-short ORB[5403.00,5444.00] vol=2.6x ATR=10.37 |
| Stop hit — per-position SL triggered | 2025-06-09 12:30:00 | 5399.37 | 5400.85 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 5326.50 | 5349.78 | 0.00 | ORB-short ORB[5340.00,5391.00] vol=2.2x ATR=10.45 |
| Stop hit — per-position SL triggered | 2025-06-17 09:35:00 | 5336.95 | 5348.98 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:30:00 | 5455.00 | 5438.83 | 0.00 | ORB-long ORB[5405.00,5452.00] vol=1.5x ATR=13.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:35:00 | 5475.23 | 5456.26 | 0.00 | T1 1.5R @ 5475.23 |
| Target hit | 2025-06-19 15:20:00 | 5495.00 | 5477.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-06-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:55:00 | 5618.50 | 5600.65 | 0.00 | ORB-long ORB[5560.00,5605.50] vol=1.9x ATR=9.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 11:10:00 | 5633.30 | 5603.94 | 0.00 | T1 1.5R @ 5633.30 |
| Stop hit — per-position SL triggered | 2025-06-24 11:15:00 | 5618.50 | 5604.25 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:10:00 | 5542.50 | 5559.82 | 0.00 | ORB-short ORB[5564.00,5595.00] vol=1.9x ATR=10.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:25:00 | 5527.09 | 5549.93 | 0.00 | T1 1.5R @ 5527.09 |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 5542.50 | 5543.80 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 11:15:00 | 5634.50 | 5614.61 | 0.00 | ORB-long ORB[5595.00,5634.00] vol=4.2x ATR=8.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:40:00 | 5647.86 | 5620.85 | 0.00 | T1 1.5R @ 5647.86 |
| Target hit | 2025-06-27 13:40:00 | 5647.00 | 5652.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2025-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 10:50:00 | 5678.50 | 5688.76 | 0.00 | ORB-short ORB[5681.50,5715.50] vol=4.5x ATR=11.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:10:00 | 5661.41 | 5686.10 | 0.00 | T1 1.5R @ 5661.41 |
| Target hit | 2025-07-04 15:20:00 | 5624.50 | 5649.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:00:00 | 5670.50 | 5677.83 | 0.00 | ORB-short ORB[5672.50,5705.00] vol=2.3x ATR=9.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:05:00 | 5656.33 | 5676.77 | 0.00 | T1 1.5R @ 5656.33 |
| Stop hit — per-position SL triggered | 2025-07-10 11:50:00 | 5670.50 | 5670.50 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:25:00 | 5614.50 | 5591.50 | 0.00 | ORB-long ORB[5568.50,5605.00] vol=2.0x ATR=10.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 10:40:00 | 5630.08 | 5595.28 | 0.00 | T1 1.5R @ 5630.08 |
| Stop hit — per-position SL triggered | 2025-07-15 10:45:00 | 5614.50 | 5595.75 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:45:00 | 5547.50 | 5569.68 | 0.00 | ORB-short ORB[5561.00,5620.00] vol=1.6x ATR=9.12 |
| Stop hit — per-position SL triggered | 2025-07-16 10:55:00 | 5556.62 | 5567.34 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:10:00 | 5637.00 | 5650.09 | 0.00 | ORB-short ORB[5653.00,5681.50] vol=3.9x ATR=7.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 11:30:00 | 5626.18 | 5648.87 | 0.00 | T1 1.5R @ 5626.18 |
| Stop hit — per-position SL triggered | 2025-07-18 12:05:00 | 5637.00 | 5645.45 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:10:00 | 5466.50 | 5496.86 | 0.00 | ORB-short ORB[5470.00,5539.00] vol=1.9x ATR=10.04 |
| Stop hit — per-position SL triggered | 2025-07-22 12:25:00 | 5476.54 | 5484.24 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 10:45:00 | 5398.00 | 5427.57 | 0.00 | ORB-short ORB[5406.00,5456.00] vol=1.8x ATR=12.21 |
| Stop hit — per-position SL triggered | 2025-07-28 11:35:00 | 5410.21 | 5416.11 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:55:00 | 5547.50 | 5505.62 | 0.00 | ORB-long ORB[5440.00,5474.50] vol=1.6x ATR=12.94 |
| Stop hit — per-position SL triggered | 2025-07-30 12:05:00 | 5534.56 | 5517.91 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 09:30:00 | 5633.50 | 5586.03 | 0.00 | ORB-long ORB[5520.00,5598.50] vol=1.6x ATR=20.32 |
| Stop hit — per-position SL triggered | 2025-08-04 09:45:00 | 5613.18 | 5600.00 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 10:50:00 | 5628.50 | 5618.56 | 0.00 | ORB-long ORB[5584.50,5625.00] vol=7.3x ATR=12.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 11:00:00 | 5646.76 | 5620.01 | 0.00 | T1 1.5R @ 5646.76 |
| Stop hit — per-position SL triggered | 2025-08-05 11:35:00 | 5628.50 | 5626.07 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:05:00 | 5603.50 | 5647.50 | 0.00 | ORB-short ORB[5648.50,5680.50] vol=1.5x ATR=10.15 |
| Stop hit — per-position SL triggered | 2025-08-07 12:05:00 | 5613.65 | 5641.15 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 11:10:00 | 5670.00 | 5684.94 | 0.00 | ORB-short ORB[5670.50,5708.50] vol=1.5x ATR=8.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:35:00 | 5657.25 | 5680.83 | 0.00 | T1 1.5R @ 5657.25 |
| Stop hit — per-position SL triggered | 2025-08-11 11:45:00 | 5670.00 | 5679.31 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:45:00 | 5695.00 | 5674.17 | 0.00 | ORB-long ORB[5648.50,5677.00] vol=1.6x ATR=10.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:55:00 | 5710.14 | 5692.84 | 0.00 | T1 1.5R @ 5710.14 |
| Target hit | 2025-08-13 12:00:00 | 5718.00 | 5725.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2025-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:45:00 | 5925.00 | 5942.17 | 0.00 | ORB-short ORB[5939.50,5969.50] vol=1.7x ATR=11.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:55:00 | 5907.57 | 5937.77 | 0.00 | T1 1.5R @ 5907.57 |
| Stop hit — per-position SL triggered | 2025-08-20 11:05:00 | 5925.00 | 5931.03 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 11:00:00 | 5914.50 | 5927.44 | 0.00 | ORB-short ORB[5923.00,5992.50] vol=2.1x ATR=8.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 12:10:00 | 5902.41 | 5922.23 | 0.00 | T1 1.5R @ 5902.41 |
| Stop hit — per-position SL triggered | 2025-08-21 13:00:00 | 5914.50 | 5917.85 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 09:55:00 | 6043.50 | 6011.42 | 0.00 | ORB-long ORB[5960.00,6028.00] vol=2.1x ATR=15.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 10:10:00 | 6067.39 | 6030.28 | 0.00 | T1 1.5R @ 6067.39 |
| Target hit | 2025-08-26 15:20:00 | 6144.00 | 6122.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:45:00 | 6528.00 | 6501.08 | 0.00 | ORB-long ORB[6440.50,6504.50] vol=1.6x ATR=15.42 |
| Stop hit — per-position SL triggered | 2025-09-05 10:10:00 | 6512.58 | 6510.17 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 11:00:00 | 6708.50 | 6636.94 | 0.00 | ORB-long ORB[6569.00,6642.00] vol=1.6x ATR=17.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 11:25:00 | 6734.68 | 6647.29 | 0.00 | T1 1.5R @ 6734.68 |
| Target hit | 2025-09-08 15:20:00 | 6816.50 | 6735.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:15:00 | 6777.00 | 6797.26 | 0.00 | ORB-short ORB[6782.50,6819.50] vol=1.9x ATR=14.51 |
| Stop hit — per-position SL triggered | 2025-09-11 11:30:00 | 6791.51 | 6785.01 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 11:10:00 | 6820.00 | 6853.41 | 0.00 | ORB-short ORB[6858.00,6900.50] vol=3.2x ATR=8.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 11:55:00 | 6807.42 | 6840.21 | 0.00 | T1 1.5R @ 6807.42 |
| Target hit | 2025-09-15 15:20:00 | 6799.00 | 6817.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 11:15:00 | 6966.50 | 6944.80 | 0.00 | ORB-long ORB[6925.00,6960.00] vol=2.3x ATR=7.88 |
| Stop hit — per-position SL triggered | 2025-09-19 11:35:00 | 6958.62 | 6947.17 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:35:00 | 7017.50 | 6978.17 | 0.00 | ORB-long ORB[6927.50,6985.50] vol=1.9x ATR=15.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:45:00 | 7040.36 | 6988.76 | 0.00 | T1 1.5R @ 7040.36 |
| Stop hit — per-position SL triggered | 2025-09-24 10:50:00 | 7017.50 | 7008.73 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 7046.50 | 7019.19 | 0.00 | ORB-long ORB[6965.50,7038.00] vol=1.9x ATR=17.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 12:35:00 | 7073.18 | 7045.54 | 0.00 | T1 1.5R @ 7073.18 |
| Stop hit — per-position SL triggered | 2025-09-26 13:15:00 | 7046.50 | 7049.61 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:55:00 | 6954.00 | 6983.54 | 0.00 | ORB-short ORB[6967.50,7027.00] vol=2.1x ATR=12.70 |
| Stop hit — per-position SL triggered | 2025-09-30 11:10:00 | 6966.70 | 6982.12 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:35:00 | 6857.50 | 6872.96 | 0.00 | ORB-short ORB[6861.50,6929.00] vol=1.5x ATR=18.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:40:00 | 6829.48 | 6869.96 | 0.00 | T1 1.5R @ 6829.48 |
| Stop hit — per-position SL triggered | 2025-10-09 10:10:00 | 6857.50 | 6858.68 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 11:00:00 | 6940.00 | 6931.24 | 0.00 | ORB-long ORB[6876.50,6931.00] vol=1.8x ATR=11.64 |
| Stop hit — per-position SL triggered | 2025-10-10 11:20:00 | 6928.36 | 6931.54 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:40:00 | 6857.50 | 6892.66 | 0.00 | ORB-short ORB[6887.00,6942.00] vol=1.9x ATR=15.28 |
| Stop hit — per-position SL triggered | 2025-10-14 09:50:00 | 6872.78 | 6886.11 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:55:00 | 6999.50 | 6981.67 | 0.00 | ORB-long ORB[6892.00,6993.00] vol=1.6x ATR=16.05 |
| Stop hit — per-position SL triggered | 2025-10-16 12:05:00 | 6983.45 | 6990.26 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:10:00 | 7069.50 | 7034.16 | 0.00 | ORB-long ORB[6988.00,7040.00] vol=1.6x ATR=13.76 |
| Stop hit — per-position SL triggered | 2025-10-17 10:20:00 | 7055.74 | 7036.74 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 11:15:00 | 6876.50 | 6864.13 | 0.00 | ORB-long ORB[6842.50,6873.00] vol=2.6x ATR=8.50 |
| Stop hit — per-position SL triggered | 2025-10-27 11:30:00 | 6868.00 | 6864.71 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:40:00 | 6967.50 | 6930.55 | 0.00 | ORB-long ORB[6905.00,6949.50] vol=1.6x ATR=14.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 09:45:00 | 6988.54 | 6938.66 | 0.00 | T1 1.5R @ 6988.54 |
| Stop hit — per-position SL triggered | 2025-10-28 09:55:00 | 6967.50 | 6942.29 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:50:00 | 6920.00 | 6932.88 | 0.00 | ORB-short ORB[6934.50,6988.50] vol=2.4x ATR=15.32 |
| Stop hit — per-position SL triggered | 2025-10-30 11:05:00 | 6935.32 | 6930.13 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 11:05:00 | 6760.00 | 6792.94 | 0.00 | ORB-short ORB[6785.00,6885.00] vol=1.8x ATR=14.20 |
| Stop hit — per-position SL triggered | 2025-11-11 11:35:00 | 6774.20 | 6792.00 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 6853.00 | 6880.02 | 0.00 | ORB-short ORB[6856.00,6925.00] vol=1.6x ATR=14.23 |
| Stop hit — per-position SL triggered | 2025-11-12 10:05:00 | 6867.23 | 6879.21 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:45:00 | 6914.50 | 6897.37 | 0.00 | ORB-long ORB[6860.50,6906.00] vol=1.7x ATR=13.67 |
| Stop hit — per-position SL triggered | 2025-11-13 11:40:00 | 6900.83 | 6908.51 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 11:10:00 | 6773.50 | 6731.44 | 0.00 | ORB-long ORB[6675.00,6740.00] vol=1.5x ATR=12.98 |
| Stop hit — per-position SL triggered | 2025-11-17 11:25:00 | 6760.52 | 6734.05 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:45:00 | 6985.00 | 6940.54 | 0.00 | ORB-long ORB[6886.50,6927.00] vol=2.0x ATR=13.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 10:55:00 | 7005.01 | 6949.32 | 0.00 | T1 1.5R @ 7005.01 |
| Target hit | 2025-11-20 15:20:00 | 7127.00 | 7070.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:30:00 | 7189.00 | 7159.32 | 0.00 | ORB-long ORB[7088.50,7185.50] vol=1.6x ATR=18.71 |
| Stop hit — per-position SL triggered | 2025-11-21 10:10:00 | 7170.29 | 7174.34 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:10:00 | 7096.00 | 7129.34 | 0.00 | ORB-short ORB[7112.00,7210.50] vol=1.7x ATR=17.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:25:00 | 7069.31 | 7121.32 | 0.00 | T1 1.5R @ 7069.31 |
| Target hit | 2025-11-27 15:20:00 | 6997.50 | 7048.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2025-11-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:55:00 | 7031.00 | 7013.42 | 0.00 | ORB-long ORB[6956.00,7019.50] vol=1.5x ATR=13.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 11:45:00 | 7051.70 | 7020.89 | 0.00 | T1 1.5R @ 7051.70 |
| Stop hit — per-position SL triggered | 2025-11-28 12:25:00 | 7031.00 | 7025.77 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:35:00 | 7185.50 | 7167.33 | 0.00 | ORB-long ORB[7129.50,7169.50] vol=1.8x ATR=17.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 09:45:00 | 7211.82 | 7175.82 | 0.00 | T1 1.5R @ 7211.82 |
| Target hit | 2025-12-10 14:25:00 | 7233.00 | 7236.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — BUY (started 2025-12-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:00:00 | 7255.00 | 7252.88 | 0.00 | ORB-long ORB[7220.00,7252.50] vol=3.0x ATR=15.31 |
| Stop hit — per-position SL triggered | 2025-12-12 10:55:00 | 7239.69 | 7254.46 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:05:00 | 7127.00 | 7139.09 | 0.00 | ORB-short ORB[7137.00,7210.50] vol=1.8x ATR=12.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 12:00:00 | 7107.87 | 7133.31 | 0.00 | T1 1.5R @ 7107.87 |
| Target hit | 2025-12-15 15:20:00 | 7119.00 | 7125.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:15:00 | 7076.00 | 7084.37 | 0.00 | ORB-short ORB[7085.00,7121.50] vol=4.7x ATR=10.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 13:00:00 | 7060.92 | 7078.28 | 0.00 | T1 1.5R @ 7060.92 |
| Stop hit — per-position SL triggered | 2025-12-16 13:20:00 | 7076.00 | 7076.18 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:35:00 | 7289.00 | 7261.57 | 0.00 | ORB-long ORB[7190.50,7274.50] vol=2.4x ATR=19.15 |
| Stop hit — per-position SL triggered | 2025-12-30 10:50:00 | 7269.85 | 7266.08 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 09:35:00 | 7313.50 | 7343.99 | 0.00 | ORB-short ORB[7326.00,7377.00] vol=3.1x ATR=17.70 |
| Stop hit — per-position SL triggered | 2026-01-02 09:45:00 | 7331.20 | 7341.09 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-01-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:45:00 | 7421.00 | 7383.74 | 0.00 | ORB-long ORB[7328.00,7374.00] vol=2.8x ATR=15.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 10:10:00 | 7444.89 | 7408.18 | 0.00 | T1 1.5R @ 7444.89 |
| Target hit | 2026-01-05 15:20:00 | 7481.00 | 7465.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2026-01-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:35:00 | 7563.00 | 7526.50 | 0.00 | ORB-long ORB[7484.50,7526.50] vol=1.6x ATR=14.61 |
| Stop hit — per-position SL triggered | 2026-01-06 09:40:00 | 7548.39 | 7528.71 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:35:00 | 7436.00 | 7472.99 | 0.00 | ORB-short ORB[7443.50,7514.50] vol=1.6x ATR=19.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:05:00 | 7407.29 | 7463.33 | 0.00 | T1 1.5R @ 7407.29 |
| Stop hit — per-position SL triggered | 2026-01-13 11:25:00 | 7436.00 | 7459.45 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-01-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:20:00 | 7388.00 | 7348.69 | 0.00 | ORB-long ORB[7319.00,7370.50] vol=1.5x ATR=16.69 |
| Stop hit — per-position SL triggered | 2026-01-16 10:45:00 | 7371.31 | 7353.27 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-01-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 11:00:00 | 7317.50 | 7296.81 | 0.00 | ORB-long ORB[7268.50,7314.00] vol=1.5x ATR=11.91 |
| Stop hit — per-position SL triggered | 2026-01-19 11:30:00 | 7305.59 | 7298.47 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 7465.50 | 7395.50 | 0.00 | ORB-long ORB[7338.50,7437.00] vol=1.6x ATR=24.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:25:00 | 7501.97 | 7423.60 | 0.00 | T1 1.5R @ 7501.97 |
| Stop hit — per-position SL triggered | 2026-03-10 11:45:00 | 7465.50 | 7432.53 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 7422.00 | 7491.43 | 0.00 | ORB-short ORB[7480.00,7574.00] vol=1.8x ATR=22.46 |
| Stop hit — per-position SL triggered | 2026-03-11 12:00:00 | 7444.46 | 7473.56 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-03-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:00:00 | 6944.50 | 6901.98 | 0.00 | ORB-long ORB[6770.00,6837.50] vol=1.5x ATR=26.60 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 6917.90 | 6903.74 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-03-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:55:00 | 6844.50 | 6872.69 | 0.00 | ORB-short ORB[6870.50,6916.50] vol=2.0x ATR=20.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:15:00 | 6813.27 | 6863.55 | 0.00 | T1 1.5R @ 6813.27 |
| Target hit | 2026-03-27 12:50:00 | 6831.50 | 6815.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 7335.00 | 7280.84 | 0.00 | ORB-long ORB[7201.00,7310.00] vol=1.6x ATR=32.72 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 7302.28 | 7297.97 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-04-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:20:00 | 7113.00 | 7126.22 | 0.00 | ORB-short ORB[7115.00,7199.50] vol=2.9x ATR=20.05 |
| Stop hit — per-position SL triggered | 2026-04-15 10:25:00 | 7133.05 | 7126.35 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:15:00 | 7280.00 | 7264.16 | 0.00 | ORB-long ORB[7235.00,7276.00] vol=5.4x ATR=10.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 12:25:00 | 7295.41 | 7273.13 | 0.00 | T1 1.5R @ 7295.41 |
| Stop hit — per-position SL triggered | 2026-04-21 12:35:00 | 7280.00 | 7273.52 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-04-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:35:00 | 7106.50 | 7150.81 | 0.00 | ORB-short ORB[7136.50,7199.00] vol=1.7x ATR=18.66 |
| Stop hit — per-position SL triggered | 2026-04-27 10:45:00 | 7125.16 | 7148.89 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 7232.00 | 7277.25 | 0.00 | ORB-short ORB[7246.00,7330.00] vol=1.5x ATR=19.07 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 7251.07 | 7275.91 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-05-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:50:00 | 7265.00 | 7341.43 | 0.00 | ORB-short ORB[7324.50,7392.00] vol=3.1x ATR=20.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:55:00 | 7234.13 | 7330.62 | 0.00 | T1 1.5R @ 7234.13 |
| Stop hit — per-position SL triggered | 2026-05-06 13:05:00 | 7265.00 | 7283.91 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 7358.00 | 7299.77 | 0.00 | ORB-long ORB[7228.00,7312.00] vol=2.3x ATR=15.65 |
| Stop hit — per-position SL triggered | 2026-05-08 12:05:00 | 7342.35 | 7314.01 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-13 11:10:00 | 5439.50 | 2025-05-13 11:50:00 | 5452.47 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-05-19 09:30:00 | 5577.50 | 2025-05-19 09:35:00 | 5565.06 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-05-23 11:05:00 | 5431.50 | 2025-05-23 11:10:00 | 5417.14 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-05-27 09:50:00 | 5382.50 | 2025-05-27 10:10:00 | 5393.33 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-06-09 10:25:00 | 5389.00 | 2025-06-09 12:30:00 | 5399.37 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-06-17 09:30:00 | 5326.50 | 2025-06-17 09:35:00 | 5336.95 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-06-19 09:30:00 | 5455.00 | 2025-06-19 11:35:00 | 5475.23 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-06-19 09:30:00 | 5455.00 | 2025-06-19 15:20:00 | 5495.00 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2025-06-24 10:55:00 | 5618.50 | 2025-06-24 11:10:00 | 5633.30 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-06-24 10:55:00 | 5618.50 | 2025-06-24 11:15:00 | 5618.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 11:10:00 | 5542.50 | 2025-06-26 11:25:00 | 5527.09 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-06-26 11:10:00 | 5542.50 | 2025-06-26 12:15:00 | 5542.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 11:15:00 | 5634.50 | 2025-06-27 11:40:00 | 5647.86 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-06-27 11:15:00 | 5634.50 | 2025-06-27 13:40:00 | 5647.00 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2025-07-04 10:50:00 | 5678.50 | 2025-07-04 11:10:00 | 5661.41 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-04 10:50:00 | 5678.50 | 2025-07-04 15:20:00 | 5624.50 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2025-07-10 11:00:00 | 5670.50 | 2025-07-10 11:05:00 | 5656.33 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-10 11:00:00 | 5670.50 | 2025-07-10 11:50:00 | 5670.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 10:25:00 | 5614.50 | 2025-07-15 10:40:00 | 5630.08 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-07-15 10:25:00 | 5614.50 | 2025-07-15 10:45:00 | 5614.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-16 10:45:00 | 5547.50 | 2025-07-16 10:55:00 | 5556.62 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-18 11:10:00 | 5637.00 | 2025-07-18 11:30:00 | 5626.18 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2025-07-18 11:10:00 | 5637.00 | 2025-07-18 12:05:00 | 5637.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 11:10:00 | 5466.50 | 2025-07-22 12:25:00 | 5476.54 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-28 10:45:00 | 5398.00 | 2025-07-28 11:35:00 | 5410.21 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-30 10:55:00 | 5547.50 | 2025-07-30 12:05:00 | 5534.56 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-04 09:30:00 | 5633.50 | 2025-08-04 09:45:00 | 5613.18 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-08-05 10:50:00 | 5628.50 | 2025-08-05 11:00:00 | 5646.76 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-08-05 10:50:00 | 5628.50 | 2025-08-05 11:35:00 | 5628.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 11:05:00 | 5603.50 | 2025-08-07 12:05:00 | 5613.65 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-11 11:10:00 | 5670.00 | 2025-08-11 11:35:00 | 5657.25 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-08-11 11:10:00 | 5670.00 | 2025-08-11 11:45:00 | 5670.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-13 09:45:00 | 5695.00 | 2025-08-13 09:55:00 | 5710.14 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-08-13 09:45:00 | 5695.00 | 2025-08-13 12:00:00 | 5718.00 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-20 09:45:00 | 5925.00 | 2025-08-20 09:55:00 | 5907.57 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-08-20 09:45:00 | 5925.00 | 2025-08-20 11:05:00 | 5925.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-21 11:00:00 | 5914.50 | 2025-08-21 12:10:00 | 5902.41 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-08-21 11:00:00 | 5914.50 | 2025-08-21 13:00:00 | 5914.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-26 09:55:00 | 6043.50 | 2025-08-26 10:10:00 | 6067.39 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-08-26 09:55:00 | 6043.50 | 2025-08-26 15:20:00 | 6144.00 | TARGET_HIT | 0.50 | 1.66% |
| BUY | retest1 | 2025-09-05 09:45:00 | 6528.00 | 2025-09-05 10:10:00 | 6512.58 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-08 11:00:00 | 6708.50 | 2025-09-08 11:25:00 | 6734.68 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-08 11:00:00 | 6708.50 | 2025-09-08 15:20:00 | 6816.50 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2025-09-11 10:15:00 | 6777.00 | 2025-09-11 11:30:00 | 6791.51 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-15 11:10:00 | 6820.00 | 2025-09-15 11:55:00 | 6807.42 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-09-15 11:10:00 | 6820.00 | 2025-09-15 15:20:00 | 6799.00 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2025-09-19 11:15:00 | 6966.50 | 2025-09-19 11:35:00 | 6958.62 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2025-09-24 09:35:00 | 7017.50 | 2025-09-24 09:45:00 | 7040.36 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-24 09:35:00 | 7017.50 | 2025-09-24 10:50:00 | 7017.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-26 09:30:00 | 7046.50 | 2025-09-26 12:35:00 | 7073.18 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-09-26 09:30:00 | 7046.50 | 2025-09-26 13:15:00 | 7046.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-30 10:55:00 | 6954.00 | 2025-09-30 11:10:00 | 6966.70 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-09 09:35:00 | 6857.50 | 2025-10-09 09:40:00 | 6829.48 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-09 09:35:00 | 6857.50 | 2025-10-09 10:10:00 | 6857.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 11:00:00 | 6940.00 | 2025-10-10 11:20:00 | 6928.36 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-10-14 09:40:00 | 6857.50 | 2025-10-14 09:50:00 | 6872.78 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-16 09:55:00 | 6999.50 | 2025-10-16 12:05:00 | 6983.45 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-17 10:10:00 | 7069.50 | 2025-10-17 10:20:00 | 7055.74 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-27 11:15:00 | 6876.50 | 2025-10-27 11:30:00 | 6868.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-10-28 09:40:00 | 6967.50 | 2025-10-28 09:45:00 | 6988.54 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-10-28 09:40:00 | 6967.50 | 2025-10-28 09:55:00 | 6967.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 10:50:00 | 6920.00 | 2025-10-30 11:05:00 | 6935.32 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-11 11:05:00 | 6760.00 | 2025-11-11 11:35:00 | 6774.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-12 10:00:00 | 6853.00 | 2025-11-12 10:05:00 | 6867.23 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-13 09:45:00 | 6914.50 | 2025-11-13 11:40:00 | 6900.83 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-17 11:10:00 | 6773.50 | 2025-11-17 11:25:00 | 6760.52 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-20 10:45:00 | 6985.00 | 2025-11-20 10:55:00 | 7005.01 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-11-20 10:45:00 | 6985.00 | 2025-11-20 15:20:00 | 7127.00 | TARGET_HIT | 0.50 | 2.03% |
| BUY | retest1 | 2025-11-21 09:30:00 | 7189.00 | 2025-11-21 10:10:00 | 7170.29 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-27 10:10:00 | 7096.00 | 2025-11-27 10:25:00 | 7069.31 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-11-27 10:10:00 | 7096.00 | 2025-11-27 15:20:00 | 6997.50 | TARGET_HIT | 0.50 | 1.39% |
| BUY | retest1 | 2025-11-28 10:55:00 | 7031.00 | 2025-11-28 11:45:00 | 7051.70 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-11-28 10:55:00 | 7031.00 | 2025-11-28 12:25:00 | 7031.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-10 09:35:00 | 7185.50 | 2025-12-10 09:45:00 | 7211.82 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-10 09:35:00 | 7185.50 | 2025-12-10 14:25:00 | 7233.00 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2025-12-12 10:00:00 | 7255.00 | 2025-12-12 10:55:00 | 7239.69 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-15 11:05:00 | 7127.00 | 2025-12-15 12:00:00 | 7107.87 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-15 11:05:00 | 7127.00 | 2025-12-15 15:20:00 | 7119.00 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2025-12-16 11:15:00 | 7076.00 | 2025-12-16 13:00:00 | 7060.92 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2025-12-16 11:15:00 | 7076.00 | 2025-12-16 13:20:00 | 7076.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 10:35:00 | 7289.00 | 2025-12-30 10:50:00 | 7269.85 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-02 09:35:00 | 7313.50 | 2026-01-02 09:45:00 | 7331.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-01-05 09:45:00 | 7421.00 | 2026-01-05 10:10:00 | 7444.89 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-01-05 09:45:00 | 7421.00 | 2026-01-05 15:20:00 | 7481.00 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2026-01-06 09:35:00 | 7563.00 | 2026-01-06 09:40:00 | 7548.39 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-13 10:35:00 | 7436.00 | 2026-01-13 11:05:00 | 7407.29 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-01-13 10:35:00 | 7436.00 | 2026-01-13 11:25:00 | 7436.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-16 10:20:00 | 7388.00 | 2026-01-16 10:45:00 | 7371.31 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-19 11:00:00 | 7317.50 | 2026-01-19 11:30:00 | 7305.59 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-03-10 10:40:00 | 7465.50 | 2026-03-10 11:25:00 | 7501.97 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-10 10:40:00 | 7465.50 | 2026-03-10 11:45:00 | 7465.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 11:10:00 | 7422.00 | 2026-03-11 12:00:00 | 7444.46 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-17 11:00:00 | 6944.50 | 2026-03-17 11:15:00 | 6917.90 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-27 09:55:00 | 6844.50 | 2026-03-27 10:15:00 | 6813.27 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-27 09:55:00 | 6844.50 | 2026-03-27 12:50:00 | 6831.50 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2026-04-10 09:45:00 | 7335.00 | 2026-04-10 10:05:00 | 7302.28 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-15 10:20:00 | 7113.00 | 2026-04-15 10:25:00 | 7133.05 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-21 11:15:00 | 7280.00 | 2026-04-21 12:25:00 | 7295.41 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2026-04-21 11:15:00 | 7280.00 | 2026-04-21 12:35:00 | 7280.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-27 10:35:00 | 7106.50 | 2026-04-27 10:45:00 | 7125.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-05 10:55:00 | 7232.00 | 2026-05-05 11:05:00 | 7251.07 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-06 10:50:00 | 7265.00 | 2026-05-06 10:55:00 | 7234.13 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-05-06 10:50:00 | 7265.00 | 2026-05-06 13:05:00 | 7265.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 11:00:00 | 7358.00 | 2026-05-08 12:05:00 | 7342.35 | STOP_HIT | 1.00 | -0.21% |
