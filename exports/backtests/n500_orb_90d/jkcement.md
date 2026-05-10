# J.K. Cement Ltd. (JKCEMENT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 5555.50
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 6 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 7
- **Target hits / Stop hits / Partials:** 6 / 7 / 9
- **Avg / median % per leg:** 0.47% / 0.43%
- **Sum % (uncompounded):** 10.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 7 | 77.8% | 3 | 2 | 4 | 0.72% | 6.5% |
| BUY @ 2nd Alert (retest1) | 9 | 7 | 77.8% | 3 | 2 | 4 | 0.72% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 8 | 61.5% | 3 | 5 | 5 | 0.29% | 3.8% |
| SELL @ 2nd Alert (retest1) | 13 | 8 | 61.5% | 3 | 5 | 5 | 0.29% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 15 | 68.2% | 6 | 7 | 9 | 0.47% | 10.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 5902.50 | 5817.37 | 0.00 | ORB-long ORB[5715.50,5795.00] vol=2.0x ATR=40.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 14:30:00 | 5963.04 | 5874.28 | 0.00 | T1 1.5R @ 5963.04 |
| Target hit | 2026-02-09 15:20:00 | 5951.50 | 5903.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 5916.50 | 5928.91 | 0.00 | ORB-short ORB[5928.50,5988.50] vol=2.5x ATR=12.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:40:00 | 5897.67 | 5926.08 | 0.00 | T1 1.5R @ 5897.67 |
| Target hit | 2026-02-10 15:20:00 | 5815.50 | 5886.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:15:00 | 5759.00 | 5710.86 | 0.00 | ORB-long ORB[5653.00,5689.00] vol=2.0x ATR=16.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 5783.50 | 5732.96 | 0.00 | T1 1.5R @ 5783.50 |
| Target hit | 2026-02-17 15:20:00 | 5864.50 | 5814.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:10:00 | 5698.50 | 5729.17 | 0.00 | ORB-short ORB[5723.00,5778.50] vol=1.5x ATR=15.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:25:00 | 5675.57 | 5720.09 | 0.00 | T1 1.5R @ 5675.57 |
| Target hit | 2026-02-24 12:25:00 | 5646.50 | 5624.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 5715.00 | 5667.61 | 0.00 | ORB-long ORB[5623.00,5676.00] vol=3.8x ATR=16.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:45:00 | 5739.34 | 5705.21 | 0.00 | T1 1.5R @ 5739.34 |
| Stop hit — per-position SL triggered | 2026-02-26 12:20:00 | 5715.00 | 5707.86 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:45:00 | 5540.50 | 5554.39 | 0.00 | ORB-short ORB[5541.00,5600.00] vol=1.7x ATR=16.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:35:00 | 5515.56 | 5548.86 | 0.00 | T1 1.5R @ 5515.56 |
| Target hit | 2026-03-04 13:35:00 | 5533.00 | 5522.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 5466.50 | 5515.55 | 0.00 | ORB-short ORB[5486.50,5557.50] vol=2.0x ATR=18.09 |
| Stop hit — per-position SL triggered | 2026-03-06 14:00:00 | 5484.59 | 5486.55 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:35:00 | 4886.00 | 4953.11 | 0.00 | ORB-short ORB[4935.00,5000.00] vol=3.3x ATR=19.26 |
| Stop hit — per-position SL triggered | 2026-03-16 10:40:00 | 4905.26 | 4945.26 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 10:50:00 | 5265.00 | 5279.99 | 0.00 | ORB-short ORB[5270.00,5318.50] vol=3.2x ATR=18.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 11:00:00 | 5237.97 | 5276.96 | 0.00 | T1 1.5R @ 5237.97 |
| Stop hit — per-position SL triggered | 2026-04-07 13:30:00 | 5265.00 | 5264.14 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 5494.00 | 5464.00 | 0.00 | ORB-long ORB[5440.00,5481.50] vol=1.6x ATR=19.88 |
| Stop hit — per-position SL triggered | 2026-04-16 10:20:00 | 5474.12 | 5469.12 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 5835.00 | 5762.89 | 0.00 | ORB-long ORB[5703.00,5788.00] vol=1.9x ATR=28.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:45:00 | 5878.33 | 5793.71 | 0.00 | T1 1.5R @ 5878.33 |
| Target hit | 2026-04-21 15:20:00 | 5925.50 | 5873.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 5822.50 | 5842.34 | 0.00 | ORB-short ORB[5830.00,5898.00] vol=1.7x ATR=20.89 |
| Stop hit — per-position SL triggered | 2026-04-23 09:50:00 | 5843.39 | 5836.16 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 5287.50 | 5310.43 | 0.00 | ORB-short ORB[5289.50,5359.00] vol=1.7x ATR=15.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:45:00 | 5264.71 | 5302.16 | 0.00 | T1 1.5R @ 5264.71 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 5287.50 | 5299.67 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 5902.50 | 2026-02-09 14:30:00 | 5963.04 | PARTIAL | 0.50 | 1.03% |
| BUY | retest1 | 2026-02-09 10:25:00 | 5902.50 | 2026-02-09 15:20:00 | 5951.50 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2026-02-10 10:55:00 | 5916.50 | 2026-02-10 11:40:00 | 5897.67 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-10 10:55:00 | 5916.50 | 2026-02-10 15:20:00 | 5815.50 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2026-02-17 10:15:00 | 5759.00 | 2026-02-17 10:30:00 | 5783.50 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-17 10:15:00 | 5759.00 | 2026-02-17 15:20:00 | 5864.50 | TARGET_HIT | 0.50 | 1.83% |
| SELL | retest1 | 2026-02-24 10:10:00 | 5698.50 | 2026-02-24 10:25:00 | 5675.57 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-24 10:10:00 | 5698.50 | 2026-02-24 12:25:00 | 5646.50 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2026-02-26 10:50:00 | 5715.00 | 2026-02-26 11:45:00 | 5739.34 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-26 10:50:00 | 5715.00 | 2026-02-26 12:20:00 | 5715.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 09:45:00 | 5540.50 | 2026-03-04 10:35:00 | 5515.56 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-04 09:45:00 | 5540.50 | 2026-03-04 13:35:00 | 5533.00 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-03-06 10:30:00 | 5466.50 | 2026-03-06 14:00:00 | 5484.59 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-16 10:35:00 | 4886.00 | 2026-03-16 10:40:00 | 4905.26 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-07 10:50:00 | 5265.00 | 2026-04-07 11:00:00 | 5237.97 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-07 10:50:00 | 5265.00 | 2026-04-07 13:30:00 | 5265.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 09:55:00 | 5494.00 | 2026-04-16 10:20:00 | 5474.12 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-21 10:10:00 | 5835.00 | 2026-04-21 10:45:00 | 5878.33 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-04-21 10:10:00 | 5835.00 | 2026-04-21 15:20:00 | 5925.50 | TARGET_HIT | 0.50 | 1.55% |
| SELL | retest1 | 2026-04-23 09:30:00 | 5822.50 | 2026-04-23 09:50:00 | 5843.39 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-05 11:05:00 | 5287.50 | 2026-05-05 11:45:00 | 5264.71 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-05-05 11:05:00 | 5287.50 | 2026-05-05 12:15:00 | 5287.50 | STOP_HIT | 0.50 | 0.00% |
