# Sundaram Finance Ltd. (SUNDARMFIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4700.10
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
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 10
- **Target hits / Stop hits / Partials:** 3 / 10 / 6
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 2.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.17% | 1.2% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.17% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.12% | 1.4% |
| SELL @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.12% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 9 | 47.4% | 3 | 10 | 6 | 0.14% | 2.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:05:00 | 5372.50 | 5386.92 | 0.00 | ORB-short ORB[5378.50,5450.50] vol=1.7x ATR=18.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:15:00 | 5345.16 | 5383.68 | 0.00 | T1 1.5R @ 5345.16 |
| Stop hit — per-position SL triggered | 2026-02-10 11:25:00 | 5372.50 | 5380.89 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 5234.00 | 5201.18 | 0.00 | ORB-long ORB[5152.50,5212.00] vol=1.5x ATR=15.41 |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 5218.59 | 5205.98 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:25:00 | 4906.00 | 4969.73 | 0.00 | ORB-short ORB[4938.50,5008.50] vol=1.7x ATR=21.62 |
| Stop hit — per-position SL triggered | 2026-03-18 10:55:00 | 4927.62 | 4951.98 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:45:00 | 4912.50 | 4937.88 | 0.00 | ORB-short ORB[4935.50,4988.50] vol=1.5x ATR=22.31 |
| Stop hit — per-position SL triggered | 2026-03-19 09:55:00 | 4934.81 | 4935.50 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:25:00 | 4809.40 | 4834.07 | 0.00 | ORB-short ORB[4835.90,4902.90] vol=2.5x ATR=13.54 |
| Stop hit — per-position SL triggered | 2026-04-15 10:30:00 | 4822.94 | 4833.75 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 4991.80 | 4966.51 | 0.00 | ORB-long ORB[4914.00,4985.90] vol=2.2x ATR=20.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:10:00 | 5022.55 | 4990.93 | 0.00 | T1 1.5R @ 5022.55 |
| Target hit | 2026-04-16 11:00:00 | 5001.90 | 5002.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-04-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:10:00 | 5019.50 | 4997.57 | 0.00 | ORB-long ORB[4940.00,4993.30] vol=1.7x ATR=14.05 |
| Stop hit — per-position SL triggered | 2026-04-17 10:20:00 | 5005.45 | 4998.32 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:40:00 | 4925.60 | 4937.46 | 0.00 | ORB-short ORB[4934.10,4984.40] vol=1.6x ATR=12.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 12:30:00 | 4907.06 | 4929.06 | 0.00 | T1 1.5R @ 4907.06 |
| Stop hit — per-position SL triggered | 2026-04-22 13:00:00 | 4925.60 | 4928.66 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:45:00 | 4836.80 | 4870.47 | 0.00 | ORB-short ORB[4852.80,4915.90] vol=1.6x ATR=16.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 12:40:00 | 4811.90 | 4858.49 | 0.00 | T1 1.5R @ 4811.90 |
| Stop hit — per-position SL triggered | 2026-04-23 12:50:00 | 4836.80 | 4857.64 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 4695.00 | 4731.85 | 0.00 | ORB-short ORB[4715.00,4775.50] vol=2.1x ATR=24.89 |
| Stop hit — per-position SL triggered | 2026-04-27 09:45:00 | 4719.89 | 4724.21 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 4640.00 | 4668.91 | 0.00 | ORB-short ORB[4652.20,4712.60] vol=2.8x ATR=12.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:15:00 | 4621.18 | 4665.04 | 0.00 | T1 1.5R @ 4621.18 |
| Target hit | 2026-04-28 15:20:00 | 4577.50 | 4611.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:00:00 | 4660.80 | 4648.62 | 0.00 | ORB-long ORB[4602.50,4648.00] vol=1.5x ATR=17.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 13:55:00 | 4687.31 | 4661.54 | 0.00 | T1 1.5R @ 4687.31 |
| Target hit | 2026-05-06 15:20:00 | 4692.80 | 4684.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-05-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:40:00 | 4750.00 | 4727.70 | 0.00 | ORB-long ORB[4704.50,4738.00] vol=2.3x ATR=14.86 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 4735.14 | 4730.48 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 11:05:00 | 5372.50 | 2026-02-10 11:15:00 | 5345.16 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-10 11:05:00 | 5372.50 | 2026-02-10 11:25:00 | 5372.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 10:50:00 | 5234.00 | 2026-02-23 11:15:00 | 5218.59 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-18 10:25:00 | 4906.00 | 2026-03-18 10:55:00 | 4927.62 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-19 09:45:00 | 4912.50 | 2026-03-19 09:55:00 | 4934.81 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-15 10:25:00 | 4809.40 | 2026-04-15 10:30:00 | 4822.94 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-16 09:50:00 | 4991.80 | 2026-04-16 10:10:00 | 5022.55 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-16 09:50:00 | 4991.80 | 2026-04-16 11:00:00 | 5001.90 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2026-04-17 10:10:00 | 5019.50 | 2026-04-17 10:20:00 | 5005.45 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-22 10:40:00 | 4925.60 | 2026-04-22 12:30:00 | 4907.06 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-22 10:40:00 | 4925.60 | 2026-04-22 13:00:00 | 4925.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 10:45:00 | 4836.80 | 2026-04-23 12:40:00 | 4811.90 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-23 10:45:00 | 4836.80 | 2026-04-23 12:50:00 | 4836.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-27 09:30:00 | 4695.00 | 2026-04-27 09:45:00 | 4719.89 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-04-28 11:05:00 | 4640.00 | 2026-04-28 11:15:00 | 4621.18 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-28 11:05:00 | 4640.00 | 2026-04-28 15:20:00 | 4577.50 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2026-05-06 10:00:00 | 4660.80 | 2026-05-06 13:55:00 | 4687.31 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-05-06 10:00:00 | 4660.80 | 2026-05-06 15:20:00 | 4692.80 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2026-05-08 10:40:00 | 4750.00 | 2026-05-08 11:15:00 | 4735.14 | STOP_HIT | 1.00 | -0.31% |
