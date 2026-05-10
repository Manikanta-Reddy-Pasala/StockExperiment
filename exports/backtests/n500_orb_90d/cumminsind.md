# Cummins India Ltd. (CUMMINSIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 5391.00
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 2
- **Avg / median % per leg:** -0.19% / -0.31%
- **Sum % (uncompounded):** -2.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.17% | -1.2% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.17% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.20% | -1.2% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.20% | -1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 2 | 15.4% | 0 | 11 | 2 | -0.19% | -2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:50:00 | 4482.00 | 4451.69 | 0.00 | ORB-long ORB[4413.60,4464.90] vol=3.7x ATR=12.28 |
| Stop hit — per-position SL triggered | 2026-02-13 11:00:00 | 4469.72 | 4453.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 4570.00 | 4534.83 | 0.00 | ORB-long ORB[4507.50,4555.40] vol=2.9x ATR=14.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:15:00 | 4591.39 | 4552.02 | 0.00 | T1 1.5R @ 4591.39 |
| Stop hit — per-position SL triggered | 2026-02-17 11:35:00 | 4570.00 | 4554.73 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:30:00 | 4838.70 | 4829.75 | 0.00 | ORB-long ORB[4769.90,4816.10] vol=2.8x ATR=20.89 |
| Stop hit — per-position SL triggered | 2026-02-23 10:40:00 | 4817.81 | 4830.02 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:25:00 | 4897.00 | 4918.54 | 0.00 | ORB-short ORB[4923.50,4984.90] vol=3.5x ATR=11.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:50:00 | 4879.15 | 4913.91 | 0.00 | T1 1.5R @ 4879.15 |
| Stop hit — per-position SL triggered | 2026-02-27 11:40:00 | 4897.00 | 4905.56 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:55:00 | 4677.50 | 4699.55 | 0.00 | ORB-short ORB[4678.00,4732.80] vol=3.7x ATR=21.01 |
| Stop hit — per-position SL triggered | 2026-03-13 10:00:00 | 4698.51 | 4698.63 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:55:00 | 4539.50 | 4614.80 | 0.00 | ORB-short ORB[4626.90,4689.20] vol=1.5x ATR=19.64 |
| Stop hit — per-position SL triggered | 2026-03-16 11:05:00 | 4559.14 | 4607.84 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:25:00 | 5194.10 | 5122.50 | 0.00 | ORB-long ORB[5079.40,5132.10] vol=2.0x ATR=20.36 |
| Stop hit — per-position SL triggered | 2026-04-22 10:30:00 | 5173.74 | 5125.74 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 5256.70 | 5225.44 | 0.00 | ORB-long ORB[5170.20,5228.10] vol=3.6x ATR=17.60 |
| Stop hit — per-position SL triggered | 2026-04-23 10:35:00 | 5239.10 | 5230.90 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:00:00 | 5118.50 | 5182.35 | 0.00 | ORB-short ORB[5179.00,5224.50] vol=1.9x ATR=15.91 |
| Stop hit — per-position SL triggered | 2026-04-24 11:25:00 | 5134.41 | 5167.30 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:10:00 | 5227.80 | 5269.91 | 0.00 | ORB-short ORB[5260.00,5308.70] vol=1.5x ATR=19.70 |
| Stop hit — per-position SL triggered | 2026-04-27 10:45:00 | 5247.50 | 5264.68 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:40:00 | 5314.10 | 5287.04 | 0.00 | ORB-long ORB[5251.30,5311.00] vol=2.7x ATR=12.98 |
| Stop hit — per-position SL triggered | 2026-04-29 10:45:00 | 5301.12 | 5287.87 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 10:50:00 | 4482.00 | 2026-02-13 11:00:00 | 4469.72 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-17 10:45:00 | 4570.00 | 2026-02-17 11:15:00 | 4591.39 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-17 10:45:00 | 4570.00 | 2026-02-17 11:35:00 | 4570.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 10:30:00 | 4838.70 | 2026-02-23 10:40:00 | 4817.81 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-27 10:25:00 | 4897.00 | 2026-02-27 10:50:00 | 4879.15 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-27 10:25:00 | 4897.00 | 2026-02-27 11:40:00 | 4897.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:55:00 | 4677.50 | 2026-03-13 10:00:00 | 4698.51 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-03-16 10:55:00 | 4539.50 | 2026-03-16 11:05:00 | 4559.14 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-22 10:25:00 | 5194.10 | 2026-04-22 10:30:00 | 5173.74 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-23 10:00:00 | 5256.70 | 2026-04-23 10:35:00 | 5239.10 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-24 11:00:00 | 5118.50 | 2026-04-24 11:25:00 | 5134.41 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-27 10:10:00 | 5227.80 | 2026-04-27 10:45:00 | 5247.50 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-29 10:40:00 | 5314.10 | 2026-04-29 10:45:00 | 5301.12 | STOP_HIT | 1.00 | -0.24% |
