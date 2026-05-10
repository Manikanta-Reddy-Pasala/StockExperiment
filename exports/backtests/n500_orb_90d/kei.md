# KEI Industries Ltd. (KEI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 5117.00
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 0 / 10 / 2
- **Avg / median % per leg:** -0.16% / -0.32%
- **Sum % (uncompounded):** -1.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.40% | -2.0% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.40% | -2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.02% | 0.1% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.02% | 0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 2 | 16.7% | 0 | 10 | 2 | -0.16% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:45:00 | 4582.70 | 4547.64 | 0.00 | ORB-long ORB[4455.70,4512.30] vol=2.2x ATR=21.82 |
| Stop hit — per-position SL triggered | 2026-02-09 11:20:00 | 4560.88 | 4550.42 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:15:00 | 4517.00 | 4538.63 | 0.00 | ORB-short ORB[4527.60,4589.10] vol=1.5x ATR=14.53 |
| Stop hit — per-position SL triggered | 2026-02-13 10:30:00 | 4531.53 | 4537.17 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 4745.00 | 4802.65 | 0.00 | ORB-short ORB[4760.00,4822.20] vol=2.5x ATR=21.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:10:00 | 4712.44 | 4790.23 | 0.00 | T1 1.5R @ 4712.44 |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 4745.00 | 4787.54 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 4932.60 | 4876.64 | 0.00 | ORB-long ORB[4796.50,4865.00] vol=2.6x ATR=15.29 |
| Stop hit — per-position SL triggered | 2026-02-25 11:00:00 | 4917.31 | 4886.16 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 4956.00 | 4986.53 | 0.00 | ORB-short ORB[4992.00,5042.00] vol=2.2x ATR=19.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 4927.11 | 4971.62 | 0.00 | T1 1.5R @ 4927.11 |
| Stop hit — per-position SL triggered | 2026-03-05 14:40:00 | 4956.00 | 4928.99 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:50:00 | 4358.00 | 4431.13 | 0.00 | ORB-short ORB[4495.00,4554.00] vol=5.3x ATR=21.36 |
| Stop hit — per-position SL triggered | 2026-03-11 10:55:00 | 4379.36 | 4423.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:55:00 | 4357.00 | 4316.81 | 0.00 | ORB-long ORB[4265.00,4327.50] vol=1.5x ATR=22.33 |
| Stop hit — per-position SL triggered | 2026-03-12 10:00:00 | 4334.67 | 4319.53 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:05:00 | 4301.00 | 4266.58 | 0.00 | ORB-long ORB[4230.00,4279.50] vol=2.2x ATR=15.17 |
| Stop hit — per-position SL triggered | 2026-03-18 12:00:00 | 4285.83 | 4277.47 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:30:00 | 4676.00 | 4634.61 | 0.00 | ORB-long ORB[4607.00,4659.30] vol=1.9x ATR=16.94 |
| Stop hit — per-position SL triggered | 2026-04-16 10:35:00 | 4659.06 | 4635.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 4781.70 | 4838.48 | 0.00 | ORB-short ORB[4828.20,4887.40] vol=1.7x ATR=15.53 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 4797.23 | 4835.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:45:00 | 4582.70 | 2026-02-09 11:20:00 | 4560.88 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-02-13 10:15:00 | 4517.00 | 2026-02-13 10:30:00 | 4531.53 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-23 11:00:00 | 4745.00 | 2026-02-23 11:10:00 | 4712.44 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-02-23 11:00:00 | 4745.00 | 2026-02-23 11:15:00 | 4745.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:45:00 | 4932.60 | 2026-02-25 11:00:00 | 4917.31 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-05 10:40:00 | 4956.00 | 2026-03-05 11:25:00 | 4927.11 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-05 10:40:00 | 4956.00 | 2026-03-05 14:40:00 | 4956.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:50:00 | 4358.00 | 2026-03-11 10:55:00 | 4379.36 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-03-12 09:55:00 | 4357.00 | 2026-03-12 10:00:00 | 4334.67 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-03-18 11:05:00 | 4301.00 | 2026-03-18 12:00:00 | 4285.83 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-16 10:30:00 | 4676.00 | 2026-04-16 10:35:00 | 4659.06 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-24 11:10:00 | 4781.70 | 2026-04-24 11:30:00 | 4797.23 | STOP_HIT | 1.00 | -0.32% |
