# JSW Energy Ltd. (JSWENERGY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 573.00
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 8
- **Avg / median % per leg:** 0.64% / 0.44%
- **Sum % (uncompounded):** 12.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 1.38% | 11.0% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 1.38% | 11.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 5 | 41.7% | 0 | 7 | 5 | 0.15% | 1.8% |
| SELL @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 0 | 7 | 5 | 0.15% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 11 | 55.0% | 3 | 9 | 8 | 0.64% | 12.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 480.70 | 482.58 | 0.00 | ORB-short ORB[481.00,485.45] vol=2.6x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:10:00 | 478.70 | 481.23 | 0.00 | T1 1.5R @ 478.70 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 480.70 | 481.18 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 485.25 | 487.82 | 0.00 | ORB-short ORB[486.15,490.85] vol=1.5x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-02-18 14:55:00 | 486.65 | 486.45 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 490.00 | 494.41 | 0.00 | ORB-short ORB[492.05,498.95] vol=2.0x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:25:00 | 487.86 | 493.73 | 0.00 | T1 1.5R @ 487.86 |
| Stop hit — per-position SL triggered | 2026-02-23 11:40:00 | 490.00 | 493.52 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 494.75 | 493.05 | 0.00 | ORB-long ORB[490.70,493.75] vol=1.9x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:55:00 | 496.96 | 496.20 | 0.00 | T1 1.5R @ 496.96 |
| Target hit | 2026-02-26 11:45:00 | 500.30 | 500.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 477.20 | 472.51 | 0.00 | ORB-long ORB[469.85,473.20] vol=2.7x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-03-05 11:20:00 | 475.96 | 472.75 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 492.50 | 488.97 | 0.00 | ORB-long ORB[484.00,491.25] vol=2.4x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:40:00 | 495.86 | 491.53 | 0.00 | T1 1.5R @ 495.86 |
| Target hit | 2026-03-12 15:10:00 | 518.40 | 518.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 11:15:00 | 494.80 | 500.36 | 0.00 | ORB-short ORB[501.00,507.90] vol=1.7x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:30:00 | 492.56 | 499.48 | 0.00 | T1 1.5R @ 492.56 |
| Stop hit — per-position SL triggered | 2026-04-09 12:05:00 | 494.80 | 498.27 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:00:00 | 531.25 | 533.76 | 0.00 | ORB-short ORB[532.50,539.00] vol=1.5x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:20:00 | 527.63 | 533.24 | 0.00 | T1 1.5R @ 527.63 |
| Stop hit — per-position SL triggered | 2026-04-16 11:05:00 | 531.25 | 532.67 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:10:00 | 555.00 | 552.33 | 0.00 | ORB-long ORB[545.00,553.30] vol=1.7x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:30:00 | 559.06 | 553.56 | 0.00 | T1 1.5R @ 559.06 |
| Target hit | 2026-04-27 15:20:00 | 574.10 | 567.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-05-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:50:00 | 558.15 | 559.85 | 0.00 | ORB-short ORB[560.70,566.00] vol=1.8x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-05-04 11:25:00 | 560.04 | 559.47 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:30:00 | 576.00 | 572.27 | 0.00 | ORB-long ORB[568.00,573.60] vol=3.2x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-05-07 11:40:00 | 573.66 | 573.66 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:40:00 | 569.40 | 572.94 | 0.00 | ORB-short ORB[571.75,577.80] vol=3.4x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:05:00 | 566.73 | 572.24 | 0.00 | T1 1.5R @ 566.73 |
| Stop hit — per-position SL triggered | 2026-05-08 14:15:00 | 569.40 | 569.77 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 480.70 | 2026-02-11 10:10:00 | 478.70 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-11 09:30:00 | 480.70 | 2026-02-11 10:15:00 | 480.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:55:00 | 485.25 | 2026-02-18 14:55:00 | 486.65 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-23 10:40:00 | 490.00 | 2026-02-23 11:25:00 | 487.86 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-23 10:40:00 | 490.00 | 2026-02-23 11:40:00 | 490.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:30:00 | 494.75 | 2026-02-26 09:55:00 | 496.96 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-26 09:30:00 | 494.75 | 2026-02-26 11:45:00 | 500.30 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2026-03-05 11:15:00 | 477.20 | 2026-03-05 11:20:00 | 475.96 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-12 09:30:00 | 492.50 | 2026-03-12 09:40:00 | 495.86 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-12 09:30:00 | 492.50 | 2026-03-12 15:10:00 | 518.40 | TARGET_HIT | 0.50 | 5.26% |
| SELL | retest1 | 2026-04-09 11:15:00 | 494.80 | 2026-04-09 11:30:00 | 492.56 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-09 11:15:00 | 494.80 | 2026-04-09 12:05:00 | 494.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 10:00:00 | 531.25 | 2026-04-16 10:20:00 | 527.63 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-16 10:00:00 | 531.25 | 2026-04-16 11:05:00 | 531.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 10:10:00 | 555.00 | 2026-04-27 10:30:00 | 559.06 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-04-27 10:10:00 | 555.00 | 2026-04-27 15:20:00 | 574.10 | TARGET_HIT | 0.50 | 3.44% |
| SELL | retest1 | 2026-05-04 10:50:00 | 558.15 | 2026-05-04 11:25:00 | 560.04 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-07 10:30:00 | 576.00 | 2026-05-07 11:40:00 | 573.66 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-05-08 10:40:00 | 569.40 | 2026-05-08 11:05:00 | 566.73 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-05-08 10:40:00 | 569.40 | 2026-05-08 14:15:00 | 569.40 | STOP_HIT | 0.50 | 0.00% |
