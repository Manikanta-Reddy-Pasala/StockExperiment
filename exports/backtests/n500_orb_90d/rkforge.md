# Ramkrishna Forgings Ltd. (RKFORGE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 607.80
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 4 / 11 / 7
- **Avg / median % per leg:** 0.54% / 0.40%
- **Sum % (uncompounded):** 11.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 9 | 50.0% | 3 | 9 | 6 | 0.63% | 11.3% |
| BUY @ 2nd Alert (retest1) | 18 | 9 | 50.0% | 3 | 9 | 6 | 0.63% | 11.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.14% | 0.6% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.14% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 11 | 50.0% | 4 | 11 | 7 | 0.54% | 11.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:10:00 | 572.80 | 569.43 | 0.00 | ORB-long ORB[564.55,571.90] vol=3.2x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:35:00 | 576.88 | 570.48 | 0.00 | T1 1.5R @ 576.88 |
| Stop hit — per-position SL triggered | 2026-02-13 11:35:00 | 572.80 | 571.43 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 563.50 | 562.45 | 0.00 | ORB-long ORB[558.05,563.25] vol=2.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 561.94 | 562.53 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 556.70 | 560.27 | 0.00 | ORB-short ORB[562.45,568.00] vol=4.0x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:25:00 | 553.77 | 558.47 | 0.00 | T1 1.5R @ 553.77 |
| Target hit | 2026-02-18 15:20:00 | 552.45 | 554.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 561.00 | 553.69 | 0.00 | ORB-long ORB[549.00,554.10] vol=1.5x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:50:00 | 564.64 | 557.67 | 0.00 | T1 1.5R @ 564.64 |
| Target hit | 2026-02-19 10:45:00 | 564.30 | 564.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 548.15 | 545.31 | 0.00 | ORB-long ORB[542.05,546.75] vol=1.6x ATR=2.07 |
| Stop hit — per-position SL triggered | 2026-02-25 10:35:00 | 546.08 | 545.77 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 548.00 | 545.51 | 0.00 | ORB-long ORB[541.80,545.20] vol=1.5x ATR=1.32 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 546.68 | 545.70 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:00:00 | 547.80 | 546.27 | 0.00 | ORB-long ORB[542.00,547.75] vol=2.1x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:10:00 | 549.99 | 547.09 | 0.00 | T1 1.5R @ 549.99 |
| Stop hit — per-position SL triggered | 2026-02-27 10:20:00 | 547.80 | 548.05 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:40:00 | 555.30 | 551.95 | 0.00 | ORB-long ORB[549.00,553.75] vol=2.1x ATR=2.59 |
| Stop hit — per-position SL triggered | 2026-03-05 11:45:00 | 552.71 | 555.33 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:45:00 | 554.60 | 556.30 | 0.00 | ORB-short ORB[555.00,561.50] vol=2.9x ATR=2.16 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 556.76 | 556.01 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:05:00 | 564.25 | 562.64 | 0.00 | ORB-long ORB[557.50,562.80] vol=4.0x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-03-11 11:10:00 | 562.39 | 562.64 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:10:00 | 539.50 | 543.46 | 0.00 | ORB-short ORB[540.00,546.40] vol=2.2x ATR=1.85 |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 541.35 | 543.39 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:45:00 | 540.50 | 536.21 | 0.00 | ORB-long ORB[531.75,535.50] vol=3.6x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 13:05:00 | 543.54 | 538.01 | 0.00 | T1 1.5R @ 543.54 |
| Stop hit — per-position SL triggered | 2026-04-10 14:05:00 | 540.50 | 538.55 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 535.00 | 533.75 | 0.00 | ORB-long ORB[529.20,534.30] vol=4.0x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:50:00 | 538.31 | 547.25 | 0.00 | T1 1.5R @ 538.31 |
| Target hit | 2026-04-21 10:25:00 | 556.65 | 556.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 573.45 | 571.16 | 0.00 | ORB-long ORB[565.00,572.80] vol=3.5x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:00:00 | 578.25 | 572.27 | 0.00 | T1 1.5R @ 578.25 |
| Target hit | 2026-04-28 15:20:00 | 602.10 | 594.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-05-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:00:00 | 628.65 | 624.73 | 0.00 | ORB-long ORB[621.50,627.05] vol=3.2x ATR=2.90 |
| Stop hit — per-position SL triggered | 2026-05-07 11:00:00 | 625.75 | 626.59 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 10:10:00 | 572.80 | 2026-02-13 10:35:00 | 576.88 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-02-13 10:10:00 | 572.80 | 2026-02-13 11:35:00 | 572.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:05:00 | 563.50 | 2026-02-17 10:40:00 | 561.94 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-18 09:55:00 | 556.70 | 2026-02-18 10:25:00 | 553.77 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-18 09:55:00 | 556.70 | 2026-02-18 15:20:00 | 552.45 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2026-02-19 09:40:00 | 561.00 | 2026-02-19 09:50:00 | 564.64 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-02-19 09:40:00 | 561.00 | 2026-02-19 10:45:00 | 564.30 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-25 10:05:00 | 548.15 | 2026-02-25 10:35:00 | 546.08 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-26 10:55:00 | 548.00 | 2026-02-26 11:30:00 | 546.68 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-27 10:00:00 | 547.80 | 2026-02-27 10:10:00 | 549.99 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-27 10:00:00 | 547.80 | 2026-02-27 10:20:00 | 547.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 09:40:00 | 555.30 | 2026-03-05 11:45:00 | 552.71 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-03-10 09:45:00 | 554.60 | 2026-03-10 10:15:00 | 556.76 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-11 11:05:00 | 564.25 | 2026-03-11 11:10:00 | 562.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-18 10:10:00 | 539.50 | 2026-03-18 10:15:00 | 541.35 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-10 10:45:00 | 540.50 | 2026-04-10 13:05:00 | 543.54 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-10 10:45:00 | 540.50 | 2026-04-10 14:05:00 | 540.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:45:00 | 535.00 | 2026-04-21 09:50:00 | 538.31 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-21 09:45:00 | 535.00 | 2026-04-21 10:25:00 | 556.65 | TARGET_HIT | 0.50 | 4.05% |
| BUY | retest1 | 2026-04-28 09:40:00 | 573.45 | 2026-04-28 10:00:00 | 578.25 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2026-04-28 09:40:00 | 573.45 | 2026-04-28 15:20:00 | 602.10 | TARGET_HIT | 0.50 | 5.00% |
| BUY | retest1 | 2026-05-07 10:00:00 | 628.65 | 2026-05-07 11:00:00 | 625.75 | STOP_HIT | 1.00 | -0.46% |
