# Indian Railway Catering And Tourism Corporation Ltd. (IRCTC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 565.50
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 7
- **Target hits / Stop hits / Partials:** 3 / 7 / 4
- **Avg / median % per leg:** 0.10% / 0.32%
- **Sum % (uncompounded):** 1.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.26% | 2.1% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.26% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.12% | -0.7% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.12% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.10% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 627.25 | 625.31 | 0.00 | ORB-long ORB[621.80,627.00] vol=3.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-02-13 09:35:00 | 625.51 | 625.32 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:40:00 | 607.40 | 610.58 | 0.00 | ORB-short ORB[609.15,617.35] vol=1.8x ATR=2.27 |
| Stop hit — per-position SL triggered | 2026-02-16 10:00:00 | 609.67 | 609.92 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 627.95 | 625.84 | 0.00 | ORB-long ORB[619.30,627.85] vol=1.8x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:50:00 | 630.32 | 627.14 | 0.00 | T1 1.5R @ 630.32 |
| Target hit | 2026-02-19 15:00:00 | 631.70 | 632.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 542.20 | 544.33 | 0.00 | ORB-short ORB[543.00,550.00] vol=1.5x ATR=1.98 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 544.18 | 543.73 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:45:00 | 533.80 | 528.91 | 0.00 | ORB-long ORB[523.50,531.20] vol=2.0x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:55:00 | 536.29 | 529.80 | 0.00 | T1 1.5R @ 536.29 |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 533.80 | 530.27 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 524.80 | 527.45 | 0.00 | ORB-short ORB[526.40,532.00] vol=2.3x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-03-20 10:10:00 | 526.55 | 525.99 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:50:00 | 550.00 | 548.34 | 0.00 | ORB-long ORB[545.15,549.80] vol=1.6x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:05:00 | 553.18 | 550.68 | 0.00 | T1 1.5R @ 553.18 |
| Target hit | 2026-04-15 15:20:00 | 553.40 | 552.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-04-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:50:00 | 553.00 | 555.17 | 0.00 | ORB-short ORB[554.50,558.10] vol=1.7x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 14:25:00 | 551.24 | 553.98 | 0.00 | T1 1.5R @ 551.24 |
| Target hit | 2026-04-23 15:20:00 | 551.00 | 553.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-05-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:55:00 | 551.50 | 546.61 | 0.00 | ORB-long ORB[542.05,546.85] vol=2.1x ATR=1.62 |
| Stop hit — per-position SL triggered | 2026-05-04 10:20:00 | 549.88 | 547.95 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:20:00 | 569.45 | 573.08 | 0.00 | ORB-short ORB[571.25,578.05] vol=2.2x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-05-07 11:50:00 | 571.20 | 570.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 09:30:00 | 627.25 | 2026-02-13 09:35:00 | 625.51 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-16 09:40:00 | 607.40 | 2026-02-16 10:00:00 | 609.67 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-19 09:30:00 | 627.95 | 2026-02-19 09:50:00 | 630.32 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-19 09:30:00 | 627.95 | 2026-02-19 15:00:00 | 631.70 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2026-03-10 09:35:00 | 542.20 | 2026-03-10 10:15:00 | 544.18 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-18 10:45:00 | 533.80 | 2026-03-18 10:55:00 | 536.29 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-18 10:45:00 | 533.80 | 2026-03-18 11:15:00 | 533.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 09:35:00 | 524.80 | 2026-03-20 10:10:00 | 526.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-15 09:50:00 | 550.00 | 2026-04-15 10:05:00 | 553.18 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-15 09:50:00 | 550.00 | 2026-04-15 15:20:00 | 553.40 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2026-04-23 10:50:00 | 553.00 | 2026-04-23 14:25:00 | 551.24 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-04-23 10:50:00 | 553.00 | 2026-04-23 15:20:00 | 551.00 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-05-04 09:55:00 | 551.50 | 2026-05-04 10:20:00 | 549.88 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-07 10:20:00 | 569.45 | 2026-05-07 11:50:00 | 571.20 | STOP_HIT | 1.00 | -0.31% |
