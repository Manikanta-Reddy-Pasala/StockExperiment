# Kalyan Jewellers India Ltd. (KALYANKJIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 425.00
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
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 7
- **Target hits / Stop hits / Partials:** 3 / 7 / 5
- **Avg / median % per leg:** 0.36% / 0.48%
- **Sum % (uncompounded):** 5.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.06% | 0.3% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.06% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.47% | 5.2% |
| SELL @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.47% | 5.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 8 | 53.3% | 3 | 7 | 5 | 0.36% | 5.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 429.25 | 430.20 | 0.00 | ORB-short ORB[429.50,435.35] vol=1.8x ATR=1.76 |
| Stop hit — per-position SL triggered | 2026-02-11 11:05:00 | 431.01 | 430.20 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:35:00 | 413.00 | 417.73 | 0.00 | ORB-short ORB[418.30,422.85] vol=1.7x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:25:00 | 411.00 | 416.03 | 0.00 | T1 1.5R @ 411.00 |
| Stop hit — per-position SL triggered | 2026-02-18 11:30:00 | 413.00 | 415.92 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 410.50 | 414.31 | 0.00 | ORB-short ORB[415.65,420.65] vol=1.5x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:10:00 | 408.53 | 412.43 | 0.00 | T1 1.5R @ 408.53 |
| Target hit | 2026-02-19 15:20:00 | 403.70 | 409.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-03-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:40:00 | 382.70 | 380.16 | 0.00 | ORB-long ORB[376.75,381.60] vol=1.9x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:50:00 | 385.40 | 381.31 | 0.00 | T1 1.5R @ 385.40 |
| Stop hit — per-position SL triggered | 2026-03-16 10:10:00 | 382.70 | 381.82 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:30:00 | 431.50 | 439.96 | 0.00 | ORB-short ORB[439.70,443.45] vol=3.5x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:35:00 | 428.16 | 435.01 | 0.00 | T1 1.5R @ 428.16 |
| Target hit | 2026-04-17 13:15:00 | 423.85 | 423.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 409.60 | 411.26 | 0.00 | ORB-short ORB[410.10,414.25] vol=1.8x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-04-23 09:50:00 | 410.77 | 410.89 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 416.50 | 414.22 | 0.00 | ORB-long ORB[411.00,414.70] vol=2.1x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 415.58 | 414.31 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 406.00 | 410.01 | 0.00 | ORB-short ORB[411.65,415.00] vol=10.9x ATR=1.20 |
| Stop hit — per-position SL triggered | 2026-05-05 11:30:00 | 407.20 | 408.90 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 412.90 | 410.76 | 0.00 | ORB-long ORB[408.55,412.05] vol=3.9x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-05-06 11:25:00 | 411.95 | 410.99 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 412.05 | 414.92 | 0.00 | ORB-short ORB[414.30,419.65] vol=2.5x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:50:00 | 409.98 | 412.86 | 0.00 | T1 1.5R @ 409.98 |
| Target hit | 2026-05-07 14:25:00 | 409.95 | 409.82 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 10:35:00 | 429.25 | 2026-02-11 11:05:00 | 431.01 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-18 10:35:00 | 413.00 | 2026-02-18 11:25:00 | 411.00 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-18 10:35:00 | 413.00 | 2026-02-18 11:30:00 | 413.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:35:00 | 410.50 | 2026-02-19 12:10:00 | 408.53 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-19 10:35:00 | 410.50 | 2026-02-19 15:20:00 | 403.70 | TARGET_HIT | 0.50 | 1.66% |
| BUY | retest1 | 2026-03-16 09:40:00 | 382.70 | 2026-03-16 09:50:00 | 385.40 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-16 09:40:00 | 382.70 | 2026-03-16 10:10:00 | 382.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-17 10:30:00 | 431.50 | 2026-04-17 10:35:00 | 428.16 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-04-17 10:30:00 | 431.50 | 2026-04-17 13:15:00 | 423.85 | TARGET_HIT | 0.50 | 1.77% |
| SELL | retest1 | 2026-04-23 09:35:00 | 409.60 | 2026-04-23 09:50:00 | 410.77 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-28 10:55:00 | 416.50 | 2026-04-28 11:00:00 | 415.58 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-05 10:55:00 | 406.00 | 2026-05-05 11:30:00 | 407.20 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-06 11:05:00 | 412.90 | 2026-05-06 11:25:00 | 411.95 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-07 09:30:00 | 412.05 | 2026-05-07 09:50:00 | 409.98 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-05-07 09:30:00 | 412.05 | 2026-05-07 14:25:00 | 409.95 | TARGET_HIT | 0.50 | 0.51% |
