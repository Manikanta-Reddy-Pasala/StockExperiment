# JK Tyre & Industries Ltd. (JKTYRE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 406.00
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 3
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 0.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.24% | 0.7% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.24% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.02% | -0.1% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.02% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.06% | 0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 530.00 | 534.29 | 0.00 | ORB-short ORB[534.00,538.90] vol=2.0x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:40:00 | 527.25 | 533.20 | 0.00 | T1 1.5R @ 527.25 |
| Stop hit — per-position SL triggered | 2026-02-26 14:40:00 | 530.00 | 531.03 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:20:00 | 422.35 | 429.71 | 0.00 | ORB-short ORB[427.50,432.90] vol=1.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-03-12 10:35:00 | 424.56 | 428.23 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:10:00 | 398.30 | 398.54 | 0.00 | ORB-short ORB[398.35,404.05] vol=2.3x ATR=1.52 |
| Stop hit — per-position SL triggered | 2026-03-27 11:50:00 | 399.82 | 398.60 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 424.65 | 428.03 | 0.00 | ORB-short ORB[426.90,431.40] vol=1.5x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:45:00 | 422.48 | 426.65 | 0.00 | T1 1.5R @ 422.48 |
| Stop hit — per-position SL triggered | 2026-04-16 11:05:00 | 424.65 | 424.59 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 429.10 | 425.38 | 0.00 | ORB-long ORB[419.95,426.20] vol=1.9x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:00:00 | 431.13 | 426.48 | 0.00 | T1 1.5R @ 431.13 |
| Target hit | 2026-04-17 13:15:00 | 431.45 | 432.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 399.85 | 398.54 | 0.00 | ORB-long ORB[396.10,399.80] vol=1.6x ATR=1.26 |
| Stop hit — per-position SL triggered | 2026-04-27 10:20:00 | 398.59 | 398.81 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-05-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:50:00 | 402.75 | 405.39 | 0.00 | ORB-short ORB[404.50,408.45] vol=2.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-05-06 11:20:00 | 403.79 | 404.93 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-26 10:50:00 | 530.00 | 2026-02-26 11:40:00 | 527.25 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-26 10:50:00 | 530.00 | 2026-02-26 14:40:00 | 530.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-12 10:20:00 | 422.35 | 2026-03-12 10:35:00 | 424.56 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-03-27 11:10:00 | 398.30 | 2026-03-27 11:50:00 | 399.82 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-16 09:40:00 | 424.65 | 2026-04-16 09:45:00 | 422.48 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-16 09:40:00 | 424.65 | 2026-04-16 11:05:00 | 424.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:55:00 | 429.10 | 2026-04-17 11:00:00 | 431.13 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-17 10:55:00 | 429.10 | 2026-04-17 13:15:00 | 431.45 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-27 09:50:00 | 399.85 | 2026-04-27 10:20:00 | 398.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-06 10:50:00 | 402.75 | 2026-05-06 11:20:00 | 403.79 | STOP_HIT | 1.00 | -0.26% |
