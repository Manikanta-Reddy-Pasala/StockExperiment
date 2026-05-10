# Syngene International Ltd. (SYNGENE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 459.50
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 3
- **Avg / median % per leg:** 0.06% / -0.28%
- **Sum % (uncompounded):** 0.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.28% | 2.0% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.28% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.16% | -1.1% |
| SELL @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.16% | -1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 4 | 28.6% | 1 | 10 | 3 | 0.06% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 461.40 | 456.88 | 0.00 | ORB-long ORB[454.05,459.00] vol=3.4x ATR=1.35 |
| Stop hit — per-position SL triggered | 2026-02-10 11:05:00 | 460.05 | 456.99 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 431.50 | 432.57 | 0.00 | ORB-short ORB[433.00,436.65] vol=7.4x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:15:00 | 429.80 | 431.97 | 0.00 | T1 1.5R @ 429.80 |
| Stop hit — per-position SL triggered | 2026-02-18 10:35:00 | 431.50 | 431.80 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 437.20 | 435.12 | 0.00 | ORB-long ORB[431.75,436.50] vol=1.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-02-19 09:40:00 | 435.31 | 435.70 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:45:00 | 439.80 | 442.34 | 0.00 | ORB-short ORB[442.20,446.90] vol=1.5x ATR=1.35 |
| Stop hit — per-position SL triggered | 2026-02-23 10:55:00 | 441.15 | 442.25 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 436.15 | 437.74 | 0.00 | ORB-short ORB[437.00,439.85] vol=3.1x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-02-25 12:20:00 | 437.43 | 437.15 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 439.20 | 437.41 | 0.00 | ORB-long ORB[434.00,437.70] vol=2.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-02-26 10:10:00 | 437.72 | 437.69 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:45:00 | 398.55 | 400.56 | 0.00 | ORB-short ORB[401.00,402.75] vol=1.5x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-03-05 10:05:00 | 399.66 | 399.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:15:00 | 409.30 | 413.50 | 0.00 | ORB-short ORB[413.75,419.80] vol=2.4x ATR=1.26 |
| Stop hit — per-position SL triggered | 2026-03-19 11:45:00 | 410.56 | 413.00 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:15:00 | 402.75 | 407.83 | 0.00 | ORB-short ORB[408.75,414.70] vol=2.0x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-03-23 11:20:00 | 404.15 | 407.71 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 409.85 | 408.17 | 0.00 | ORB-long ORB[405.00,409.30] vol=2.7x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:10:00 | 412.09 | 409.41 | 0.00 | T1 1.5R @ 412.09 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 409.85 | 409.45 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 431.10 | 426.60 | 0.00 | ORB-long ORB[421.25,427.50] vol=2.5x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 433.76 | 428.05 | 0.00 | T1 1.5R @ 433.76 |
| Target hit | 2026-04-27 15:20:00 | 439.20 | 435.22 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 11:00:00 | 461.40 | 2026-02-10 11:05:00 | 460.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-18 09:55:00 | 431.50 | 2026-02-18 10:15:00 | 429.80 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-18 09:55:00 | 431.50 | 2026-02-18 10:35:00 | 431.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 09:30:00 | 437.20 | 2026-02-19 09:40:00 | 435.31 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-23 10:45:00 | 439.80 | 2026-02-23 10:55:00 | 441.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-25 10:40:00 | 436.15 | 2026-02-25 12:20:00 | 437.43 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-26 09:45:00 | 439.20 | 2026-02-26 10:10:00 | 437.72 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-05 09:45:00 | 398.55 | 2026-03-05 10:05:00 | 399.66 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-19 11:15:00 | 409.30 | 2026-03-19 11:45:00 | 410.56 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-23 11:15:00 | 402.75 | 2026-03-23 11:20:00 | 404.15 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-15 09:30:00 | 409.85 | 2026-04-15 10:10:00 | 412.09 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-15 09:30:00 | 409.85 | 2026-04-15 10:15:00 | 409.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:40:00 | 431.10 | 2026-04-27 09:50:00 | 433.76 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-27 09:40:00 | 431.10 | 2026-04-27 15:20:00 | 439.20 | TARGET_HIT | 0.50 | 1.88% |
