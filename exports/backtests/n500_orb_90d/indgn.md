# Indegene Ltd. (INDGN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 530.00
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 6
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 2.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.30% | 2.7% |
| SELL @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.30% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.14% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 478.00 | 481.67 | 0.00 | ORB-short ORB[483.00,487.15] vol=1.7x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:20:00 | 475.54 | 480.10 | 0.00 | T1 1.5R @ 475.54 |
| Target hit | 2026-02-11 15:20:00 | 473.40 | 476.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:20:00 | 499.35 | 498.35 | 0.00 | ORB-long ORB[494.65,498.95] vol=10.5x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-02-19 10:25:00 | 497.47 | 498.32 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:20:00 | 492.05 | 490.41 | 0.00 | ORB-long ORB[486.75,489.95] vol=1.5x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:35:00 | 494.11 | 490.95 | 0.00 | T1 1.5R @ 494.11 |
| Stop hit — per-position SL triggered | 2026-02-20 11:25:00 | 492.05 | 491.89 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 482.75 | 486.50 | 0.00 | ORB-short ORB[484.25,490.90] vol=1.9x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-02-23 12:30:00 | 483.96 | 485.32 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 490.25 | 486.91 | 0.00 | ORB-long ORB[481.00,488.00] vol=2.0x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:55:00 | 493.66 | 488.62 | 0.00 | T1 1.5R @ 493.66 |
| Target hit | 2026-02-25 12:50:00 | 491.05 | 493.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 500.90 | 497.81 | 0.00 | ORB-long ORB[491.95,497.00] vol=7.1x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:45:00 | 503.90 | 499.08 | 0.00 | T1 1.5R @ 503.90 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 500.90 | 499.40 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-02 09:30:00 | 487.40 | 485.00 | 0.00 | ORB-long ORB[480.00,486.90] vol=1.5x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-03-02 10:00:00 | 485.12 | 485.31 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 460.45 | 459.99 | 0.00 | ORB-long ORB[454.60,459.75] vol=4.4x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-03-10 11:10:00 | 458.98 | 460.03 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:35:00 | 433.00 | 436.29 | 0.00 | ORB-short ORB[438.05,443.50] vol=6.7x ATR=2.15 |
| Stop hit — per-position SL triggered | 2026-03-13 10:40:00 | 435.15 | 436.15 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:10:00 | 488.05 | 491.04 | 0.00 | ORB-short ORB[489.85,494.00] vol=2.5x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:00:00 | 486.28 | 490.52 | 0.00 | T1 1.5R @ 486.28 |
| Stop hit — per-position SL triggered | 2026-04-17 13:30:00 | 488.05 | 489.67 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:50:00 | 490.30 | 486.47 | 0.00 | ORB-long ORB[482.10,488.00] vol=3.0x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-04-22 11:00:00 | 488.88 | 486.97 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:15:00 | 493.80 | 493.37 | 0.00 | ORB-long ORB[487.05,492.95] vol=1.9x ATR=1.91 |
| Stop hit — per-position SL triggered | 2026-04-27 11:40:00 | 491.89 | 493.33 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 502.70 | 504.59 | 0.00 | ORB-short ORB[503.05,508.00] vol=1.6x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:40:00 | 499.58 | 503.58 | 0.00 | T1 1.5R @ 499.58 |
| Target hit | 2026-04-29 15:20:00 | 495.55 | 499.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 531.45 | 536.92 | 0.00 | ORB-short ORB[534.00,541.55] vol=3.8x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 533.90 | 535.57 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 10:35:00 | 478.00 | 2026-02-11 11:20:00 | 475.54 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-11 10:35:00 | 478.00 | 2026-02-11 15:20:00 | 473.40 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2026-02-19 10:20:00 | 499.35 | 2026-02-19 10:25:00 | 497.47 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-20 10:20:00 | 492.05 | 2026-02-20 10:35:00 | 494.11 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-20 10:20:00 | 492.05 | 2026-02-20 11:25:00 | 492.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 10:55:00 | 482.75 | 2026-02-23 12:30:00 | 483.96 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-25 09:45:00 | 490.25 | 2026-02-25 09:55:00 | 493.66 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-02-25 09:45:00 | 490.25 | 2026-02-25 12:50:00 | 491.05 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-02-26 09:40:00 | 500.90 | 2026-02-26 09:45:00 | 503.90 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-26 09:40:00 | 500.90 | 2026-02-26 09:55:00 | 500.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-02 09:30:00 | 487.40 | 2026-03-02 10:00:00 | 485.12 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-10 10:50:00 | 460.45 | 2026-03-10 11:10:00 | 458.98 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-13 10:35:00 | 433.00 | 2026-03-13 10:40:00 | 435.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-04-17 11:10:00 | 488.05 | 2026-04-17 12:00:00 | 486.28 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-17 11:10:00 | 488.05 | 2026-04-17 13:30:00 | 488.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:50:00 | 490.30 | 2026-04-22 11:00:00 | 488.88 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-27 11:15:00 | 493.80 | 2026-04-27 11:40:00 | 491.89 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-29 10:00:00 | 502.70 | 2026-04-29 10:40:00 | 499.58 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-04-29 10:00:00 | 502.70 | 2026-04-29 15:20:00 | 495.55 | TARGET_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2026-05-07 09:30:00 | 531.45 | 2026-05-07 10:15:00 | 533.90 | STOP_HIT | 1.00 | -0.46% |
