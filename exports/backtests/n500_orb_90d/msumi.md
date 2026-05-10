# Motherson Sumi Wiring India Ltd. (MSUMI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 42.56
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 4
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 0.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.03% | -0.2% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.03% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.12% | 1.0% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.12% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 6 | 40.0% | 2 | 9 | 4 | 0.05% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 43.48 | 43.18 | 0.00 | ORB-long ORB[42.75,43.12] vol=2.6x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:00:00 | 43.68 | 43.23 | 0.00 | T1 1.5R @ 43.68 |
| Stop hit — per-position SL triggered | 2026-02-11 11:10:00 | 43.48 | 43.26 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 43.04 | 42.95 | 0.00 | ORB-long ORB[42.63,42.92] vol=2.5x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 42.93 | 42.95 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:40:00 | 42.55 | 42.89 | 0.00 | ORB-short ORB[42.69,43.13] vol=1.7x ATR=0.09 |
| Stop hit — per-position SL triggered | 2026-02-18 10:45:00 | 42.64 | 42.87 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 43.14 | 43.35 | 0.00 | ORB-short ORB[43.21,43.77] vol=3.4x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:30:00 | 42.97 | 43.25 | 0.00 | T1 1.5R @ 42.97 |
| Target hit | 2026-02-26 15:20:00 | 42.91 | 43.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:00:00 | 40.18 | 40.31 | 0.00 | ORB-short ORB[40.26,40.83] vol=1.8x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-03-10 12:50:00 | 40.33 | 40.28 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 40.54 | 40.91 | 0.00 | ORB-short ORB[40.70,41.30] vol=2.1x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:10:00 | 40.32 | 40.63 | 0.00 | T1 1.5R @ 40.32 |
| Target hit | 2026-03-11 15:20:00 | 40.21 | 40.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 39.22 | 39.49 | 0.00 | ORB-short ORB[39.24,39.80] vol=1.7x ATR=0.14 |
| Stop hit — per-position SL triggered | 2026-04-16 10:45:00 | 39.36 | 39.41 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 39.91 | 39.68 | 0.00 | ORB-long ORB[39.35,39.85] vol=1.6x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:40:00 | 40.12 | 39.77 | 0.00 | T1 1.5R @ 40.12 |
| Stop hit — per-position SL triggered | 2026-04-17 09:45:00 | 39.91 | 39.79 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 42.07 | 41.90 | 0.00 | ORB-long ORB[41.68,42.05] vol=1.9x ATR=0.17 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 41.90 | 41.92 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 43.04 | 42.72 | 0.00 | ORB-long ORB[41.98,42.48] vol=2.6x ATR=0.22 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 42.82 | 42.82 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:30:00 | 42.35 | 42.81 | 0.00 | ORB-short ORB[42.75,43.20] vol=2.8x ATR=0.16 |
| Stop hit — per-position SL triggered | 2026-05-08 10:55:00 | 42.51 | 42.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:55:00 | 43.48 | 2026-02-11 11:00:00 | 43.68 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-11 10:55:00 | 43.48 | 2026-02-11 11:10:00 | 43.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:35:00 | 43.04 | 2026-02-17 10:40:00 | 42.93 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-18 10:40:00 | 42.55 | 2026-02-18 10:45:00 | 42.64 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-26 10:40:00 | 43.14 | 2026-02-26 11:30:00 | 42.97 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-26 10:40:00 | 43.14 | 2026-02-26 15:20:00 | 42.91 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-10 11:00:00 | 40.18 | 2026-03-10 12:50:00 | 40.33 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-03-11 10:10:00 | 40.54 | 2026-03-11 13:10:00 | 40.32 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-11 10:10:00 | 40.54 | 2026-03-11 15:20:00 | 40.21 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2026-04-16 09:45:00 | 39.22 | 2026-04-16 10:45:00 | 39.36 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-17 09:35:00 | 39.91 | 2026-04-17 09:40:00 | 40.12 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-17 09:35:00 | 39.91 | 2026-04-17 09:45:00 | 39.91 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:30:00 | 42.07 | 2026-05-06 10:05:00 | 41.90 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-07 09:40:00 | 43.04 | 2026-05-07 09:50:00 | 42.82 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-05-08 10:30:00 | 42.35 | 2026-05-08 10:55:00 | 42.51 | STOP_HIT | 1.00 | -0.39% |
