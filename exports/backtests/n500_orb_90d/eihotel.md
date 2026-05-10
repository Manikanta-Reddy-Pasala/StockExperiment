# EIH Ltd. (EIHOTEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 336.00
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 3
- **Avg / median % per leg:** 0.12% / -0.30%
- **Sum % (uncompounded):** 1.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.05% | 0.3% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.05% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.16% | 1.4% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.16% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 6 | 40.0% | 3 | 9 | 3 | 0.12% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 328.50 | 329.78 | 0.00 | ORB-short ORB[329.15,332.15] vol=2.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 329.61 | 329.75 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 331.55 | 333.05 | 0.00 | ORB-short ORB[332.50,336.25] vol=2.4x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:30:00 | 330.57 | 332.78 | 0.00 | T1 1.5R @ 330.57 |
| Target hit | 2026-02-19 15:20:00 | 325.10 | 330.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 327.65 | 325.19 | 0.00 | ORB-long ORB[322.00,325.30] vol=5.4x ATR=1.03 |
| Stop hit — per-position SL triggered | 2026-02-20 11:00:00 | 326.62 | 325.29 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:50:00 | 324.80 | 326.58 | 0.00 | ORB-short ORB[325.15,329.20] vol=2.8x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:35:00 | 323.00 | 324.33 | 0.00 | T1 1.5R @ 323.00 |
| Target hit | 2026-02-23 13:45:00 | 324.20 | 324.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2026-02-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:25:00 | 318.30 | 320.34 | 0.00 | ORB-short ORB[320.10,323.60] vol=2.2x ATR=1.01 |
| Stop hit — per-position SL triggered | 2026-02-24 11:10:00 | 319.31 | 319.18 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 318.55 | 320.26 | 0.00 | ORB-short ORB[318.80,323.10] vol=4.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-02-25 10:05:00 | 319.66 | 319.89 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 315.60 | 317.27 | 0.00 | ORB-short ORB[316.75,321.00] vol=1.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-02-27 11:55:00 | 316.72 | 316.13 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 320.05 | 317.85 | 0.00 | ORB-long ORB[314.50,318.80] vol=1.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-03-11 12:00:00 | 318.94 | 319.12 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:50:00 | 305.80 | 304.03 | 0.00 | ORB-long ORB[301.00,304.95] vol=1.7x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:05:00 | 308.09 | 305.25 | 0.00 | T1 1.5R @ 308.09 |
| Target hit | 2026-04-13 15:10:00 | 308.60 | 308.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 331.25 | 329.65 | 0.00 | ORB-long ORB[327.60,330.45] vol=1.5x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-04-27 10:10:00 | 329.88 | 329.73 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 326.55 | 328.18 | 0.00 | ORB-short ORB[327.00,330.20] vol=2.8x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 327.11 | 326.73 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:00:00 | 328.00 | 326.99 | 0.00 | ORB-long ORB[324.60,327.95] vol=2.2x ATR=1.00 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 327.00 | 327.00 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:35:00 | 328.50 | 2026-02-13 09:40:00 | 329.61 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-19 11:05:00 | 331.55 | 2026-02-19 11:30:00 | 330.57 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-19 11:05:00 | 331.55 | 2026-02-19 15:20:00 | 325.10 | TARGET_HIT | 0.50 | 1.95% |
| BUY | retest1 | 2026-02-20 10:50:00 | 327.65 | 2026-02-20 11:00:00 | 326.62 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-23 09:50:00 | 324.80 | 2026-02-23 11:35:00 | 323.00 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-02-23 09:50:00 | 324.80 | 2026-02-23 13:45:00 | 324.20 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2026-02-24 10:25:00 | 318.30 | 2026-02-24 11:10:00 | 319.31 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-25 09:40:00 | 318.55 | 2026-02-25 10:05:00 | 319.66 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-27 10:15:00 | 315.60 | 2026-02-27 11:55:00 | 316.72 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-11 09:55:00 | 320.05 | 2026-03-11 12:00:00 | 318.94 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-13 09:50:00 | 305.80 | 2026-04-13 10:05:00 | 308.09 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-04-13 09:50:00 | 305.80 | 2026-04-13 15:10:00 | 308.60 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2026-04-27 09:50:00 | 331.25 | 2026-04-27 10:10:00 | 329.88 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-28 11:10:00 | 326.55 | 2026-04-28 11:20:00 | 327.11 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-05-05 10:00:00 | 328.00 | 2026-05-05 10:10:00 | 327.00 | STOP_HIT | 1.00 | -0.30% |
