# Sona BLW Precision Forgings Ltd. (SONACOMS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 579.65
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
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 13
- **Target hits / Stop hits / Partials:** 1 / 13 / 5
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 1.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.09% | -0.9% |
| BUY @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.09% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.21% | 1.9% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.21% | 1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 6 | 31.6% | 1 | 13 | 5 | 0.05% | 1.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 532.15 | 530.88 | 0.00 | ORB-long ORB[526.20,531.90] vol=1.6x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-02-10 11:15:00 | 530.68 | 530.89 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 535.35 | 533.16 | 0.00 | ORB-long ORB[531.10,535.00] vol=1.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-02-11 12:15:00 | 534.01 | 533.96 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:15:00 | 532.20 | 531.40 | 0.00 | ORB-long ORB[524.20,529.70] vol=17.4x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:55:00 | 534.57 | 531.46 | 0.00 | T1 1.5R @ 534.57 |
| Stop hit — per-position SL triggered | 2026-02-20 14:05:00 | 532.20 | 531.83 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 534.90 | 532.13 | 0.00 | ORB-long ORB[529.60,532.15] vol=2.8x ATR=1.52 |
| Stop hit — per-position SL triggered | 2026-02-25 09:50:00 | 533.38 | 532.29 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:05:00 | 539.15 | 540.26 | 0.00 | ORB-short ORB[539.25,545.00] vol=1.6x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-02-26 11:35:00 | 540.60 | 540.16 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 537.90 | 540.52 | 0.00 | ORB-short ORB[538.50,545.85] vol=1.7x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:50:00 | 535.66 | 539.02 | 0.00 | T1 1.5R @ 535.66 |
| Stop hit — per-position SL triggered | 2026-02-27 10:40:00 | 537.90 | 537.53 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:30:00 | 516.10 | 511.22 | 0.00 | ORB-long ORB[505.70,513.20] vol=1.8x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-03-10 12:05:00 | 513.89 | 513.13 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 499.90 | 504.40 | 0.00 | ORB-short ORB[504.65,511.50] vol=1.7x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-03-13 10:45:00 | 501.84 | 504.13 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:55:00 | 478.20 | 482.75 | 0.00 | ORB-short ORB[480.65,485.55] vol=1.7x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 480.46 | 482.65 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 536.55 | 533.59 | 0.00 | ORB-long ORB[526.15,533.65] vol=1.7x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:15:00 | 540.04 | 534.77 | 0.00 | T1 1.5R @ 540.04 |
| Stop hit — per-position SL triggered | 2026-04-10 10:35:00 | 536.55 | 535.35 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:10:00 | 582.95 | 584.15 | 0.00 | ORB-short ORB[583.00,587.90] vol=2.7x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:20:00 | 580.01 | 583.57 | 0.00 | T1 1.5R @ 580.01 |
| Target hit | 2026-04-23 15:20:00 | 574.05 | 578.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 591.75 | 588.93 | 0.00 | ORB-long ORB[586.00,591.60] vol=2.6x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 590.22 | 589.16 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 580.00 | 576.19 | 0.00 | ORB-long ORB[569.35,577.15] vol=2.1x ATR=2.89 |
| Stop hit — per-position SL triggered | 2026-05-05 09:55:00 | 577.11 | 576.27 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:55:00 | 580.00 | 583.54 | 0.00 | ORB-short ORB[580.95,588.00] vol=2.4x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:10:00 | 576.65 | 582.79 | 0.00 | T1 1.5R @ 576.65 |
| Stop hit — per-position SL triggered | 2026-05-07 12:00:00 | 580.00 | 579.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 11:00:00 | 532.15 | 2026-02-10 11:15:00 | 530.68 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-11 10:55:00 | 535.35 | 2026-02-11 12:15:00 | 534.01 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-20 11:15:00 | 532.20 | 2026-02-20 11:55:00 | 534.57 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-20 11:15:00 | 532.20 | 2026-02-20 14:05:00 | 532.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:45:00 | 534.90 | 2026-02-25 09:50:00 | 533.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-26 11:05:00 | 539.15 | 2026-02-26 11:35:00 | 540.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-27 09:30:00 | 537.90 | 2026-02-27 09:50:00 | 535.66 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-27 09:30:00 | 537.90 | 2026-02-27 10:40:00 | 537.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:30:00 | 516.10 | 2026-03-10 12:05:00 | 513.89 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-13 10:40:00 | 499.90 | 2026-03-13 10:45:00 | 501.84 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-16 10:55:00 | 478.20 | 2026-03-16 11:15:00 | 480.46 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-10 10:05:00 | 536.55 | 2026-04-10 10:15:00 | 540.04 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-10 10:05:00 | 536.55 | 2026-04-10 10:35:00 | 536.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 10:10:00 | 582.95 | 2026-04-23 10:20:00 | 580.01 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-23 10:10:00 | 582.95 | 2026-04-23 15:20:00 | 574.05 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2026-04-28 11:00:00 | 591.75 | 2026-04-28 11:05:00 | 590.22 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-05 09:50:00 | 580.00 | 2026-05-05 09:55:00 | 577.11 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-05-07 09:55:00 | 580.00 | 2026-05-07 10:10:00 | 576.65 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-05-07 09:55:00 | 580.00 | 2026-05-07 12:00:00 | 580.00 | STOP_HIT | 0.50 | 0.00% |
