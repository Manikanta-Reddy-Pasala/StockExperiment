# Lenskart Solutions Ltd. (LENSKART)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 490.80
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
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 4
- **Avg / median % per leg:** -0.00% / 0.00%
- **Sum % (uncompounded):** -0.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.38% | -0.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.38% | -0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.06% | 0.7% |
| SELL @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.06% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 5 | 35.7% | 1 | 9 | 4 | -0.00% | -0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 491.00 | 486.72 | 0.00 | ORB-long ORB[483.00,489.50] vol=4.1x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-02-19 10:55:00 | 489.39 | 486.90 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 508.45 | 507.15 | 0.00 | ORB-long ORB[502.90,507.80] vol=2.2x ATR=2.20 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 506.25 | 507.18 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 497.70 | 501.06 | 0.00 | ORB-short ORB[498.95,504.95] vol=3.1x ATR=2.15 |
| Stop hit — per-position SL triggered | 2026-03-13 10:35:00 | 499.85 | 500.12 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 10:50:00 | 493.65 | 497.13 | 0.00 | ORB-short ORB[495.75,501.50] vol=2.2x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-04-07 11:25:00 | 495.37 | 496.38 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:05:00 | 533.55 | 536.05 | 0.00 | ORB-short ORB[535.50,542.20] vol=1.9x ATR=1.39 |
| Stop hit — per-position SL triggered | 2026-04-23 11:20:00 | 534.94 | 535.90 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:55:00 | 534.85 | 539.25 | 0.00 | ORB-short ORB[536.80,542.80] vol=2.8x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 537.64 | 539.13 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 11:00:00 | 527.50 | 530.41 | 0.00 | ORB-short ORB[528.10,534.45] vol=2.3x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:05:00 | 525.64 | 530.06 | 0.00 | T1 1.5R @ 525.64 |
| Target hit | 2026-04-27 14:45:00 | 525.65 | 525.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 512.10 | 515.67 | 0.00 | ORB-short ORB[516.00,522.65] vol=4.3x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:05:00 | 508.98 | 513.18 | 0.00 | T1 1.5R @ 508.98 |
| Stop hit — per-position SL triggered | 2026-05-05 10:25:00 | 512.10 | 512.82 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:50:00 | 501.90 | 505.44 | 0.00 | ORB-short ORB[507.55,512.60] vol=2.5x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:00:00 | 499.52 | 504.86 | 0.00 | T1 1.5R @ 499.52 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 501.90 | 503.95 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:15:00 | 493.10 | 496.92 | 0.00 | ORB-short ORB[498.55,503.35] vol=1.5x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:20:00 | 490.66 | 496.27 | 0.00 | T1 1.5R @ 490.66 |
| Stop hit — per-position SL triggered | 2026-05-07 10:25:00 | 493.10 | 496.05 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-19 10:50:00 | 491.00 | 2026-02-19 10:55:00 | 489.39 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-26 09:40:00 | 508.45 | 2026-02-26 09:55:00 | 506.25 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-13 10:00:00 | 497.70 | 2026-03-13 10:35:00 | 499.85 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-07 10:50:00 | 493.65 | 2026-04-07 11:25:00 | 495.37 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-23 11:05:00 | 533.55 | 2026-04-23 11:20:00 | 534.94 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-24 09:55:00 | 534.85 | 2026-04-24 10:00:00 | 537.64 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-04-27 11:00:00 | 527.50 | 2026-04-27 11:05:00 | 525.64 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-27 11:00:00 | 527.50 | 2026-04-27 14:45:00 | 525.65 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2026-05-05 09:45:00 | 512.10 | 2026-05-05 10:05:00 | 508.98 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-05-05 09:45:00 | 512.10 | 2026-05-05 10:25:00 | 512.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 10:50:00 | 501.90 | 2026-05-06 11:00:00 | 499.52 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-05-06 10:50:00 | 501.90 | 2026-05-06 11:10:00 | 501.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 10:15:00 | 493.10 | 2026-05-07 10:20:00 | 490.66 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-05-07 10:15:00 | 493.10 | 2026-05-07 10:25:00 | 493.10 | STOP_HIT | 0.50 | 0.00% |
