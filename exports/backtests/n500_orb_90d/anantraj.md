# Anant Raj Ltd. (ANANTRAJ)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 561.75
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
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 8
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 3.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 0 | 5 | 4 | 0.24% | 2.1% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 0 | 5 | 4 | 0.24% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 0 | 7 | 4 | 0.08% | 0.9% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 0 | 7 | 4 | 0.08% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 8 | 40.0% | 0 | 12 | 8 | 0.15% | 3.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:35:00 | 552.50 | 557.08 | 0.00 | ORB-short ORB[556.20,564.00] vol=2.0x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:20:00 | 549.61 | 555.08 | 0.00 | T1 1.5R @ 549.61 |
| Stop hit — per-position SL triggered | 2026-02-12 12:45:00 | 552.50 | 554.38 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:10:00 | 538.35 | 542.74 | 0.00 | ORB-short ORB[540.10,545.00] vol=1.6x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:55:00 | 535.90 | 540.65 | 0.00 | T1 1.5R @ 535.90 |
| Stop hit — per-position SL triggered | 2026-02-17 12:35:00 | 538.35 | 540.32 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 551.45 | 545.00 | 0.00 | ORB-long ORB[542.50,548.25] vol=5.4x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:10:00 | 554.49 | 547.43 | 0.00 | T1 1.5R @ 554.49 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 551.45 | 547.95 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 539.90 | 542.74 | 0.00 | ORB-short ORB[541.80,548.50] vol=2.5x ATR=1.98 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 541.88 | 542.51 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 546.20 | 542.00 | 0.00 | ORB-long ORB[537.10,542.10] vol=2.3x ATR=2.27 |
| Stop hit — per-position SL triggered | 2026-02-26 09:50:00 | 543.93 | 542.26 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 527.75 | 532.09 | 0.00 | ORB-short ORB[532.70,537.90] vol=1.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:45:00 | 525.52 | 531.03 | 0.00 | T1 1.5R @ 525.52 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 527.75 | 529.89 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 459.00 | 465.77 | 0.00 | ORB-short ORB[466.30,473.05] vol=2.6x ATR=1.98 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 460.98 | 465.42 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 489.25 | 486.17 | 0.00 | ORB-long ORB[479.90,486.80] vol=1.7x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:30:00 | 492.75 | 487.18 | 0.00 | T1 1.5R @ 492.75 |
| Stop hit — per-position SL triggered | 2026-04-10 10:45:00 | 489.25 | 487.66 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 505.95 | 508.53 | 0.00 | ORB-short ORB[507.05,512.85] vol=1.6x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:45:00 | 503.10 | 507.50 | 0.00 | T1 1.5R @ 503.10 |
| Stop hit — per-position SL triggered | 2026-04-16 11:00:00 | 505.95 | 504.97 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 517.15 | 513.75 | 0.00 | ORB-long ORB[508.30,515.00] vol=2.9x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:35:00 | 520.28 | 515.62 | 0.00 | T1 1.5R @ 520.28 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 517.15 | 516.03 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 511.35 | 514.21 | 0.00 | ORB-short ORB[513.00,520.00] vol=1.8x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-04-23 11:30:00 | 512.75 | 514.07 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:45:00 | 531.25 | 529.00 | 0.00 | ORB-long ORB[525.10,531.00] vol=2.0x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:55:00 | 534.79 | 530.30 | 0.00 | T1 1.5R @ 534.79 |
| Stop hit — per-position SL triggered | 2026-05-06 10:00:00 | 531.25 | 530.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:35:00 | 552.50 | 2026-02-12 11:20:00 | 549.61 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-12 10:35:00 | 552.50 | 2026-02-12 12:45:00 | 552.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 11:10:00 | 538.35 | 2026-02-17 11:55:00 | 535.90 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-17 11:10:00 | 538.35 | 2026-02-17 12:35:00 | 538.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 11:05:00 | 551.45 | 2026-02-18 11:10:00 | 554.49 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-18 11:05:00 | 551.45 | 2026-02-18 11:15:00 | 551.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 539.90 | 2026-02-24 09:35:00 | 541.88 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-26 09:45:00 | 546.20 | 2026-02-26 09:50:00 | 543.93 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-27 10:40:00 | 527.75 | 2026-02-27 10:45:00 | 525.52 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-27 10:40:00 | 527.75 | 2026-02-27 11:00:00 | 527.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:40:00 | 459.00 | 2026-03-13 10:50:00 | 460.98 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-10 10:15:00 | 489.25 | 2026-04-10 10:30:00 | 492.75 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-04-10 10:15:00 | 489.25 | 2026-04-10 10:45:00 | 489.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:35:00 | 505.95 | 2026-04-16 09:45:00 | 503.10 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-16 09:35:00 | 505.95 | 2026-04-16 11:00:00 | 505.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:30:00 | 517.15 | 2026-04-21 09:35:00 | 520.28 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-21 09:30:00 | 517.15 | 2026-04-21 09:40:00 | 517.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:10:00 | 511.35 | 2026-04-23 11:30:00 | 512.75 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-06 09:45:00 | 531.25 | 2026-05-06 09:55:00 | 534.79 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-05-06 09:45:00 | 531.25 | 2026-05-06 10:00:00 | 531.25 | STOP_HIT | 0.50 | 0.00% |
