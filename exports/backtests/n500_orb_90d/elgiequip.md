# Elgi Equipments Ltd. (ELGIEQUIP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 561.70
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
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 0.33% / 0.00%
- **Sum % (uncompounded):** 4.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.51% | 4.1% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.51% | 4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.09% | 0.6% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.09% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.33% | 4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 540.50 | 536.19 | 0.00 | ORB-long ORB[532.20,538.75] vol=2.6x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:05:00 | 543.73 | 537.53 | 0.00 | T1 1.5R @ 543.73 |
| Target hit | 2026-02-23 15:20:00 | 549.45 | 544.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:15:00 | 541.90 | 544.30 | 0.00 | ORB-short ORB[543.05,548.85] vol=1.9x ATR=1.98 |
| Stop hit — per-position SL triggered | 2026-02-24 11:45:00 | 543.88 | 544.08 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:05:00 | 505.00 | 504.42 | 0.00 | ORB-long ORB[497.50,504.50] vol=5.6x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:35:00 | 508.35 | 504.89 | 0.00 | T1 1.5R @ 508.35 |
| Target hit | 2026-03-05 15:20:00 | 519.00 | 510.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-03-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:10:00 | 472.55 | 470.16 | 0.00 | ORB-long ORB[465.15,472.15] vol=1.5x ATR=2.30 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 470.25 | 470.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:00:00 | 480.30 | 482.71 | 0.00 | ORB-short ORB[481.00,486.85] vol=2.1x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:45:00 | 477.03 | 481.68 | 0.00 | T1 1.5R @ 477.03 |
| Stop hit — per-position SL triggered | 2026-03-27 13:00:00 | 480.30 | 479.67 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 539.00 | 537.72 | 0.00 | ORB-long ORB[535.00,538.90] vol=3.3x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-04-16 09:40:00 | 537.06 | 537.45 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 548.80 | 547.65 | 0.00 | ORB-long ORB[539.70,547.95] vol=2.6x ATR=2.18 |
| Stop hit — per-position SL triggered | 2026-04-17 09:40:00 | 546.62 | 547.63 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:35:00 | 544.95 | 550.63 | 0.00 | ORB-short ORB[550.05,555.80] vol=1.8x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:45:00 | 541.96 | 548.75 | 0.00 | T1 1.5R @ 541.96 |
| Stop hit — per-position SL triggered | 2026-05-05 12:20:00 | 544.95 | 548.01 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:35:00 | 541.70 | 543.60 | 0.00 | ORB-short ORB[541.75,547.90] vol=2.9x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-05-06 10:50:00 | 543.30 | 543.52 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 570.40 | 567.38 | 0.00 | ORB-long ORB[561.25,567.25] vol=3.1x ATR=2.22 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 568.18 | 569.72 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-23 10:50:00 | 540.50 | 2026-02-23 11:05:00 | 543.73 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-23 10:50:00 | 540.50 | 2026-02-23 15:20:00 | 549.45 | TARGET_HIT | 0.50 | 1.66% |
| SELL | retest1 | 2026-02-24 11:15:00 | 541.90 | 2026-02-24 11:45:00 | 543.88 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-05 10:05:00 | 505.00 | 2026-03-05 11:35:00 | 508.35 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-05 10:05:00 | 505.00 | 2026-03-05 15:20:00 | 519.00 | TARGET_HIT | 0.50 | 2.77% |
| BUY | retest1 | 2026-03-17 10:10:00 | 472.55 | 2026-03-17 10:25:00 | 470.25 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-27 10:00:00 | 480.30 | 2026-03-27 10:45:00 | 477.03 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-03-27 10:00:00 | 480.30 | 2026-03-27 13:00:00 | 480.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 09:30:00 | 539.00 | 2026-04-16 09:40:00 | 537.06 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-17 09:35:00 | 548.80 | 2026-04-17 09:40:00 | 546.62 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-05-05 10:35:00 | 544.95 | 2026-05-05 11:45:00 | 541.96 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-05-05 10:35:00 | 544.95 | 2026-05-05 12:20:00 | 544.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 10:35:00 | 541.70 | 2026-05-06 10:50:00 | 543.30 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-08 09:30:00 | 570.40 | 2026-05-08 09:50:00 | 568.18 | STOP_HIT | 1.00 | -0.39% |
