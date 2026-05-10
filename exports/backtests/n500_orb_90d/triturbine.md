# Triveni Turbine Ltd. (TRITURBINE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 598.20
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 4
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 1.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.14% | 1.1% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.14% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.02% | -0.1% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.02% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 4 | 33.3% | 0 | 8 | 4 | 0.09% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 510.15 | 508.42 | 0.00 | ORB-long ORB[505.00,509.25] vol=2.3x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:40:00 | 513.13 | 509.03 | 0.00 | T1 1.5R @ 513.13 |
| Stop hit — per-position SL triggered | 2026-02-09 11:45:00 | 510.15 | 509.04 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 490.85 | 489.92 | 0.00 | ORB-long ORB[485.50,489.95] vol=5.4x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 489.48 | 490.14 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:15:00 | 469.65 | 466.82 | 0.00 | ORB-long ORB[462.75,468.35] vol=1.9x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:05:00 | 472.35 | 467.99 | 0.00 | T1 1.5R @ 472.35 |
| Stop hit — per-position SL triggered | 2026-03-18 11:10:00 | 469.65 | 468.01 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:55:00 | 435.00 | 440.93 | 0.00 | ORB-short ORB[438.80,445.20] vol=1.8x ATR=1.92 |
| Stop hit — per-position SL triggered | 2026-03-24 11:25:00 | 436.92 | 440.29 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:55:00 | 465.50 | 462.87 | 0.00 | ORB-long ORB[457.05,462.00] vol=1.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-04-10 10:55:00 | 463.71 | 463.96 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 583.65 | 577.93 | 0.00 | ORB-long ORB[574.35,582.90] vol=2.3x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 587.45 | 582.68 | 0.00 | T1 1.5R @ 587.45 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 583.65 | 584.38 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:30:00 | 568.20 | 572.35 | 0.00 | ORB-short ORB[570.55,578.00] vol=1.5x ATR=2.58 |
| Stop hit — per-position SL triggered | 2026-04-30 10:40:00 | 570.78 | 572.24 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 570.25 | 572.56 | 0.00 | ORB-short ORB[570.55,578.65] vol=2.6x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:45:00 | 565.68 | 571.31 | 0.00 | T1 1.5R @ 565.68 |
| Stop hit — per-position SL triggered | 2026-05-08 11:10:00 | 570.25 | 570.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:00:00 | 510.15 | 2026-02-09 11:40:00 | 513.13 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-09 11:00:00 | 510.15 | 2026-02-09 11:45:00 | 510.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:45:00 | 490.85 | 2026-02-25 10:00:00 | 489.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-18 10:15:00 | 469.65 | 2026-03-18 11:05:00 | 472.35 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-18 10:15:00 | 469.65 | 2026-03-18 11:10:00 | 469.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 10:55:00 | 435.00 | 2026-03-24 11:25:00 | 436.92 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-10 09:55:00 | 465.50 | 2026-04-10 10:55:00 | 463.71 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-28 09:45:00 | 583.65 | 2026-04-28 10:15:00 | 587.45 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-28 09:45:00 | 583.65 | 2026-04-28 11:05:00 | 583.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:30:00 | 568.20 | 2026-04-30 10:40:00 | 570.78 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-05-08 09:50:00 | 570.25 | 2026-05-08 10:45:00 | 565.68 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2026-05-08 09:50:00 | 570.25 | 2026-05-08 11:10:00 | 570.25 | STOP_HIT | 0.50 | 0.00% |
