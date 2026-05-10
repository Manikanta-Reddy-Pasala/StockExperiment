# Birlasoft Ltd. (BSOFT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 362.50
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 0.05% / -0.32%
- **Sum % (uncompounded):** 0.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.34% | -0.7% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.34% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.25% | 1.0% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.25% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.05% | 0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 379.65 | 382.14 | 0.00 | ORB-short ORB[381.55,386.95] vol=1.6x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:55:00 | 377.30 | 381.23 | 0.00 | T1 1.5R @ 377.30 |
| Target hit | 2026-02-18 15:20:00 | 375.00 | 377.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 374.30 | 377.19 | 0.00 | ORB-short ORB[375.00,380.00] vol=2.2x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-03-12 10:35:00 | 376.19 | 375.65 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 368.35 | 371.14 | 0.00 | ORB-short ORB[370.35,374.15] vol=2.4x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-04-28 09:45:00 | 369.56 | 371.01 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:50:00 | 376.00 | 373.26 | 0.00 | ORB-long ORB[370.75,374.00] vol=2.2x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-04-29 09:55:00 | 374.64 | 373.43 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 371.80 | 370.71 | 0.00 | ORB-long ORB[368.35,371.75] vol=2.0x ATR=1.20 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 370.60 | 370.77 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 09:40:00 | 379.65 | 2026-02-18 09:55:00 | 377.30 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-02-18 09:40:00 | 379.65 | 2026-02-18 15:20:00 | 375.00 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2026-03-12 09:30:00 | 374.30 | 2026-03-12 10:35:00 | 376.19 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-04-28 09:40:00 | 368.35 | 2026-04-28 09:45:00 | 369.56 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-29 09:50:00 | 376.00 | 2026-04-29 09:55:00 | 374.64 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-04 09:30:00 | 371.80 | 2026-05-04 09:50:00 | 370.60 | STOP_HIT | 1.00 | -0.32% |
