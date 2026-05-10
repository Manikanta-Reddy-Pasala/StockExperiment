# Newgen Software Technologies Ltd. (NEWGEN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 506.10
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** -0.09% / -0.41%
- **Sum % (uncompounded):** -0.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.09% | -0.5% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.09% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.09% | -0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 461.85 | 458.80 | 0.00 | ORB-long ORB[455.60,460.20] vol=1.6x ATR=1.92 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 459.93 | 458.95 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 477.20 | 474.71 | 0.00 | ORB-long ORB[471.95,476.00] vol=3.1x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:15:00 | 480.09 | 476.65 | 0.00 | T1 1.5R @ 480.09 |
| Target hit | 2026-04-21 13:40:00 | 478.75 | 478.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-04-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:50:00 | 495.90 | 489.48 | 0.00 | ORB-long ORB[486.25,492.55] vol=1.5x ATR=2.59 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 493.31 | 489.92 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:50:00 | 492.45 | 490.35 | 0.00 | ORB-long ORB[486.00,492.00] vol=1.9x ATR=2.29 |
| Stop hit — per-position SL triggered | 2026-04-29 10:25:00 | 490.16 | 490.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-04-10 09:30:00 | 461.85 | 2026-04-10 09:35:00 | 459.93 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-21 09:30:00 | 477.20 | 2026-04-21 10:15:00 | 480.09 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-21 09:30:00 | 477.20 | 2026-04-21 13:40:00 | 478.75 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-04-28 10:50:00 | 495.90 | 2026-04-28 11:00:00 | 493.31 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-29 09:50:00 | 492.45 | 2026-04-29 10:25:00 | 490.16 | STOP_HIT | 1.00 | -0.47% |
