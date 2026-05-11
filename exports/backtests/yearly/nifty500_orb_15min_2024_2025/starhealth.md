# Star Health and Allied Insurance Company Ltd. (STARHEALTH)

## Backtest Summary

- **Window:** 2025-04-07 09:15:00 → 2026-05-08 15:25:00 (20038 bars)
- **Last close:** 519.05
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.09% / 0.00%
- **Sum % (uncompounded):** -0.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.49% | -1.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.49% | -1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.31% | 0.6% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.31% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.09% | -0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:40:00 | 369.25 | 366.04 | 0.00 | ORB-long ORB[362.90,366.65] vol=1.9x ATR=1.72 |
| Stop hit — per-position SL triggered | 2025-04-15 09:45:00 | 367.53 | 366.16 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:35:00 | 391.25 | 388.93 | 0.00 | ORB-long ORB[383.40,389.15] vol=4.9x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-04-16 09:40:00 | 389.26 | 389.06 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:45:00 | 398.60 | 399.85 | 0.00 | ORB-short ORB[399.00,404.70] vol=1.7x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:55:00 | 396.15 | 399.63 | 0.00 | T1 1.5R @ 396.15 |
| Stop hit — per-position SL triggered | 2025-04-23 11:10:00 | 398.60 | 399.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-04-15 09:40:00 | 369.25 | 2025-04-15 09:45:00 | 367.53 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-04-16 09:35:00 | 391.25 | 2025-04-16 09:40:00 | 389.26 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-04-23 10:45:00 | 398.60 | 2025-04-23 10:55:00 | 396.15 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-04-23 10:45:00 | 398.60 | 2025-04-23 11:10:00 | 398.60 | STOP_HIT | 0.50 | 0.00% |
