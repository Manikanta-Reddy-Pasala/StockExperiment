# Elecon Engineering Co. Ltd. (ELECON)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 533.75
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 3 / 1 / 4
- **Avg / median % per leg:** 10.60% / 7.97%
- **Sum % (uncompounded):** 84.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 7 | 87.5% | 3 | 1 | 4 | 10.60% | 84.8% |
| BUY @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 3 | 1 | 4 | 10.60% | 84.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 7 | 87.5% | 3 | 1 | 4 | 10.60% | 84.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 312.18 | 222.26 | 285.31 | Stage2 pullback-breakout RSI=69 vol=4.1x ATR=11.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 00:00:00 | 334.70 | 228.21 | 303.87 | T1 booked 50% @ 334.70 |
| Target hit | 2023-09-08 00:00:00 | 425.75 | 289.61 | 429.36 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 00:00:00 | 422.00 | 318.87 | 397.70 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=19.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 00:00:00 | 461.80 | 323.79 | 413.43 | T1 booked 50% @ 461.80 |
| Target hit | 2023-12-12 00:00:00 | 456.75 | 353.31 | 458.04 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-12-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 00:00:00 | 470.13 | 362.07 | 456.03 | Stage2 pullback-breakout RSI=58 vol=2.6x ATR=18.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 00:00:00 | 507.62 | 371.26 | 464.30 | T1 booked 50% @ 507.62 |
| Target hit | 2024-02-09 00:00:00 | 507.08 | 402.57 | 519.44 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-04-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 00:00:00 | 530.50 | 431.36 | 489.78 | Stage2 pullback-breakout RSI=65 vol=2.5x ATR=20.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 00:00:00 | 571.40 | 432.87 | 498.67 | T1 booked 50% @ 571.40 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 530.50 | 445.36 | 538.35 | SL hit (bars_held=11) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 312.18 | 2023-07-11 00:00:00 | 334.70 | PARTIAL | 0.50 | 7.21% |
| BUY | retest1 | 2023-07-03 00:00:00 | 312.18 | 2023-09-08 00:00:00 | 425.75 | TARGET_HIT | 0.50 | 36.38% |
| BUY | retest1 | 2023-10-31 00:00:00 | 422.00 | 2023-11-06 00:00:00 | 461.80 | PARTIAL | 0.50 | 9.43% |
| BUY | retest1 | 2023-10-31 00:00:00 | 422.00 | 2023-12-12 00:00:00 | 456.75 | TARGET_HIT | 0.50 | 8.23% |
| BUY | retest1 | 2023-12-26 00:00:00 | 470.13 | 2024-01-08 00:00:00 | 507.62 | PARTIAL | 0.50 | 7.97% |
| BUY | retest1 | 2023-12-26 00:00:00 | 470.13 | 2024-02-09 00:00:00 | 507.08 | TARGET_HIT | 0.50 | 7.86% |
| BUY | retest1 | 2024-04-18 00:00:00 | 530.50 | 2024-04-19 00:00:00 | 571.40 | PARTIAL | 0.50 | 7.71% |
| BUY | retest1 | 2024-04-18 00:00:00 | 530.50 | 2024-05-06 00:00:00 | 530.50 | STOP_HIT | 0.50 | 0.00% |
