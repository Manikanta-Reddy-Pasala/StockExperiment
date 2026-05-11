# Krishna Institute of Medical Sciences Ltd. (KIMS)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 717.60
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 9.25% / 7.13%
- **Sum % (uncompounded):** 36.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 1 | 1 | 2 | 9.25% | 37.0% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 9.25% | 37.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 1 | 1 | 2 | 9.25% | 37.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 00:00:00 | 438.30 | 407.12 | 426.20 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=11.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 00:00:00 | 461.42 | 409.24 | 436.01 | T1 booked 50% @ 461.42 |
| Target hit | 2024-10-07 00:00:00 | 534.15 | 444.87 | 539.71 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 00:00:00 | 584.00 | 472.34 | 555.25 | Stage2 pullback-breakout RSI=67 vol=2.4x ATR=20.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 00:00:00 | 625.62 | 482.99 | 580.21 | T1 booked 50% @ 625.62 |
| Stop hit — per-position SL triggered | 2024-12-12 00:00:00 | 599.85 | 490.54 | 594.53 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-08 00:00:00 | 438.30 | 2024-08-16 00:00:00 | 461.42 | PARTIAL | 0.50 | 5.28% |
| BUY | retest1 | 2024-08-08 00:00:00 | 438.30 | 2024-10-07 00:00:00 | 534.15 | TARGET_HIT | 0.50 | 21.87% |
| BUY | retest1 | 2024-11-21 00:00:00 | 584.00 | 2024-12-04 00:00:00 | 625.62 | PARTIAL | 0.50 | 7.13% |
| BUY | retest1 | 2024-11-21 00:00:00 | 584.00 | 2024-12-12 00:00:00 | 599.85 | STOP_HIT | 0.50 | 2.71% |
