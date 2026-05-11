# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3852.70
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
| TARGET_HIT | 2 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 2 / 0 / 2
- **Avg / median % per leg:** 10.57% / 9.63%
- **Sum % (uncompounded):** 42.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 10.57% | 42.3% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 10.57% | 42.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 10.57% | 42.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 774.53 | 510.63 | 743.41 | Stage2 pullback-breakout RSI=58 vol=1.5x ATR=28.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 00:00:00 | 832.30 | 513.81 | 751.77 | T1 booked 50% @ 832.30 |
| Target hit | 2024-02-12 00:00:00 | 787.42 | 534.95 | 788.44 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 747.70 | 585.71 | 725.76 | Stage2 pullback-breakout RSI=53 vol=2.6x ATR=36.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-27 00:00:00 | 819.70 | 591.31 | 739.20 | T1 booked 50% @ 819.70 |
| Target hit | 2024-04-29 00:00:00 | 923.75 | 657.71 | 943.77 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-31 00:00:00 | 774.53 | 2024-02-01 00:00:00 | 832.30 | PARTIAL | 0.50 | 7.46% |
| BUY | retest1 | 2024-01-31 00:00:00 | 774.53 | 2024-02-12 00:00:00 | 787.42 | TARGET_HIT | 0.50 | 1.66% |
| BUY | retest1 | 2024-03-21 00:00:00 | 747.70 | 2024-03-27 00:00:00 | 819.70 | PARTIAL | 0.50 | 9.63% |
| BUY | retest1 | 2024-03-21 00:00:00 | 747.70 | 2024-04-29 00:00:00 | 923.75 | TARGET_HIT | 0.50 | 23.55% |
