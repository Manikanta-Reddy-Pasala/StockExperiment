# Cummins India Ltd. (CUMMINSIND)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 5401.00
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 0
- **Target hits / Stop hits / Partials:** 3 / 0 / 3
- **Avg / median % per leg:** 5.51% / 4.40%
- **Sum % (uncompounded):** 33.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 6 | 100.0% | 3 | 0 | 3 | 5.51% | 33.0% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 3 | 0 | 3 | 5.51% | 33.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 3 | 0 | 3 | 5.51% | 33.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-06 05:30:00 | 3635.70 | 3285.89 | 3543.50 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=79.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 05:30:00 | 3795.43 | 3294.90 | 3579.83 | T1 booked 50% @ 3795.43 |
| Target hit | 2025-09-25 05:30:00 | 3976.90 | 3470.11 | 3978.72 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 05:30:00 | 4073.90 | 3550.78 | 3974.48 | Stage2 pullback-breakout RSI=63 vol=2.2x ATR=75.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 05:30:00 | 4225.29 | 3564.58 | 4024.56 | T1 booked 50% @ 4225.29 |
| Target hit | 2025-11-18 05:30:00 | 4252.40 | 3672.74 | 4268.41 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 05:30:00 | 4391.40 | 3926.27 | 4139.52 | Stage2 pullback-breakout RSI=64 vol=2.5x ATR=148.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 05:30:00 | 4688.11 | 3973.64 | 4353.50 | T1 booked 50% @ 4688.11 |
| Target hit | 2026-03-04 05:30:00 | 4584.80 | 4046.26 | 4627.33 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-06 05:30:00 | 3635.70 | 2025-08-08 05:30:00 | 3795.43 | PARTIAL | 0.50 | 4.39% |
| BUY | retest1 | 2025-08-06 05:30:00 | 3635.70 | 2025-09-25 05:30:00 | 3976.90 | TARGET_HIT | 0.50 | 9.38% |
| BUY | retest1 | 2025-10-23 05:30:00 | 4073.90 | 2025-10-27 05:30:00 | 4225.29 | PARTIAL | 0.50 | 3.72% |
| BUY | retest1 | 2025-10-23 05:30:00 | 4073.90 | 2025-11-18 05:30:00 | 4252.40 | TARGET_HIT | 0.50 | 4.38% |
| BUY | retest1 | 2026-02-05 05:30:00 | 4391.40 | 2026-02-18 05:30:00 | 4688.11 | PARTIAL | 0.50 | 6.76% |
| BUY | retest1 | 2026-02-05 05:30:00 | 4391.40 | 2026-03-04 05:30:00 | 4584.80 | TARGET_HIT | 0.50 | 4.40% |
