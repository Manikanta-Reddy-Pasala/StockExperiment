# UNO Minda Ltd. (UNOMINDA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1167.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 4
- **Avg / median % per leg:** 2.75% / 5.32%
- **Sum % (uncompounded):** 24.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 1 | 4 | 4 | 2.75% | 24.8% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 1 | 4 | 4 | 2.75% | 24.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 1 | 4 | 4 | 2.75% | 24.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 00:00:00 | 590.25 | 539.41 | 572.23 | Stage2 pullback-breakout RSI=62 vol=1.5x ATR=15.30 |
| Stop hit — per-position SL triggered | 2023-07-21 00:00:00 | 567.30 | 541.96 | 574.07 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-07-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 00:00:00 | 595.15 | 544.20 | 577.39 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=15.62 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 571.71 | 544.88 | 577.60 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 00:00:00 | 613.80 | 548.33 | 583.39 | Stage2 pullback-breakout RSI=66 vol=5.7x ATR=17.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 00:00:00 | 648.50 | 549.21 | 588.48 | T1 booked 50% @ 648.50 |
| Stop hit — per-position SL triggered | 2023-08-24 00:00:00 | 613.80 | 552.87 | 602.09 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 602.00 | 571.27 | 588.81 | Stage2 pullback-breakout RSI=57 vol=5.9x ATR=16.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 00:00:00 | 634.05 | 571.87 | 592.86 | T1 booked 50% @ 634.05 |
| Target hit | 2023-12-11 00:00:00 | 646.20 | 587.92 | 650.84 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2023-12-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 00:00:00 | 687.65 | 596.81 | 660.53 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=18.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 00:00:00 | 724.94 | 605.35 | 683.84 | T1 booked 50% @ 724.94 |
| Stop hit — per-position SL triggered | 2024-01-15 00:00:00 | 702.95 | 607.43 | 688.59 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 673.80 | 625.76 | 644.39 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=22.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 00:00:00 | 718.35 | 629.91 | 669.06 | T1 booked 50% @ 718.35 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-12 00:00:00 | 590.25 | 2023-07-21 00:00:00 | 567.30 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest1 | 2023-07-31 00:00:00 | 595.15 | 2023-08-02 00:00:00 | 571.71 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest1 | 2023-08-16 00:00:00 | 613.80 | 2023-08-17 00:00:00 | 648.50 | PARTIAL | 0.50 | 5.65% |
| BUY | retest1 | 2023-08-16 00:00:00 | 613.80 | 2023-08-24 00:00:00 | 613.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-08 00:00:00 | 602.00 | 2023-11-09 00:00:00 | 634.05 | PARTIAL | 0.50 | 5.32% |
| BUY | retest1 | 2023-11-08 00:00:00 | 602.00 | 2023-12-11 00:00:00 | 646.20 | TARGET_HIT | 0.50 | 7.34% |
| BUY | retest1 | 2023-12-29 00:00:00 | 687.65 | 2024-01-11 00:00:00 | 724.94 | PARTIAL | 0.50 | 5.42% |
| BUY | retest1 | 2023-12-29 00:00:00 | 687.65 | 2024-01-15 00:00:00 | 702.95 | STOP_HIT | 0.50 | 2.22% |
| BUY | retest1 | 2024-03-26 00:00:00 | 673.80 | 2024-04-04 00:00:00 | 718.35 | PARTIAL | 0.50 | 6.61% |
