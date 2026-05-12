# Sona BLW Precision Forgings Ltd. (SONACOMS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 571.45
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 1.57% / 3.61%
- **Sum % (uncompounded):** 14.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 1.57% | 14.1% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 1.57% | 14.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 2 | 4 | 3 | 1.57% | 14.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 00:00:00 | 535.30 | 485.63 | 519.85 | Stage2 pullback-breakout RSI=61 vol=2.6x ATR=12.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 00:00:00 | 560.37 | 490.83 | 536.33 | T1 booked 50% @ 560.37 |
| Target hit | 2023-08-02 00:00:00 | 554.60 | 498.23 | 559.10 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 00:00:00 | 588.25 | 533.43 | 559.89 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=15.25 |
| Stop hit — per-position SL triggered | 2023-11-28 00:00:00 | 565.38 | 536.71 | 567.99 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2023-12-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 00:00:00 | 571.40 | 540.74 | 562.36 | Stage2 pullback-breakout RSI=54 vol=2.4x ATR=17.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 00:00:00 | 605.40 | 542.20 | 569.89 | T1 booked 50% @ 605.40 |
| Target hit | 2024-01-11 00:00:00 | 606.70 | 551.93 | 612.06 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 618.20 | 556.72 | 597.95 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=17.78 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 591.53 | 561.42 | 611.62 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2024-02-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 00:00:00 | 639.85 | 564.68 | 611.98 | Stage2 pullback-breakout RSI=61 vol=3.7x ATR=21.52 |
| Stop hit — per-position SL triggered | 2024-02-22 00:00:00 | 607.57 | 565.63 | 612.10 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 664.10 | 567.43 | 620.25 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=23.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 00:00:00 | 710.33 | 570.74 | 635.62 | T1 booked 50% @ 710.33 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 664.10 | 576.07 | 653.13 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-07 00:00:00 | 535.30 | 2023-07-20 00:00:00 | 560.37 | PARTIAL | 0.50 | 4.68% |
| BUY | retest1 | 2023-07-07 00:00:00 | 535.30 | 2023-08-02 00:00:00 | 554.60 | TARGET_HIT | 0.50 | 3.61% |
| BUY | retest1 | 2023-11-15 00:00:00 | 588.25 | 2023-11-28 00:00:00 | 565.38 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest1 | 2023-12-21 00:00:00 | 571.40 | 2023-12-27 00:00:00 | 605.40 | PARTIAL | 0.50 | 5.95% |
| BUY | retest1 | 2023-12-21 00:00:00 | 571.40 | 2024-01-11 00:00:00 | 606.70 | TARGET_HIT | 0.50 | 6.18% |
| BUY | retest1 | 2024-01-31 00:00:00 | 618.20 | 2024-02-09 00:00:00 | 591.53 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest1 | 2024-02-20 00:00:00 | 639.85 | 2024-02-22 00:00:00 | 607.57 | STOP_HIT | 1.00 | -5.05% |
| BUY | retest1 | 2024-02-26 00:00:00 | 664.10 | 2024-02-29 00:00:00 | 710.33 | PARTIAL | 0.50 | 6.96% |
| BUY | retest1 | 2024-02-26 00:00:00 | 664.10 | 2024-03-06 00:00:00 | 664.10 | STOP_HIT | 0.50 | 0.00% |
