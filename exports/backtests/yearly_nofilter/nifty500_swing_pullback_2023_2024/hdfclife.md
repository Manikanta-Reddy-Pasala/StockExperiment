# HDFC Life Insurance Company Ltd. (HDFCLIFE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 616.00
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
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 2.65% / 3.76%
- **Sum % (uncompounded):** 10.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 2 | 2 | 2.65% | 10.6% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 2 | 2 | 2.65% | 10.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 2 | 2 | 2.65% | 10.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 644.75 | 588.64 | 636.54 | Stage2 pullback-breakout RSI=56 vol=1.8x ATR=12.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 00:00:00 | 668.96 | 592.83 | 644.49 | T1 booked 50% @ 668.96 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 644.75 | 594.07 | 646.45 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 651.80 | 606.77 | 628.18 | Stage2 pullback-breakout RSI=66 vol=2.1x ATR=12.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 00:00:00 | 677.06 | 610.86 | 648.42 | T1 booked 50% @ 677.06 |
| Stop hit — per-position SL triggered | 2023-12-08 00:00:00 | 671.25 | 615.41 | 662.99 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-31 00:00:00 | 644.75 | 2023-09-11 00:00:00 | 668.96 | PARTIAL | 0.50 | 3.76% |
| BUY | retest1 | 2023-08-31 00:00:00 | 644.75 | 2023-09-13 00:00:00 | 644.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-17 00:00:00 | 651.80 | 2023-11-29 00:00:00 | 677.06 | PARTIAL | 0.50 | 3.88% |
| BUY | retest1 | 2023-11-17 00:00:00 | 651.80 | 2023-12-08 00:00:00 | 671.25 | STOP_HIT | 0.50 | 2.98% |
