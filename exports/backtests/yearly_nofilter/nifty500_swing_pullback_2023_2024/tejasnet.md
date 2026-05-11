# Tejas Networks Ltd. (TEJASNET)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 482.90
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
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 4.18% / 5.51%
- **Sum % (uncompounded):** 16.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.18% | 16.7% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.18% | 16.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.18% | 16.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 00:00:00 | 741.45 | 647.14 | 716.85 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=20.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 00:00:00 | 782.28 | 649.75 | 728.45 | T1 booked 50% @ 782.28 |
| Target hit | 2023-07-24 00:00:00 | 776.60 | 662.37 | 781.54 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 871.10 | 701.71 | 834.03 | Stage2 pullback-breakout RSI=64 vol=3.5x ATR=28.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 927.55 | 712.50 | 858.47 | T1 booked 50% @ 927.55 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 871.10 | 715.80 | 862.01 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 00:00:00 | 741.45 | 2023-07-13 00:00:00 | 782.28 | PARTIAL | 0.50 | 5.51% |
| BUY | retest1 | 2023-07-11 00:00:00 | 741.45 | 2023-07-24 00:00:00 | 776.60 | TARGET_HIT | 0.50 | 4.74% |
| BUY | retest1 | 2023-08-31 00:00:00 | 871.10 | 2023-09-08 00:00:00 | 927.55 | PARTIAL | 0.50 | 6.48% |
| BUY | retest1 | 2023-08-31 00:00:00 | 871.10 | 2023-09-12 00:00:00 | 871.10 | STOP_HIT | 0.50 | 0.00% |
