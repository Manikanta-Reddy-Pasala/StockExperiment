# Minda Corporation Ltd. (MINDACORP)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 535.95
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 2.76% / 6.04%
- **Sum % (uncompounded):** 16.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.76% | 16.6% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.76% | 16.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.76% | 16.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 293.80 | 241.87 | 284.47 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=10.15 |
| Stop hit — per-position SL triggered | 2023-07-04 00:00:00 | 278.57 | 242.30 | 284.48 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2023-08-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 00:00:00 | 307.90 | 257.71 | 299.23 | Stage2 pullback-breakout RSI=57 vol=2.2x ATR=10.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 00:00:00 | 329.41 | 262.29 | 309.43 | T1 booked 50% @ 329.41 |
| Target hit | 2023-09-21 00:00:00 | 326.50 | 273.77 | 333.20 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 335.65 | 279.09 | 331.00 | Stage2 pullback-breakout RSI=55 vol=1.6x ATR=10.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 00:00:00 | 357.35 | 284.45 | 338.08 | T1 booked 50% @ 357.35 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 335.65 | 285.57 | 338.52 | SL hit (bars_held=11) |

### Cycle 4 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 373.15 | 298.17 | 345.13 | Stage2 pullback-breakout RSI=65 vol=4.7x ATR=13.68 |
| Stop hit — per-position SL triggered | 2023-12-13 00:00:00 | 381.65 | 305.53 | 365.03 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 293.80 | 2023-07-04 00:00:00 | 278.57 | STOP_HIT | 1.00 | -5.18% |
| BUY | retest1 | 2023-08-17 00:00:00 | 307.90 | 2023-08-29 00:00:00 | 329.41 | PARTIAL | 0.50 | 6.98% |
| BUY | retest1 | 2023-08-17 00:00:00 | 307.90 | 2023-09-21 00:00:00 | 326.50 | TARGET_HIT | 0.50 | 6.04% |
| BUY | retest1 | 2023-10-06 00:00:00 | 335.65 | 2023-10-19 00:00:00 | 357.35 | PARTIAL | 0.50 | 6.47% |
| BUY | retest1 | 2023-10-06 00:00:00 | 335.65 | 2023-10-23 00:00:00 | 335.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-29 00:00:00 | 373.15 | 2023-12-13 00:00:00 | 381.65 | STOP_HIT | 1.00 | 2.28% |
