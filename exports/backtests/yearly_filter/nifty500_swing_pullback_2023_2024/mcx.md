# Multi Commodity Exchange of India Ltd. (MCX)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3107.70
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
- **Avg / median % per leg:** 4.89% / 5.53%
- **Sum % (uncompounded):** 29.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 6 | 100.0% | 3 | 0 | 3 | 4.89% | 29.3% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 3 | 0 | 3 | 4.89% | 29.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 3 | 0 | 3 | 4.89% | 29.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 337.65 | 299.84 | 320.91 | Stage2 pullback-breakout RSI=67 vol=5.8x ATR=8.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 00:00:00 | 354.56 | 300.49 | 325.09 | T1 booked 50% @ 354.56 |
| Target hit | 2023-09-20 00:00:00 | 343.91 | 306.61 | 344.76 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 683.08 | 461.83 | 635.55 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=25.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 00:00:00 | 734.60 | 470.91 | 655.39 | T1 booked 50% @ 734.60 |
| Target hit | 2024-02-12 00:00:00 | 697.79 | 487.46 | 699.21 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 712.44 | 546.75 | 681.74 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=25.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 00:00:00 | 764.18 | 559.14 | 707.85 | T1 booked 50% @ 764.18 |
| Target hit | 2024-05-09 00:00:00 | 751.81 | 596.99 | 777.46 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-31 00:00:00 | 337.65 | 2023-09-01 00:00:00 | 354.56 | PARTIAL | 0.50 | 5.01% |
| BUY | retest1 | 2023-08-31 00:00:00 | 337.65 | 2023-09-20 00:00:00 | 343.91 | TARGET_HIT | 0.50 | 1.85% |
| BUY | retest1 | 2024-01-29 00:00:00 | 683.08 | 2024-02-02 00:00:00 | 734.60 | PARTIAL | 0.50 | 7.54% |
| BUY | retest1 | 2024-01-29 00:00:00 | 683.08 | 2024-02-12 00:00:00 | 697.79 | TARGET_HIT | 0.50 | 2.15% |
| BUY | retest1 | 2024-04-01 00:00:00 | 712.44 | 2024-04-10 00:00:00 | 764.18 | PARTIAL | 0.50 | 7.26% |
| BUY | retest1 | 2024-04-01 00:00:00 | 712.44 | 2024-05-09 00:00:00 | 751.81 | TARGET_HIT | 0.50 | 5.53% |
