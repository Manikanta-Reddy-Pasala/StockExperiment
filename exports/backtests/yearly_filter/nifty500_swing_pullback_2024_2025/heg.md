# H.E.G. Ltd. (HEG)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 597.70
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
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 2.02% / 5.35%
- **Sum % (uncompounded):** 10.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.02% | 10.1% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.02% | 10.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.02% | 10.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 00:00:00 | 439.17 | 404.99 | 427.62 | Stage2 pullback-breakout RSI=55 vol=4.7x ATR=16.00 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 415.17 | 406.07 | 428.83 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-08-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 00:00:00 | 433.95 | 406.54 | 422.93 | Stage2 pullback-breakout RSI=54 vol=1.9x ATR=15.84 |
| Stop hit — per-position SL triggered | 2024-08-14 00:00:00 | 410.18 | 406.49 | 420.87 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2024-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 00:00:00 | 413.86 | 405.58 | 404.07 | Stage2 pullback-breakout RSI=56 vol=1.9x ATR=11.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 00:00:00 | 436.01 | 406.48 | 412.59 | T1 booked 50% @ 436.01 |
| Target hit | 2024-10-07 00:00:00 | 445.58 | 413.85 | 454.67 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-11-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 00:00:00 | 438.30 | 422.20 | 425.37 | Stage2 pullback-breakout RSI=54 vol=2.6x ATR=17.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 00:00:00 | 473.55 | 423.34 | 434.99 | T1 booked 50% @ 473.55 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-30 00:00:00 | 439.17 | 2024-08-05 00:00:00 | 415.17 | STOP_HIT | 1.00 | -5.46% |
| BUY | retest1 | 2024-08-13 00:00:00 | 433.95 | 2024-08-14 00:00:00 | 410.18 | STOP_HIT | 1.00 | -5.48% |
| BUY | retest1 | 2024-09-13 00:00:00 | 413.86 | 2024-09-18 00:00:00 | 436.01 | PARTIAL | 0.50 | 5.35% |
| BUY | retest1 | 2024-09-13 00:00:00 | 413.86 | 2024-10-07 00:00:00 | 445.58 | TARGET_HIT | 0.50 | 7.66% |
| BUY | retest1 | 2024-11-28 00:00:00 | 438.30 | 2024-12-03 00:00:00 | 473.55 | PARTIAL | 0.50 | 8.04% |
