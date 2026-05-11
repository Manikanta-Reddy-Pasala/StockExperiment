# Swan Corp Ltd. (SWANCORP)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 353.05
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
- **Avg / median % per leg:** 18.21% / 9.21%
- **Sum % (uncompounded):** 72.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 18.21% | 72.8% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 18.21% | 72.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 18.21% | 72.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 05:30:00 | 304.10 | 260.01 | 292.09 | Stage2 pullback-breakout RSI=61 vol=2.9x ATR=12.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 05:30:00 | 329.20 | 262.87 | 302.86 | T1 booked 50% @ 329.20 |
| Stop hit — per-position SL triggered | 2023-10-23 05:30:00 | 304.10 | 263.76 | 303.72 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-12-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 05:30:00 | 448.35 | 300.50 | 415.51 | Stage2 pullback-breakout RSI=68 vol=3.7x ATR=20.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 05:30:00 | 489.66 | 311.66 | 442.11 | T1 booked 50% @ 489.66 |
| Target hit | 2024-03-06 05:30:00 | 696.65 | 453.49 | 712.70 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-12 05:30:00 | 304.10 | 2023-10-19 05:30:00 | 329.20 | PARTIAL | 0.50 | 8.25% |
| BUY | retest1 | 2023-10-12 05:30:00 | 304.10 | 2023-10-23 05:30:00 | 304.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-05 05:30:00 | 448.35 | 2023-12-14 05:30:00 | 489.66 | PARTIAL | 0.50 | 9.21% |
| BUY | retest1 | 2023-12-05 05:30:00 | 448.35 | 2024-03-06 05:30:00 | 696.65 | TARGET_HIT | 0.50 | 55.38% |
