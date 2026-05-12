# Housing & Urban Development Corporation Ltd. (HUDCO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 227.23
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
- **Avg / median % per leg:** 31.72% / 6.19%
- **Sum % (uncompounded):** 126.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 31.72% | 126.9% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 31.72% | 126.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 31.72% | 126.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 00:00:00 | 60.30 | 50.10 | 58.59 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 00:00:00 | 63.81 | 50.94 | 59.43 | T1 booked 50% @ 63.81 |
| Target hit | 2023-08-04 00:00:00 | 61.65 | 52.04 | 61.90 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 85.75 | 66.95 | 81.55 | Stage2 pullback-breakout RSI=64 vol=4.5x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 00:00:00 | 91.06 | 67.98 | 84.19 | T1 booked 50% @ 91.06 |
| Target hit | 2024-03-12 00:00:00 | 182.35 | 114.68 | 192.72 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 00:00:00 | 60.30 | 2023-07-24 00:00:00 | 63.81 | PARTIAL | 0.50 | 5.82% |
| BUY | retest1 | 2023-07-11 00:00:00 | 60.30 | 2023-08-04 00:00:00 | 61.65 | TARGET_HIT | 0.50 | 2.24% |
| BUY | retest1 | 2023-11-30 00:00:00 | 85.75 | 2023-12-07 00:00:00 | 91.06 | PARTIAL | 0.50 | 6.19% |
| BUY | retest1 | 2023-11-30 00:00:00 | 85.75 | 2024-03-12 00:00:00 | 182.35 | TARGET_HIT | 0.50 | 112.65% |
