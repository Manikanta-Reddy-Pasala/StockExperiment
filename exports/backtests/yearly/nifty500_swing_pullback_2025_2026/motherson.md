# Samvardhana Motherson International Ltd. (MOTHERSON)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 132.03
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 0
- **Target hits / Stop hits / Partials:** 3 / 0 / 3
- **Avg / median % per leg:** 5.71% / 6.61%
- **Sum % (uncompounded):** 34.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 6 | 100.0% | 3 | 0 | 3 | 5.71% | 34.3% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 3 | 0 | 3 | 5.71% | 34.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 3 | 0 | 3 | 5.71% | 34.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 05:30:00 | 109.13 | 102.52 | 105.58 | Stage2 pullback-breakout RSI=62 vol=3.1x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 05:30:00 | 115.15 | 103.33 | 109.24 | T1 booked 50% @ 115.15 |
| Target hit | 2026-01-07 05:30:00 | 119.35 | 107.22 | 119.44 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 121.32 | 108.20 | 114.32 | Stage2 pullback-breakout RSI=63 vol=2.4x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 05:30:00 | 129.34 | 108.93 | 117.92 | T1 booked 50% @ 129.34 |
| Target hit | 2026-03-04 05:30:00 | 122.88 | 112.02 | 127.89 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 118.11 | 112.38 | 113.57 | Stage2 pullback-breakout RSI=54 vol=1.8x ATR=5.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 05:30:00 | 128.58 | 113.32 | 119.82 | T1 booked 50% @ 128.58 |
| Target hit | 2026-04-30 05:30:00 | 121.21 | 114.01 | 122.04 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-05-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 05:30:00 | 130.43 | 114.42 | 123.04 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=4.69 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-13 05:30:00 | 109.13 | 2025-11-27 05:30:00 | 115.15 | PARTIAL | 0.50 | 5.51% |
| BUY | retest1 | 2025-11-13 05:30:00 | 109.13 | 2026-01-07 05:30:00 | 119.35 | TARGET_HIT | 0.50 | 9.36% |
| BUY | retest1 | 2026-02-03 05:30:00 | 121.32 | 2026-02-10 05:30:00 | 129.34 | PARTIAL | 0.50 | 6.61% |
| BUY | retest1 | 2026-02-03 05:30:00 | 121.32 | 2026-03-04 05:30:00 | 122.88 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2026-04-08 05:30:00 | 118.11 | 2026-04-22 05:30:00 | 128.58 | PARTIAL | 0.50 | 8.86% |
| BUY | retest1 | 2026-04-08 05:30:00 | 118.11 | 2026-04-30 05:30:00 | 121.21 | TARGET_HIT | 0.50 | 2.62% |
