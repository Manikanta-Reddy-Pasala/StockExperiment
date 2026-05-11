# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 6710.50
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.18% / -2.89%
- **Sum % (uncompounded):** 0.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.18% | 0.9% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.18% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.18% | 0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 6809.50 | 5944.81 | 6602.41 | Stage2 pullback-breakout RSI=65 vol=1.5x ATR=131.05 |
| Stop hit — per-position SL triggered | 2025-07-14 05:30:00 | 6612.92 | 6033.45 | 6772.56 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2025-10-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 05:30:00 | 6104.50 | 6086.63 | 5947.08 | Stage2 pullback-breakout RSI=56 vol=2.3x ATR=136.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 05:30:00 | 6376.79 | 6091.27 | 6026.76 | T1 booked 50% @ 6376.79 |
| Target hit | 2025-11-11 05:30:00 | 6539.50 | 6188.32 | 6567.16 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-01-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 05:30:00 | 6642.50 | 6265.85 | 6427.22 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=134.29 |
| Stop hit — per-position SL triggered | 2026-01-12 05:30:00 | 6441.07 | 6278.39 | 6477.78 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2026-02-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 05:30:00 | 6386.50 | 6250.01 | 6174.37 | Stage2 pullback-breakout RSI=57 vol=3.2x ATR=203.24 |
| Stop hit — per-position SL triggered | 2026-02-13 05:30:00 | 6081.64 | 6248.56 | 6174.72 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 6809.50 | 2025-07-14 05:30:00 | 6612.92 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2025-10-07 05:30:00 | 6104.50 | 2025-10-10 05:30:00 | 6376.79 | PARTIAL | 0.50 | 4.46% |
| BUY | retest1 | 2025-10-07 05:30:00 | 6104.50 | 2025-11-11 05:30:00 | 6539.50 | TARGET_HIT | 0.50 | 7.13% |
| BUY | retest1 | 2026-01-06 05:30:00 | 6642.50 | 2026-01-12 05:30:00 | 6441.07 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2026-02-11 05:30:00 | 6386.50 | 2026-02-13 05:30:00 | 6081.64 | STOP_HIT | 1.00 | -4.77% |
