# NBCC (India) Ltd. (NBCC)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 100.64
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
- **Avg / median % per leg:** 8.60% / 8.86%
- **Sum % (uncompounded):** 51.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 8.60% | 51.6% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 8.60% | 51.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 8.60% | 51.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 05:30:00 | 58.67 | 38.08 | 53.16 | Stage2 pullback-breakout RSI=69 vol=4.3x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-01-17 05:30:00 | 58.63 | 40.14 | 57.44 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-01-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 05:30:00 | 63.03 | 40.55 | 58.03 | Stage2 pullback-breakout RSI=67 vol=2.7x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 05:30:00 | 68.61 | 40.82 | 58.97 | T1 booked 50% @ 68.61 |
| Target hit | 2024-02-13 05:30:00 | 82.93 | 47.89 | 86.08 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 05:30:00 | 85.47 | 58.03 | 80.47 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 05:30:00 | 94.02 | 59.52 | 83.84 | T1 booked 50% @ 94.02 |
| Stop hit — per-position SL triggered | 2024-04-15 05:30:00 | 85.47 | 60.36 | 84.86 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-04-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 05:30:00 | 89.50 | 61.54 | 84.82 | Stage2 pullback-breakout RSI=59 vol=2.3x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-05-08 05:30:00 | 90.60 | 64.31 | 88.39 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-03 05:30:00 | 58.67 | 2024-01-17 05:30:00 | 58.63 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest1 | 2024-01-19 05:30:00 | 63.03 | 2024-01-20 05:30:00 | 68.61 | PARTIAL | 0.50 | 8.86% |
| BUY | retest1 | 2024-01-19 05:30:00 | 63.03 | 2024-02-13 05:30:00 | 82.93 | TARGET_HIT | 0.50 | 31.57% |
| BUY | retest1 | 2024-04-02 05:30:00 | 85.47 | 2024-04-09 05:30:00 | 94.02 | PARTIAL | 0.50 | 10.00% |
| BUY | retest1 | 2024-04-02 05:30:00 | 85.47 | 2024-04-15 05:30:00 | 85.47 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-23 05:30:00 | 89.50 | 2024-05-08 05:30:00 | 90.60 | STOP_HIT | 1.00 | 1.23% |
