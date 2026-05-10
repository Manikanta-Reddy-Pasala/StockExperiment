# BAJAJ-AUTO (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 10711.50
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 2.39% / 3.19%
- **Sum % (uncompounded):** 9.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 2 | 2 | 2.39% | 9.5% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 2 | 2 | 2.39% | 9.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 2 | 2 | 2.39% | 9.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 05:30:00 | 9282.00 | 8888.12 | 9056.49 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=148.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 05:30:00 | 9578.14 | 8899.27 | 9128.94 | T1 booked 50% @ 9578.14 |
| Stop hit — per-position SL triggered | 2026-01-13 05:30:00 | 9554.00 | 8953.32 | 9388.76 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2026-04-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 05:30:00 | 9813.50 | 9161.77 | 9220.67 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=289.88 |
| Stop hit — per-position SL triggered | 2026-04-27 05:30:00 | 9662.00 | 9215.27 | 9524.11 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-04-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 05:30:00 | 9994.00 | 9228.99 | 9568.31 | Stage2 pullback-breakout RSI=63 vol=2.9x ATR=248.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 05:30:00 | 10490.47 | 9270.11 | 9808.73 | T1 booked 50% @ 10490.47 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-30 05:30:00 | 9282.00 | 2026-01-01 05:30:00 | 9578.14 | PARTIAL | 0.50 | 3.19% |
| BUY | retest1 | 2025-12-30 05:30:00 | 9282.00 | 2026-01-13 05:30:00 | 9554.00 | STOP_HIT | 0.50 | 2.93% |
| BUY | retest1 | 2026-04-10 05:30:00 | 9813.50 | 2026-04-27 05:30:00 | 9662.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest1 | 2026-04-30 05:30:00 | 9994.00 | 2026-05-07 05:30:00 | 10490.47 | PARTIAL | 0.50 | 4.97% |
