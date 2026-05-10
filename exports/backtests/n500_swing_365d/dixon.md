# Dixon Technologies (India) Ltd. (DIXON)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 10803.00
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 0
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 4.91% / 5.88%
- **Sum % (uncompounded):** 14.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 1 | 1 | 1 | 4.91% | 14.7% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 1 | 1 | 1 | 4.91% | 14.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 1 | 1 | 1 | 4.91% | 14.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 14983.00 | 14750.64 | 14596.40 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=490.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 05:30:00 | 15964.53 | 14816.30 | 15186.39 | T1 booked 50% @ 15964.53 |
| Target hit | 2025-08-08 05:30:00 | 15864.00 | 15107.45 | 16352.28 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 05:30:00 | 17582.00 | 15308.49 | 16663.15 | Stage2 pullback-breakout RSI=67 vol=2.2x ATR=431.64 |
| Stop hit — per-position SL triggered | 2025-09-15 05:30:00 | 17985.00 | 15555.96 | 17470.29 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 14983.00 | 2025-07-14 05:30:00 | 15964.53 | PARTIAL | 0.50 | 6.55% |
| BUY | retest1 | 2025-06-30 05:30:00 | 14983.00 | 2025-08-08 05:30:00 | 15864.00 | TARGET_HIT | 0.50 | 5.88% |
| BUY | retest1 | 2025-09-01 05:30:00 | 17582.00 | 2025-09-15 05:30:00 | 17985.00 | STOP_HIT | 1.00 | 2.29% |
