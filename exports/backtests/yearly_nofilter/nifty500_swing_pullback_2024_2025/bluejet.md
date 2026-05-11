# Blue Jet Healthcare Ltd. (BLUEJET)

## Backtest Summary

- **Window:** 2023-11-01 00:00:00 → 2026-05-11 00:00:00 (625 bars)
- **Last close:** 476.50
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 2.63% / 2.30%
- **Sum % (uncompounded):** 7.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.63% | 7.9% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.63% | 7.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.63% | 7.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 00:00:00 | 509.45 | 412.77 | 478.38 | Stage2 pullback-breakout RSI=63 vol=3.3x ATR=22.03 |
| Stop hit — per-position SL triggered | 2024-09-20 00:00:00 | 521.15 | 422.40 | 501.72 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 547.05 | 444.02 | 496.65 | Stage2 pullback-breakout RSI=64 vol=5.8x ATR=25.38 |
| Stop hit — per-position SL triggered | 2024-11-22 00:00:00 | 533.50 | 452.73 | 521.14 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-12-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 00:00:00 | 544.00 | 462.33 | 514.01 | Stage2 pullback-breakout RSI=62 vol=5.3x ATR=21.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 00:00:00 | 587.88 | 464.60 | 525.46 | T1 booked 50% @ 587.88 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-06 00:00:00 | 509.45 | 2024-09-20 00:00:00 | 521.15 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest1 | 2024-11-06 00:00:00 | 547.05 | 2024-11-22 00:00:00 | 533.50 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest1 | 2024-12-17 00:00:00 | 544.00 | 2024-12-19 00:00:00 | 587.88 | PARTIAL | 0.50 | 8.07% |
