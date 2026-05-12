# DLF Ltd. (DLF)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 608.25
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 4.28% / 0.00%
- **Sum % (uncompounded):** 29.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 4.28% | 30.0% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 4.28% | 30.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 4.28% | 30.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 00:00:00 | 513.60 | 416.72 | 488.91 | Stage2 pullback-breakout RSI=69 vol=2.1x ATR=11.12 |
| Stop hit — per-position SL triggered | 2023-07-10 00:00:00 | 496.91 | 418.42 | 491.32 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-07-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 00:00:00 | 510.80 | 427.88 | 494.85 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=12.53 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 492.00 | 430.98 | 498.48 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 548.70 | 459.12 | 525.03 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=13.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 00:00:00 | 575.54 | 464.08 | 539.52 | T1 booked 50% @ 575.54 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 548.70 | 468.89 | 548.38 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2023-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 00:00:00 | 563.50 | 473.03 | 545.50 | Stage2 pullback-breakout RSI=60 vol=2.4x ATR=14.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 00:00:00 | 593.26 | 476.28 | 555.37 | T1 booked 50% @ 593.26 |
| Target hit | 2024-01-23 00:00:00 | 739.30 | 570.15 | 757.14 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 948.25 | 676.84 | 875.30 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=28.23 |
| Stop hit — per-position SL triggered | 2024-04-03 00:00:00 | 905.91 | 681.65 | 883.22 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 00:00:00 | 513.60 | 2023-07-10 00:00:00 | 496.91 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest1 | 2023-07-27 00:00:00 | 510.80 | 2023-08-02 00:00:00 | 492.00 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest1 | 2023-10-06 00:00:00 | 548.70 | 2023-10-13 00:00:00 | 575.54 | PARTIAL | 0.50 | 4.89% |
| BUY | retest1 | 2023-10-06 00:00:00 | 548.70 | 2023-10-20 00:00:00 | 548.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-31 00:00:00 | 563.50 | 2023-11-03 00:00:00 | 593.26 | PARTIAL | 0.50 | 5.28% |
| BUY | retest1 | 2023-10-31 00:00:00 | 563.50 | 2024-01-23 00:00:00 | 739.30 | TARGET_HIT | 0.50 | 31.20% |
| BUY | retest1 | 2024-04-01 00:00:00 | 948.25 | 2024-04-03 00:00:00 | 905.91 | STOP_HIT | 1.00 | -4.47% |
