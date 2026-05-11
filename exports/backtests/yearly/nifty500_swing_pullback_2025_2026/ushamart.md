# Usha Martin Ltd. (USHAMART)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 472.85
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
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 1.68% / 1.31%
- **Sum % (uncompounded):** 11.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.68% | 11.7% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.68% | 11.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.68% | 11.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 05:30:00 | 371.15 | 342.89 | 361.21 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=13.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 05:30:00 | 397.31 | 345.74 | 372.28 | T1 booked 50% @ 397.31 |
| Stop hit — per-position SL triggered | 2025-09-03 05:30:00 | 376.00 | 346.40 | 373.53 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-09-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 05:30:00 | 407.10 | 349.91 | 382.39 | Stage2 pullback-breakout RSI=69 vol=6.9x ATR=13.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 05:30:00 | 433.92 | 352.63 | 394.91 | T1 booked 50% @ 433.92 |
| Target hit | 2025-10-24 05:30:00 | 448.40 | 373.58 | 452.32 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-11-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 05:30:00 | 481.30 | 382.55 | 463.13 | Stage2 pullback-breakout RSI=61 vol=3.7x ATR=17.60 |
| Stop hit — per-position SL triggered | 2025-11-13 05:30:00 | 454.90 | 384.89 | 462.58 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2025-12-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 05:30:00 | 461.50 | 394.52 | 437.19 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=16.50 |
| Stop hit — per-position SL triggered | 2025-12-30 05:30:00 | 450.90 | 400.02 | 446.53 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-02-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 05:30:00 | 425.75 | 407.42 | 415.92 | Stage2 pullback-breakout RSI=54 vol=10.5x ATR=15.77 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 402.09 | 407.84 | 417.37 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-19 05:30:00 | 371.15 | 2025-09-01 05:30:00 | 397.31 | PARTIAL | 0.50 | 7.05% |
| BUY | retest1 | 2025-08-19 05:30:00 | 371.15 | 2025-09-03 05:30:00 | 376.00 | STOP_HIT | 0.50 | 1.31% |
| BUY | retest1 | 2025-09-16 05:30:00 | 407.10 | 2025-09-22 05:30:00 | 433.92 | PARTIAL | 0.50 | 6.59% |
| BUY | retest1 | 2025-09-16 05:30:00 | 407.10 | 2025-10-24 05:30:00 | 448.40 | TARGET_HIT | 0.50 | 10.14% |
| BUY | retest1 | 2025-11-10 05:30:00 | 481.30 | 2025-11-13 05:30:00 | 454.90 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest1 | 2025-12-15 05:30:00 | 461.50 | 2025-12-30 05:30:00 | 450.90 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest1 | 2026-02-25 05:30:00 | 425.75 | 2026-03-02 05:30:00 | 402.09 | STOP_HIT | 1.00 | -5.56% |
