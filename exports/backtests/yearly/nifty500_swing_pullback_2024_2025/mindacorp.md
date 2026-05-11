# Minda Corporation Ltd. (MINDACORP)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 537.50
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
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.78% / 0.49%
- **Sum % (uncompounded):** 3.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.78% | 3.9% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.78% | 3.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 3 | 1 | 0.78% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 05:30:00 | 514.95 | 420.30 | 485.09 | Stage2 pullback-breakout RSI=67 vol=4.0x ATR=15.21 |
| Stop hit — per-position SL triggered | 2024-08-16 05:30:00 | 517.45 | 429.37 | 504.28 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 05:30:00 | 587.95 | 459.19 | 552.15 | Stage2 pullback-breakout RSI=60 vol=3.8x ATR=28.01 |
| Stop hit — per-position SL triggered | 2024-10-08 05:30:00 | 561.95 | 472.14 | 577.23 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-12-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 05:30:00 | 526.70 | 485.13 | 503.41 | Stage2 pullback-breakout RSI=59 vol=3.5x ATR=18.16 |
| Stop hit — per-position SL triggered | 2024-12-20 05:30:00 | 508.20 | 488.99 | 516.05 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-01-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 05:30:00 | 521.50 | 489.71 | 507.96 | Stage2 pullback-breakout RSI=56 vol=2.3x ATR=18.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 05:30:00 | 557.76 | 493.25 | 522.09 | T1 booked 50% @ 557.76 |
| Target hit | 2025-01-27 05:30:00 | 544.50 | 499.58 | 550.11 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-01 05:30:00 | 514.95 | 2024-08-16 05:30:00 | 517.45 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest1 | 2024-09-23 05:30:00 | 587.95 | 2024-10-08 05:30:00 | 561.95 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest1 | 2024-12-06 05:30:00 | 526.70 | 2024-12-20 05:30:00 | 508.20 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2025-01-01 05:30:00 | 521.50 | 2025-01-15 05:30:00 | 557.76 | PARTIAL | 0.50 | 6.95% |
| BUY | retest1 | 2025-01-01 05:30:00 | 521.50 | 2025-01-27 05:30:00 | 544.50 | TARGET_HIT | 0.50 | 4.41% |
