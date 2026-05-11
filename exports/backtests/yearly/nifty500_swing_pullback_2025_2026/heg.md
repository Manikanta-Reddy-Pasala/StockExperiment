# H.E.G. Ltd. (HEG)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 597.70
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 2 / 5 / 3
- **Avg / median % per leg:** 1.87% / 4.15%
- **Sum % (uncompounded):** 18.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 1.87% | 18.7% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 1.87% | 18.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 5 | 50.0% | 2 | 5 | 3 | 1.87% | 18.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 05:30:00 | 509.70 | 453.86 | 500.34 | Stage2 pullback-breakout RSI=56 vol=3.2x ATR=19.05 |
| Stop hit — per-position SL triggered | 2025-07-11 05:30:00 | 505.45 | 458.44 | 500.84 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-07-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 05:30:00 | 531.45 | 459.16 | 503.76 | Stage2 pullback-breakout RSI=65 vol=4.4x ATR=17.25 |
| Stop hit — per-position SL triggered | 2025-07-28 05:30:00 | 514.85 | 466.21 | 521.20 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-09-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 05:30:00 | 518.55 | 476.56 | 499.55 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=16.14 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 506.75 | 480.42 | 510.63 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-10-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 05:30:00 | 534.80 | 487.00 | 517.16 | Stage2 pullback-breakout RSI=61 vol=1.5x ATR=15.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 05:30:00 | 565.23 | 487.93 | 523.21 | T1 booked 50% @ 565.23 |
| Stop hit — per-position SL triggered | 2025-11-10 05:30:00 | 534.80 | 493.24 | 542.89 | SL hit (bars_held=8) |

### Cycle 5 — BUY (started 2025-12-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 05:30:00 | 539.00 | 501.88 | 529.94 | Stage2 pullback-breakout RSI=55 vol=1.6x ATR=15.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 05:30:00 | 569.24 | 502.93 | 534.56 | T1 booked 50% @ 569.24 |
| Target hit | 2026-01-09 05:30:00 | 574.55 | 513.29 | 582.39 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2026-02-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 05:30:00 | 564.75 | 522.34 | 547.32 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=24.40 |
| Stop hit — per-position SL triggered | 2026-03-09 05:30:00 | 528.15 | 524.86 | 548.93 | SL hit (bars_held=8) |

### Cycle 7 — BUY (started 2026-03-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 05:30:00 | 572.30 | 522.77 | 519.68 | Stage2 pullback-breakout RSI=62 vol=14.2x ATR=26.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 05:30:00 | 625.99 | 527.28 | 554.23 | T1 booked 50% @ 625.99 |
| Target hit | 2026-04-30 05:30:00 | 596.05 | 538.80 | 612.97 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-27 05:30:00 | 509.70 | 2025-07-11 05:30:00 | 505.45 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest1 | 2025-07-14 05:30:00 | 531.45 | 2025-07-28 05:30:00 | 514.85 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest1 | 2025-09-12 05:30:00 | 518.55 | 2025-09-26 05:30:00 | 506.75 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest1 | 2025-10-28 05:30:00 | 534.80 | 2025-10-29 05:30:00 | 565.23 | PARTIAL | 0.50 | 5.69% |
| BUY | retest1 | 2025-10-28 05:30:00 | 534.80 | 2025-11-10 05:30:00 | 534.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 05:30:00 | 539.00 | 2025-12-26 05:30:00 | 569.24 | PARTIAL | 0.50 | 5.61% |
| BUY | retest1 | 2025-12-23 05:30:00 | 539.00 | 2026-01-09 05:30:00 | 574.55 | TARGET_HIT | 0.50 | 6.60% |
| BUY | retest1 | 2026-02-24 05:30:00 | 564.75 | 2026-03-09 05:30:00 | 528.15 | STOP_HIT | 1.00 | -6.48% |
| BUY | retest1 | 2026-03-27 05:30:00 | 572.30 | 2026-04-16 05:30:00 | 625.99 | PARTIAL | 0.50 | 9.38% |
| BUY | retest1 | 2026-03-27 05:30:00 | 572.30 | 2026-04-30 05:30:00 | 596.05 | TARGET_HIT | 0.50 | 4.15% |
