# Fortis Healthcare Ltd. (FORTIS)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 952.35
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 2
- **Avg / median % per leg:** -0.05% / -2.83%
- **Sum % (uncompounded):** -0.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.05% | -0.5% |
| BUY @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.05% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.05% | -0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 05:30:00 | 503.40 | 425.84 | 480.97 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=16.90 |
| Stop hit — per-position SL triggered | 2024-08-08 05:30:00 | 489.15 | 432.84 | 491.77 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 05:30:00 | 517.10 | 434.89 | 494.69 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=18.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 05:30:00 | 553.69 | 441.38 | 513.53 | T1 booked 50% @ 553.69 |
| Target hit | 2024-10-04 05:30:00 | 581.40 | 475.92 | 587.40 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 05:30:00 | 643.20 | 507.78 | 615.67 | Stage2 pullback-breakout RSI=64 vol=2.7x ATR=19.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 05:30:00 | 682.00 | 514.70 | 630.01 | T1 booked 50% @ 682.00 |
| Stop hit — per-position SL triggered | 2024-11-27 05:30:00 | 643.20 | 520.91 | 643.70 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-12-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 05:30:00 | 725.80 | 554.94 | 687.27 | Stage2 pullback-breakout RSI=65 vol=2.1x ATR=24.65 |
| Stop hit — per-position SL triggered | 2025-01-10 05:30:00 | 688.82 | 568.92 | 704.34 | SL hit (bars_held=9) |

### Cycle 5 — BUY (started 2025-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 05:30:00 | 669.70 | 580.56 | 644.63 | Stage2 pullback-breakout RSI=55 vol=1.7x ATR=27.95 |
| Stop hit — per-position SL triggered | 2025-02-10 05:30:00 | 627.77 | 582.40 | 643.93 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2025-03-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 05:30:00 | 639.10 | 586.73 | 622.01 | Stage2 pullback-breakout RSI=54 vol=1.5x ATR=25.46 |
| Stop hit — per-position SL triggered | 2025-03-13 05:30:00 | 600.91 | 589.61 | 624.24 | SL hit (bars_held=7) |

### Cycle 7 — BUY (started 2025-03-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 05:30:00 | 698.35 | 594.44 | 638.89 | Stage2 pullback-breakout RSI=68 vol=2.6x ATR=27.31 |
| Stop hit — per-position SL triggered | 2025-04-02 05:30:00 | 657.39 | 596.05 | 645.44 | SL hit (bars_held=2) |

### Cycle 8 — BUY (started 2025-04-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 05:30:00 | 685.55 | 606.18 | 661.25 | Stage2 pullback-breakout RSI=58 vol=2.2x ATR=28.19 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-25 05:30:00 | 503.40 | 2024-08-08 05:30:00 | 489.15 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest1 | 2024-08-13 05:30:00 | 517.10 | 2024-08-23 05:30:00 | 553.69 | PARTIAL | 0.50 | 7.08% |
| BUY | retest1 | 2024-08-13 05:30:00 | 517.10 | 2024-10-04 05:30:00 | 581.40 | TARGET_HIT | 0.50 | 12.43% |
| BUY | retest1 | 2024-11-12 05:30:00 | 643.20 | 2024-11-21 05:30:00 | 682.00 | PARTIAL | 0.50 | 6.03% |
| BUY | retest1 | 2024-11-12 05:30:00 | 643.20 | 2024-11-27 05:30:00 | 643.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 05:30:00 | 725.80 | 2025-01-10 05:30:00 | 688.82 | STOP_HIT | 1.00 | -5.10% |
| BUY | retest1 | 2025-02-05 05:30:00 | 669.70 | 2025-02-10 05:30:00 | 627.77 | STOP_HIT | 1.00 | -6.26% |
| BUY | retest1 | 2025-03-04 05:30:00 | 639.10 | 2025-03-13 05:30:00 | 600.91 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest1 | 2025-03-28 05:30:00 | 698.35 | 2025-04-02 05:30:00 | 657.39 | STOP_HIT | 1.00 | -5.87% |
