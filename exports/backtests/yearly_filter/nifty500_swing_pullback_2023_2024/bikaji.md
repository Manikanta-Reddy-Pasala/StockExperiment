# Bikaji Foods International Ltd. (BIKAJI)

## Backtest Summary

- **Window:** 2022-11-16 00:00:00 → 2026-05-11 00:00:00 (863 bars)
- **Last close:** 658.60
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
- **Avg / median % per leg:** 1.51% / 2.62%
- **Sum % (uncompounded):** 10.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.51% | 10.6% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.51% | 10.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.51% | 10.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 512.35 | 405.70 | 483.65 | Stage2 pullback-breakout RSI=66 vol=3.5x ATR=16.36 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 487.81 | 411.08 | 496.03 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 00:00:00 | 491.30 | 425.65 | 484.72 | Stage2 pullback-breakout RSI=55 vol=1.5x ATR=13.72 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 470.72 | 429.02 | 483.80 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 499.15 | 432.48 | 479.53 | Stage2 pullback-breakout RSI=61 vol=2.4x ATR=14.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 00:00:00 | 527.50 | 435.10 | 490.39 | T1 booked 50% @ 527.50 |
| Target hit | 2023-12-07 00:00:00 | 524.30 | 453.33 | 533.99 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-12-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 00:00:00 | 559.20 | 464.02 | 541.21 | Stage2 pullback-breakout RSI=62 vol=2.9x ATR=16.58 |
| Stop hit — per-position SL triggered | 2024-01-10 00:00:00 | 573.85 | 473.42 | 556.51 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 526.00 | 498.21 | 504.29 | Stage2 pullback-breakout RSI=58 vol=4.1x ATR=16.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 00:00:00 | 558.63 | 501.05 | 522.59 | T1 booked 50% @ 558.63 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 526.00 | 501.31 | 522.99 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-05 00:00:00 | 512.35 | 2023-09-12 00:00:00 | 487.81 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest1 | 2023-10-16 00:00:00 | 491.30 | 2023-10-25 00:00:00 | 470.72 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest1 | 2023-11-06 00:00:00 | 499.15 | 2023-11-09 00:00:00 | 527.50 | PARTIAL | 0.50 | 5.68% |
| BUY | retest1 | 2023-11-06 00:00:00 | 499.15 | 2023-12-07 00:00:00 | 524.30 | TARGET_HIT | 0.50 | 5.04% |
| BUY | retest1 | 2023-12-27 00:00:00 | 559.20 | 2024-01-10 00:00:00 | 573.85 | STOP_HIT | 1.00 | 2.62% |
| BUY | retest1 | 2024-04-02 00:00:00 | 526.00 | 2024-04-12 00:00:00 | 558.63 | PARTIAL | 0.50 | 6.20% |
| BUY | retest1 | 2024-04-02 00:00:00 | 526.00 | 2024-04-15 00:00:00 | 526.00 | STOP_HIT | 0.50 | 0.00% |
