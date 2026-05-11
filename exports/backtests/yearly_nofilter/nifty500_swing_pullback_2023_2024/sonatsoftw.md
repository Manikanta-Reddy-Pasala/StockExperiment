# Sonata Software Ltd. (SONATSOFTW)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 287.60
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 2 / 6 / 3
- **Avg / median % per leg:** 1.37% / 0.73%
- **Sum % (uncompounded):** 15.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 2 | 6 | 3 | 1.37% | 15.0% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 6 | 3 | 1.37% | 15.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 2 | 6 | 3 | 1.37% | 15.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 526.83 | 390.94 | 499.23 | Stage2 pullback-breakout RSI=63 vol=7.6x ATR=17.74 |
| Stop hit — per-position SL triggered | 2023-07-28 00:00:00 | 530.70 | 403.83 | 516.46 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 533.60 | 428.86 | 516.92 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=14.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 00:00:00 | 563.44 | 434.94 | 531.62 | T1 booked 50% @ 563.44 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 533.60 | 437.15 | 534.38 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-10-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 00:00:00 | 588.78 | 462.98 | 545.72 | Stage2 pullback-breakout RSI=68 vol=3.9x ATR=21.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 00:00:00 | 631.67 | 472.20 | 574.68 | T1 booked 50% @ 631.67 |
| Target hit | 2023-11-28 00:00:00 | 626.83 | 496.83 | 638.13 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 684.43 | 501.85 | 645.85 | Stage2 pullback-breakout RSI=65 vol=1.9x ATR=24.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 00:00:00 | 733.94 | 506.12 | 659.02 | T1 booked 50% @ 733.94 |
| Target hit | 2024-01-01 00:00:00 | 722.90 | 544.77 | 727.50 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 755.50 | 559.22 | 719.99 | Stage2 pullback-breakout RSI=62 vol=4.4x ATR=24.01 |
| Stop hit — per-position SL triggered | 2024-01-24 00:00:00 | 719.49 | 574.63 | 741.22 | SL hit (bars_held=8) |

### Cycle 6 — BUY (started 2024-02-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 00:00:00 | 808.25 | 587.31 | 753.57 | Stage2 pullback-breakout RSI=68 vol=3.1x ATR=29.78 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 763.58 | 599.56 | 772.59 | SL hit (bars_held=6) |

### Cycle 7 — BUY (started 2024-02-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 00:00:00 | 842.20 | 607.72 | 784.82 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=33.35 |
| Stop hit — per-position SL triggered | 2024-03-02 00:00:00 | 822.15 | 629.19 | 814.57 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2024-04-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 00:00:00 | 760.45 | 654.60 | 752.58 | Stage2 pullback-breakout RSI=51 vol=2.1x ATR=26.50 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 720.69 | 657.72 | 745.96 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 00:00:00 | 526.83 | 2023-07-28 00:00:00 | 530.70 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest1 | 2023-09-04 00:00:00 | 533.60 | 2023-09-11 00:00:00 | 563.44 | PARTIAL | 0.50 | 5.59% |
| BUY | retest1 | 2023-09-04 00:00:00 | 533.60 | 2023-09-13 00:00:00 | 533.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-27 00:00:00 | 588.78 | 2023-11-07 00:00:00 | 631.67 | PARTIAL | 0.50 | 7.28% |
| BUY | retest1 | 2023-10-27 00:00:00 | 588.78 | 2023-11-28 00:00:00 | 626.83 | TARGET_HIT | 0.50 | 6.46% |
| BUY | retest1 | 2023-12-01 00:00:00 | 684.43 | 2023-12-05 00:00:00 | 733.94 | PARTIAL | 0.50 | 7.23% |
| BUY | retest1 | 2023-12-01 00:00:00 | 684.43 | 2024-01-01 00:00:00 | 722.90 | TARGET_HIT | 0.50 | 5.62% |
| BUY | retest1 | 2024-01-12 00:00:00 | 755.50 | 2024-01-24 00:00:00 | 719.49 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest1 | 2024-02-05 00:00:00 | 808.25 | 2024-02-13 00:00:00 | 763.58 | STOP_HIT | 1.00 | -5.53% |
| BUY | retest1 | 2024-02-19 00:00:00 | 842.20 | 2024-03-02 00:00:00 | 822.15 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest1 | 2024-04-08 00:00:00 | 760.45 | 2024-04-15 00:00:00 | 720.69 | STOP_HIT | 1.00 | -5.23% |
