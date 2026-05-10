# Graphite India Ltd. (GRAPHITE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 751.75
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 6
- **Target hits / Stop hits / Partials:** 1 / 7 / 3
- **Avg / median % per leg:** 0.42% / 0.00%
- **Sum % (uncompounded):** 4.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 1 | 7 | 3 | 0.42% | 4.6% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 1 | 7 | 3 | 0.42% | 4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 5 | 45.5% | 1 | 7 | 3 | 0.42% | 4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 05:30:00 | 585.30 | 511.49 | 559.43 | Stage2 pullback-breakout RSI=64 vol=3.9x ATR=17.80 |
| Stop hit — per-position SL triggered | 2025-07-22 05:30:00 | 558.60 | 515.37 | 567.38 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2025-09-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 05:30:00 | 565.45 | 521.83 | 535.12 | Stage2 pullback-breakout RSI=68 vol=3.4x ATR=15.41 |
| Stop hit — per-position SL triggered | 2025-10-03 05:30:00 | 542.34 | 525.58 | 550.33 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2025-10-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 05:30:00 | 590.35 | 527.11 | 557.54 | Stage2 pullback-breakout RSI=69 vol=9.1x ATR=18.56 |
| Stop hit — per-position SL triggered | 2025-10-10 05:30:00 | 562.52 | 528.09 | 560.88 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-10-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 05:30:00 | 590.30 | 531.26 | 560.69 | Stage2 pullback-breakout RSI=66 vol=1.8x ATR=16.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 05:30:00 | 623.36 | 532.24 | 567.34 | T1 booked 50% @ 623.36 |
| Stop hit — per-position SL triggered | 2025-11-06 05:30:00 | 590.30 | 536.34 | 585.85 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2025-11-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 05:30:00 | 596.40 | 539.01 | 578.76 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=24.40 |
| Stop hit — per-position SL triggered | 2025-11-20 05:30:00 | 559.80 | 539.53 | 576.34 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2025-12-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 05:30:00 | 564.30 | 540.63 | 547.32 | Stage2 pullback-breakout RSI=57 vol=2.7x ATR=16.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 05:30:00 | 596.99 | 542.23 | 559.02 | T1 booked 50% @ 596.99 |
| Target hit | 2026-01-13 05:30:00 | 599.30 | 552.22 | 610.53 | Trail-exit close<EMA20 |

### Cycle 7 — BUY (started 2026-02-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 05:30:00 | 668.25 | 565.79 | 628.56 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=30.57 |
| Stop hit — per-position SL triggered | 2026-02-24 05:30:00 | 683.65 | 576.43 | 661.24 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2026-02-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 05:30:00 | 730.15 | 579.37 | 672.80 | Stage2 pullback-breakout RSI=67 vol=1.7x ATR=29.91 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 685.28 | 582.01 | 680.00 | SL hit (bars_held=2) |

### Cycle 9 — BUY (started 2026-03-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 05:30:00 | 640.20 | 588.65 | 626.90 | Stage2 pullback-breakout RSI=52 vol=8.2x ATR=33.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 05:30:00 | 706.29 | 596.34 | 650.41 | T1 booked 50% @ 706.29 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-14 05:30:00 | 585.30 | 2025-07-22 05:30:00 | 558.60 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest1 | 2025-09-18 05:30:00 | 565.45 | 2025-10-03 05:30:00 | 542.34 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest1 | 2025-10-08 05:30:00 | 590.35 | 2025-10-10 05:30:00 | 562.52 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest1 | 2025-10-28 05:30:00 | 590.30 | 2025-10-29 05:30:00 | 623.36 | PARTIAL | 0.50 | 5.60% |
| BUY | retest1 | 2025-10-28 05:30:00 | 590.30 | 2025-11-06 05:30:00 | 590.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-18 05:30:00 | 596.40 | 2025-11-20 05:30:00 | 559.80 | STOP_HIT | 1.00 | -6.14% |
| BUY | retest1 | 2025-12-22 05:30:00 | 564.30 | 2025-12-29 05:30:00 | 596.99 | PARTIAL | 0.50 | 5.79% |
| BUY | retest1 | 2025-12-22 05:30:00 | 564.30 | 2026-01-13 05:30:00 | 599.30 | TARGET_HIT | 0.50 | 6.20% |
| BUY | retest1 | 2026-02-10 05:30:00 | 668.25 | 2026-02-24 05:30:00 | 683.65 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest1 | 2026-02-26 05:30:00 | 730.15 | 2026-03-02 05:30:00 | 685.28 | STOP_HIT | 1.00 | -6.15% |
| BUY | retest1 | 2026-03-27 05:30:00 | 640.20 | 2026-04-20 05:30:00 | 706.29 | PARTIAL | 0.50 | 10.32% |
