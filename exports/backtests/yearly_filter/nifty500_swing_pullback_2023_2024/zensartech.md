# Zensar Technolgies Ltd. (ZENSARTECH)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 516.65
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 3.16% / 2.21%
- **Sum % (uncompounded):** 28.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 1 | 5 | 3 | 3.16% | 28.5% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 5 | 3 | 3.16% | 28.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 1 | 5 | 3 | 3.16% | 28.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 00:00:00 | 408.05 | 296.38 | 388.48 | Stage2 pullback-breakout RSI=65 vol=5.4x ATR=14.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 00:00:00 | 436.43 | 299.08 | 396.56 | T1 booked 50% @ 436.43 |
| Target hit | 2023-08-30 00:00:00 | 496.90 | 351.32 | 499.25 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 526.95 | 353.07 | 501.88 | Stage2 pullback-breakout RSI=65 vol=2.3x ATR=17.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 00:00:00 | 562.24 | 358.68 | 512.76 | T1 booked 50% @ 562.24 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 526.95 | 367.74 | 524.67 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 515.25 | 419.08 | 500.43 | Stage2 pullback-breakout RSI=56 vol=2.4x ATR=18.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 00:00:00 | 551.38 | 422.53 | 509.83 | T1 booked 50% @ 551.38 |
| Stop hit — per-position SL triggered | 2023-12-07 00:00:00 | 526.65 | 434.15 | 525.48 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-02-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 00:00:00 | 592.15 | 481.68 | 570.86 | Stage2 pullback-breakout RSI=58 vol=3.3x ATR=21.11 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 560.48 | 484.57 | 572.88 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-03-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 00:00:00 | 591.00 | 495.01 | 552.21 | Stage2 pullback-breakout RSI=65 vol=3.2x ATR=20.28 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 560.58 | 496.48 | 555.24 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 620.75 | 520.55 | 590.73 | Stage2 pullback-breakout RSI=61 vol=9.8x ATR=23.60 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 585.35 | 527.72 | 602.25 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-13 00:00:00 | 408.05 | 2023-07-17 00:00:00 | 436.43 | PARTIAL | 0.50 | 6.95% |
| BUY | retest1 | 2023-07-13 00:00:00 | 408.05 | 2023-08-30 00:00:00 | 496.90 | TARGET_HIT | 0.50 | 21.77% |
| BUY | retest1 | 2023-08-31 00:00:00 | 526.95 | 2023-09-05 00:00:00 | 562.24 | PARTIAL | 0.50 | 6.70% |
| BUY | retest1 | 2023-08-31 00:00:00 | 526.95 | 2023-09-12 00:00:00 | 526.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-16 00:00:00 | 515.25 | 2023-11-21 00:00:00 | 551.38 | PARTIAL | 0.50 | 7.01% |
| BUY | retest1 | 2023-11-16 00:00:00 | 515.25 | 2023-12-07 00:00:00 | 526.65 | STOP_HIT | 0.50 | 2.21% |
| BUY | retest1 | 2024-02-06 00:00:00 | 592.15 | 2024-02-09 00:00:00 | 560.48 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest1 | 2024-03-07 00:00:00 | 591.00 | 2024-03-12 00:00:00 | 560.58 | STOP_HIT | 1.00 | -5.15% |
| BUY | retest1 | 2024-04-26 00:00:00 | 620.75 | 2024-05-09 00:00:00 | 585.35 | STOP_HIT | 1.00 | -5.70% |
