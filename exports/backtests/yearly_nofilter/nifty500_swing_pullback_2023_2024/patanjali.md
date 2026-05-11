# Patanjali Foods Ltd. (PATANJALI)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 456.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 2.30% / 1.56%
- **Sum % (uncompounded):** 20.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 2.30% | 20.7% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 2.30% | 20.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 2 | 4 | 3 | 2.30% | 20.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 00:00:00 | 436.80 | 390.79 | 425.89 | Stage2 pullback-breakout RSI=59 vol=3.5x ATR=12.37 |
| Stop hit — per-position SL triggered | 2023-09-26 00:00:00 | 420.30 | 394.48 | 427.64 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 00:00:00 | 433.05 | 396.31 | 421.22 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=12.05 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 414.97 | 400.14 | 429.51 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2023-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 00:00:00 | 455.97 | 401.45 | 431.38 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=14.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 00:00:00 | 485.82 | 405.25 | 448.18 | T1 booked 50% @ 485.82 |
| Target hit | 2023-11-20 00:00:00 | 463.10 | 410.80 | 463.64 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 468.20 | 413.22 | 458.79 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=14.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 00:00:00 | 497.36 | 415.74 | 465.47 | T1 booked 50% @ 497.36 |
| Target hit | 2024-01-17 00:00:00 | 522.07 | 445.71 | 531.96 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-02-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 00:00:00 | 544.13 | 456.03 | 527.33 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=16.74 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 519.02 | 458.42 | 529.53 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2024-04-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 00:00:00 | 488.80 | 468.68 | 466.51 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=18.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 00:00:00 | 525.02 | 470.24 | 480.76 | T1 booked 50% @ 525.02 |
| Stop hit — per-position SL triggered | 2024-05-02 00:00:00 | 488.80 | 471.20 | 486.17 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-11 00:00:00 | 436.80 | 2023-09-26 00:00:00 | 420.30 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest1 | 2023-10-10 00:00:00 | 433.05 | 2023-10-25 00:00:00 | 414.97 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest1 | 2023-10-31 00:00:00 | 455.97 | 2023-11-08 00:00:00 | 485.82 | PARTIAL | 0.50 | 6.55% |
| BUY | retest1 | 2023-10-31 00:00:00 | 455.97 | 2023-11-20 00:00:00 | 463.10 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2023-11-29 00:00:00 | 468.20 | 2023-12-05 00:00:00 | 497.36 | PARTIAL | 0.50 | 6.23% |
| BUY | retest1 | 2023-11-29 00:00:00 | 468.20 | 2024-01-17 00:00:00 | 522.07 | TARGET_HIT | 0.50 | 11.51% |
| BUY | retest1 | 2024-02-07 00:00:00 | 544.13 | 2024-02-12 00:00:00 | 519.02 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest1 | 2024-04-22 00:00:00 | 488.80 | 2024-04-26 00:00:00 | 525.02 | PARTIAL | 0.50 | 7.41% |
| BUY | retest1 | 2024-04-22 00:00:00 | 488.80 | 2024-05-02 00:00:00 | 488.80 | STOP_HIT | 0.50 | 0.00% |
