# LIC Housing Finance Ltd. (LICHSGFIN)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 581.50
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 3
- **Target hits / Stop hits / Partials:** 3 / 4 / 4
- **Avg / median % per leg:** 4.09% / 4.73%
- **Sum % (uncompounded):** 45.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 8 | 72.7% | 3 | 4 | 4 | 4.09% | 45.0% |
| BUY @ 2nd Alert (retest1) | 11 | 8 | 72.7% | 3 | 4 | 4 | 4.09% | 45.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 8 | 72.7% | 3 | 4 | 4 | 4.09% | 45.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 05:30:00 | 413.45 | 383.30 | 392.52 | Stage2 pullback-breakout RSI=69 vol=2.9x ATR=8.79 |
| Stop hit — per-position SL triggered | 2023-08-03 05:30:00 | 400.26 | 384.39 | 398.31 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2023-08-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 05:30:00 | 426.65 | 384.81 | 401.01 | Stage2 pullback-breakout RSI=65 vol=7.7x ATR=12.60 |
| Stop hit — per-position SL triggered | 2023-08-21 05:30:00 | 420.55 | 388.70 | 415.66 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-09-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 05:30:00 | 447.80 | 392.20 | 422.90 | Stage2 pullback-breakout RSI=70 vol=2.6x ATR=10.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 05:30:00 | 469.38 | 396.75 | 438.80 | T1 booked 50% @ 469.38 |
| Stop hit — per-position SL triggered | 2023-09-18 05:30:00 | 455.90 | 397.95 | 442.15 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-10-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 05:30:00 | 477.30 | 403.56 | 455.24 | Stage2 pullback-breakout RSI=69 vol=1.7x ATR=11.78 |
| Stop hit — per-position SL triggered | 2023-10-09 05:30:00 | 459.63 | 406.10 | 459.50 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2023-11-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 05:30:00 | 466.15 | 418.14 | 456.73 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=11.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 05:30:00 | 488.19 | 422.60 | 462.96 | T1 booked 50% @ 488.19 |
| Target hit | 2024-01-23 05:30:00 | 560.75 | 461.02 | 562.48 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-01-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 05:30:00 | 600.00 | 464.70 | 568.66 | Stage2 pullback-breakout RSI=68 vol=2.5x ATR=17.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 05:30:00 | 634.45 | 469.41 | 583.17 | T1 booked 50% @ 634.45 |
| Target hit | 2024-03-06 05:30:00 | 640.15 | 507.91 | 640.76 | Trail-exit close<EMA20 |

### Cycle 7 — BUY (started 2024-03-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 05:30:00 | 610.95 | 519.53 | 605.89 | Stage2 pullback-breakout RSI=51 vol=1.7x ATR=16.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 05:30:00 | 644.40 | 524.26 | 617.29 | T1 booked 50% @ 644.40 |
| Target hit | 2024-05-06 05:30:00 | 634.05 | 546.94 | 651.12 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-28 05:30:00 | 413.45 | 2023-08-03 05:30:00 | 400.26 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest1 | 2023-08-04 05:30:00 | 426.65 | 2023-08-21 05:30:00 | 420.55 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest1 | 2023-09-04 05:30:00 | 447.80 | 2023-09-14 05:30:00 | 469.38 | PARTIAL | 0.50 | 4.82% |
| BUY | retest1 | 2023-09-04 05:30:00 | 447.80 | 2023-09-18 05:30:00 | 455.90 | STOP_HIT | 0.50 | 1.81% |
| BUY | retest1 | 2023-10-03 05:30:00 | 477.30 | 2023-10-09 05:30:00 | 459.63 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest1 | 2023-11-15 05:30:00 | 466.15 | 2023-11-30 05:30:00 | 488.19 | PARTIAL | 0.50 | 4.73% |
| BUY | retest1 | 2023-11-15 05:30:00 | 466.15 | 2024-01-23 05:30:00 | 560.75 | TARGET_HIT | 0.50 | 20.29% |
| BUY | retest1 | 2024-01-29 05:30:00 | 600.00 | 2024-02-01 05:30:00 | 634.45 | PARTIAL | 0.50 | 5.74% |
| BUY | retest1 | 2024-01-29 05:30:00 | 600.00 | 2024-03-06 05:30:00 | 640.15 | TARGET_HIT | 0.50 | 6.69% |
| BUY | retest1 | 2024-03-28 05:30:00 | 610.95 | 2024-04-04 05:30:00 | 644.40 | PARTIAL | 0.50 | 5.48% |
| BUY | retest1 | 2024-03-28 05:30:00 | 610.95 | 2024-05-06 05:30:00 | 634.05 | TARGET_HIT | 0.50 | 3.78% |
