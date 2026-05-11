# Indian Hotels Co. Ltd. (INDHOTEL)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 673.05
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 2 / 2 / 4
- **Avg / median % per leg:** 4.30% / 4.55%
- **Sum % (uncompounded):** 34.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 7 | 87.5% | 2 | 2 | 4 | 4.30% | 34.4% |
| BUY @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 2 | 2 | 4 | 4.30% | 34.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 7 | 87.5% | 2 | 2 | 4 | 4.30% | 34.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 05:30:00 | 401.10 | 355.24 | 388.42 | Stage2 pullback-breakout RSI=62 vol=2.6x ATR=9.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 05:30:00 | 419.13 | 358.03 | 395.71 | T1 booked 50% @ 419.13 |
| Target hit | 2023-09-21 05:30:00 | 406.80 | 366.31 | 413.74 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-01-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 05:30:00 | 451.45 | 393.39 | 435.90 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=10.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 05:30:00 | 472.00 | 398.57 | 450.13 | T1 booked 50% @ 472.00 |
| Stop hit — per-position SL triggered | 2024-01-17 05:30:00 | 461.95 | 399.89 | 452.83 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 05:30:00 | 496.15 | 405.18 | 466.13 | Stage2 pullback-breakout RSI=68 vol=2.7x ATR=13.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 05:30:00 | 524.13 | 411.80 | 485.66 | T1 booked 50% @ 524.13 |
| Target hit | 2024-03-13 05:30:00 | 544.70 | 445.32 | 563.72 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 05:30:00 | 585.85 | 455.33 | 563.32 | Stage2 pullback-breakout RSI=61 vol=2.0x ATR=18.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 05:30:00 | 622.06 | 462.68 | 580.67 | T1 booked 50% @ 622.06 |
| Stop hit — per-position SL triggered | 2024-04-08 05:30:00 | 585.85 | 465.59 | 585.81 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-23 05:30:00 | 401.10 | 2023-08-31 05:30:00 | 419.13 | PARTIAL | 0.50 | 4.50% |
| BUY | retest1 | 2023-08-23 05:30:00 | 401.10 | 2023-09-21 05:30:00 | 406.80 | TARGET_HIT | 0.50 | 1.42% |
| BUY | retest1 | 2024-01-03 05:30:00 | 451.45 | 2024-01-15 05:30:00 | 472.00 | PARTIAL | 0.50 | 4.55% |
| BUY | retest1 | 2024-01-03 05:30:00 | 451.45 | 2024-01-17 05:30:00 | 461.95 | STOP_HIT | 0.50 | 2.33% |
| BUY | retest1 | 2024-01-29 05:30:00 | 496.15 | 2024-02-07 05:30:00 | 524.13 | PARTIAL | 0.50 | 5.64% |
| BUY | retest1 | 2024-01-29 05:30:00 | 496.15 | 2024-03-13 05:30:00 | 544.70 | TARGET_HIT | 0.50 | 9.79% |
| BUY | retest1 | 2024-03-27 05:30:00 | 585.85 | 2024-04-04 05:30:00 | 622.06 | PARTIAL | 0.50 | 6.18% |
| BUY | retest1 | 2024-03-27 05:30:00 | 585.85 | 2024-04-08 05:30:00 | 585.85 | STOP_HIT | 0.50 | 0.00% |
