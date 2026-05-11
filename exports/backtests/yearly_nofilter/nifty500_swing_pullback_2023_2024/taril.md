# Transformers And Rectifiers (India) Ltd. (TARIL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (909 bars)
- **Last close:** 314.80
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 3 / 1 / 3
- **Avg / median % per leg:** 14.84% / 10.80%
- **Sum % (uncompounded):** 103.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 3 | 1 | 3 | 14.84% | 103.9% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 3 | 1 | 3 | 14.84% | 103.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 3 | 1 | 3 | 14.84% | 103.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 00:00:00 | 63.18 | 40.01 | 57.31 | Stage2 pullback-breakout RSI=67 vol=1.7x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 70.00 | 40.33 | 58.70 | T1 booked 50% @ 70.00 |
| Target hit | 2023-10-20 00:00:00 | 80.15 | 50.67 | 82.04 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 00:00:00 | 85.35 | 52.05 | 81.03 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 00:00:00 | 94.99 | 55.79 | 86.14 | T1 booked 50% @ 94.99 |
| Target hit | 2023-11-30 00:00:00 | 92.13 | 60.15 | 93.46 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 180.70 | 95.75 | 163.23 | Stage2 pullback-breakout RSI=65 vol=1.5x ATR=10.36 |
| Stop hit — per-position SL triggered | 2024-02-29 00:00:00 | 165.17 | 98.09 | 166.22 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 197.83 | 110.70 | 171.89 | Stage2 pullback-breakout RSI=69 vol=2.7x ATR=9.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 00:00:00 | 217.51 | 112.68 | 178.95 | T1 booked 50% @ 217.51 |
| Target hit | 2024-05-07 00:00:00 | 288.08 | 147.64 | 293.46 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-07 00:00:00 | 63.18 | 2023-09-08 00:00:00 | 70.00 | PARTIAL | 0.50 | 10.80% |
| BUY | retest1 | 2023-09-07 00:00:00 | 63.18 | 2023-10-20 00:00:00 | 80.15 | TARGET_HIT | 0.50 | 26.86% |
| BUY | retest1 | 2023-10-30 00:00:00 | 85.35 | 2023-11-13 00:00:00 | 94.99 | PARTIAL | 0.50 | 11.29% |
| BUY | retest1 | 2023-10-30 00:00:00 | 85.35 | 2023-11-30 00:00:00 | 92.13 | TARGET_HIT | 0.50 | 7.94% |
| BUY | retest1 | 2024-02-26 00:00:00 | 180.70 | 2024-02-29 00:00:00 | 165.17 | STOP_HIT | 1.00 | -8.60% |
| BUY | retest1 | 2024-03-28 00:00:00 | 197.83 | 2024-04-02 00:00:00 | 217.51 | PARTIAL | 0.50 | 9.95% |
| BUY | retest1 | 2024-03-28 00:00:00 | 197.83 | 2024-05-07 00:00:00 | 288.08 | TARGET_HIT | 0.50 | 45.62% |
