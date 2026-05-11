# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 742.40
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
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 1 / 6 / 3
- **Avg / median % per leg:** 1.12% / 1.63%
- **Sum % (uncompounded):** 11.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 1 | 6 | 3 | 1.12% | 11.2% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 1 | 6 | 3 | 1.12% | 11.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 5 | 50.0% | 1 | 6 | 3 | 1.12% | 11.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 05:30:00 | 319.40 | 266.13 | 310.58 | Stage2 pullback-breakout RSI=58 vol=4.2x ATR=11.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 05:30:00 | 341.97 | 272.79 | 322.17 | T1 booked 50% @ 341.97 |
| Target hit | 2023-09-20 05:30:00 | 326.40 | 278.91 | 331.33 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 05:30:00 | 338.80 | 282.72 | 330.02 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=10.19 |
| Stop hit — per-position SL triggered | 2023-10-09 05:30:00 | 323.52 | 284.45 | 328.86 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-10-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 05:30:00 | 339.20 | 285.78 | 329.08 | Stage2 pullback-breakout RSI=58 vol=4.8x ATR=10.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 05:30:00 | 360.77 | 289.59 | 332.95 | T1 booked 50% @ 360.77 |
| Stop hit — per-position SL triggered | 2023-10-26 05:30:00 | 339.20 | 290.03 | 333.10 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-11-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 05:30:00 | 349.10 | 296.85 | 336.77 | Stage2 pullback-breakout RSI=61 vol=3.9x ATR=11.48 |
| Stop hit — per-position SL triggered | 2023-11-28 05:30:00 | 331.88 | 299.18 | 336.73 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2024-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 05:30:00 | 417.45 | 323.00 | 397.46 | Stage2 pullback-breakout RSI=68 vol=2.8x ATR=12.91 |
| Stop hit — per-position SL triggered | 2024-01-15 05:30:00 | 398.08 | 327.89 | 401.54 | SL hit (bars_held=6) |

### Cycle 6 — BUY (started 2024-01-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 05:30:00 | 425.15 | 328.85 | 403.79 | Stage2 pullback-breakout RSI=66 vol=8.2x ATR=15.00 |
| Stop hit — per-position SL triggered | 2024-02-01 05:30:00 | 432.10 | 339.66 | 422.99 | Time-stop (10d <3%) |

### Cycle 7 — BUY (started 2024-04-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 05:30:00 | 458.65 | 375.74 | 431.50 | Stage2 pullback-breakout RSI=60 vol=5.0x ATR=18.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 05:30:00 | 495.29 | 377.63 | 438.74 | T1 booked 50% @ 495.29 |
| Stop hit — per-position SL triggered | 2024-04-23 05:30:00 | 458.65 | 387.53 | 469.52 | SL hit (bars_held=11) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-18 05:30:00 | 319.40 | 2023-09-05 05:30:00 | 341.97 | PARTIAL | 0.50 | 7.07% |
| BUY | retest1 | 2023-08-18 05:30:00 | 319.40 | 2023-09-20 05:30:00 | 326.40 | TARGET_HIT | 0.50 | 2.19% |
| BUY | retest1 | 2023-10-03 05:30:00 | 338.80 | 2023-10-09 05:30:00 | 323.52 | STOP_HIT | 1.00 | -4.51% |
| BUY | retest1 | 2023-10-12 05:30:00 | 339.20 | 2023-10-25 05:30:00 | 360.77 | PARTIAL | 0.50 | 6.36% |
| BUY | retest1 | 2023-10-12 05:30:00 | 339.20 | 2023-10-26 05:30:00 | 339.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-17 05:30:00 | 349.10 | 2023-11-28 05:30:00 | 331.88 | STOP_HIT | 1.00 | -4.93% |
| BUY | retest1 | 2024-01-05 05:30:00 | 417.45 | 2024-01-15 05:30:00 | 398.08 | STOP_HIT | 1.00 | -4.64% |
| BUY | retest1 | 2024-01-16 05:30:00 | 425.15 | 2024-02-01 05:30:00 | 432.10 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest1 | 2024-04-04 05:30:00 | 458.65 | 2024-04-08 05:30:00 | 495.29 | PARTIAL | 0.50 | 7.99% |
| BUY | retest1 | 2024-04-04 05:30:00 | 458.65 | 2024-04-23 05:30:00 | 458.65 | STOP_HIT | 0.50 | 0.00% |
