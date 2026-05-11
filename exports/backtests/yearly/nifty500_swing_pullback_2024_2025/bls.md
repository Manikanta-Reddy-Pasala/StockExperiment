# BLS International Services Ltd. (BLS)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 289.30
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 2.65% / 7.31%
- **Sum % (uncompounded):** 21.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 2.65% | 21.2% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 2.65% | 21.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 1 | 4 | 3 | 2.65% | 21.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 05:30:00 | 372.45 | 325.53 | 347.75 | Stage2 pullback-breakout RSI=64 vol=3.4x ATR=13.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 05:30:00 | 399.68 | 328.90 | 361.68 | T1 booked 50% @ 399.68 |
| Stop hit — per-position SL triggered | 2024-07-11 05:30:00 | 372.45 | 329.33 | 362.74 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-08-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 05:30:00 | 385.10 | 333.34 | 357.92 | Stage2 pullback-breakout RSI=63 vol=7.4x ATR=15.97 |
| Stop hit — per-position SL triggered | 2024-08-14 05:30:00 | 361.15 | 336.14 | 367.85 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-10-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 05:30:00 | 397.85 | 359.87 | 378.32 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=17.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 05:30:00 | 432.74 | 363.84 | 396.61 | T1 booked 50% @ 432.74 |
| Stop hit — per-position SL triggered | 2024-11-13 05:30:00 | 397.85 | 364.04 | 395.42 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-12-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 05:30:00 | 419.25 | 367.16 | 394.21 | Stage2 pullback-breakout RSI=63 vol=4.2x ATR=15.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 05:30:00 | 450.39 | 370.40 | 410.53 | T1 booked 50% @ 450.39 |
| Target hit | 2025-01-09 05:30:00 | 467.20 | 391.31 | 473.97 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2025-01-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 05:30:00 | 501.85 | 394.72 | 464.27 | Stage2 pullback-breakout RSI=61 vol=8.7x ATR=25.06 |
| Stop hit — per-position SL triggered | 2025-01-20 05:30:00 | 464.27 | 395.46 | 464.78 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-01 05:30:00 | 372.45 | 2024-07-10 05:30:00 | 399.68 | PARTIAL | 0.50 | 7.31% |
| BUY | retest1 | 2024-07-01 05:30:00 | 372.45 | 2024-07-11 05:30:00 | 372.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-06 05:30:00 | 385.10 | 2024-08-14 05:30:00 | 361.15 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest1 | 2024-10-31 05:30:00 | 397.85 | 2024-11-12 05:30:00 | 432.74 | PARTIAL | 0.50 | 8.77% |
| BUY | retest1 | 2024-10-31 05:30:00 | 397.85 | 2024-11-13 05:30:00 | 397.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 05:30:00 | 419.25 | 2024-12-10 05:30:00 | 450.39 | PARTIAL | 0.50 | 7.43% |
| BUY | retest1 | 2024-12-03 05:30:00 | 419.25 | 2025-01-09 05:30:00 | 467.20 | TARGET_HIT | 0.50 | 11.44% |
| BUY | retest1 | 2025-01-17 05:30:00 | 501.85 | 2025-01-20 05:30:00 | 464.27 | STOP_HIT | 1.00 | -7.49% |
