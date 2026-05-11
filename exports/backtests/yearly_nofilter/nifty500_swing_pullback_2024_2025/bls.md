# BLS International Services Ltd. (BLS)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 281.65
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 3
- **Avg / median % per leg:** 2.88% / 7.31%
- **Sum % (uncompounded):** 17.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 3 | 3 | 2.88% | 17.3% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 3 | 3 | 2.88% | 17.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 3 | 3 | 2.88% | 17.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 00:00:00 | 372.45 | 325.53 | 347.75 | Stage2 pullback-breakout RSI=64 vol=3.4x ATR=13.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 00:00:00 | 399.68 | 328.90 | 361.68 | T1 booked 50% @ 399.68 |
| Stop hit — per-position SL triggered | 2024-07-11 00:00:00 | 372.45 | 329.33 | 362.74 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-08-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 00:00:00 | 385.10 | 333.34 | 357.92 | Stage2 pullback-breakout RSI=63 vol=7.4x ATR=15.97 |
| Stop hit — per-position SL triggered | 2024-08-14 00:00:00 | 361.15 | 336.14 | 367.85 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 00:00:00 | 397.85 | 359.87 | 378.32 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=17.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 00:00:00 | 432.74 | 363.84 | 396.61 | T1 booked 50% @ 432.74 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 397.85 | 364.04 | 395.42 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-12-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 00:00:00 | 419.25 | 367.16 | 394.21 | Stage2 pullback-breakout RSI=63 vol=4.2x ATR=15.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 00:00:00 | 450.39 | 370.40 | 410.53 | T1 booked 50% @ 450.39 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-01 00:00:00 | 372.45 | 2024-07-10 00:00:00 | 399.68 | PARTIAL | 0.50 | 7.31% |
| BUY | retest1 | 2024-07-01 00:00:00 | 372.45 | 2024-07-11 00:00:00 | 372.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-06 00:00:00 | 385.10 | 2024-08-14 00:00:00 | 361.15 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest1 | 2024-10-31 00:00:00 | 397.85 | 2024-11-12 00:00:00 | 432.74 | PARTIAL | 0.50 | 8.77% |
| BUY | retest1 | 2024-10-31 00:00:00 | 397.85 | 2024-11-13 00:00:00 | 397.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 00:00:00 | 419.25 | 2024-12-10 00:00:00 | 450.39 | PARTIAL | 0.50 | 7.43% |
