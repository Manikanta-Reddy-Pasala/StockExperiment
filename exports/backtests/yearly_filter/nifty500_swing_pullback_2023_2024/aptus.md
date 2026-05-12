# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 282.80
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
- **Avg / median % per leg:** 1.48% / 3.87%
- **Sum % (uncompounded):** 11.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 1.48% | 11.8% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 1.48% | 11.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 1 | 4 | 3 | 1.48% | 11.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 00:00:00 | 300.45 | 280.06 | 290.09 | Stage2 pullback-breakout RSI=61 vol=2.6x ATR=9.53 |
| Stop hit — per-position SL triggered | 2023-11-22 00:00:00 | 286.16 | 281.31 | 292.91 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 296.05 | 281.67 | 290.72 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=8.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 00:00:00 | 312.79 | 282.06 | 292.72 | T1 booked 50% @ 312.79 |
| Stop hit — per-position SL triggered | 2023-12-07 00:00:00 | 296.05 | 282.40 | 293.88 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-12-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 00:00:00 | 318.05 | 282.75 | 296.18 | Stage2 pullback-breakout RSI=70 vol=3.7x ATR=9.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 00:00:00 | 336.61 | 284.14 | 304.83 | T1 booked 50% @ 336.61 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 318.05 | 286.23 | 313.55 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-01-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 00:00:00 | 344.05 | 292.06 | 325.55 | Stage2 pullback-breakout RSI=67 vol=3.2x ATR=10.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 00:00:00 | 365.36 | 295.58 | 338.29 | T1 booked 50% @ 365.36 |
| Target hit | 2024-02-09 00:00:00 | 357.35 | 305.01 | 360.82 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 00:00:00 | 347.10 | 314.77 | 325.49 | Stage2 pullback-breakout RSI=63 vol=2.5x ATR=11.46 |
| Stop hit — per-position SL triggered | 2024-04-30 00:00:00 | 329.91 | 315.59 | 328.65 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-10 00:00:00 | 300.45 | 2023-11-22 00:00:00 | 286.16 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest1 | 2023-12-01 00:00:00 | 296.05 | 2023-12-05 00:00:00 | 312.79 | PARTIAL | 0.50 | 5.66% |
| BUY | retest1 | 2023-12-01 00:00:00 | 296.05 | 2023-12-07 00:00:00 | 296.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-08 00:00:00 | 318.05 | 2023-12-13 00:00:00 | 336.61 | PARTIAL | 0.50 | 5.84% |
| BUY | retest1 | 2023-12-08 00:00:00 | 318.05 | 2023-12-20 00:00:00 | 318.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-11 00:00:00 | 344.05 | 2024-01-19 00:00:00 | 365.36 | PARTIAL | 0.50 | 6.19% |
| BUY | retest1 | 2024-01-11 00:00:00 | 344.05 | 2024-02-09 00:00:00 | 357.35 | TARGET_HIT | 0.50 | 3.87% |
| BUY | retest1 | 2024-04-24 00:00:00 | 347.10 | 2024-04-30 00:00:00 | 329.91 | STOP_HIT | 1.00 | -4.95% |
