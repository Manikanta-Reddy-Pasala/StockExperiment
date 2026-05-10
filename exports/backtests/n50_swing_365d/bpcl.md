# BPCL (BPCL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 302.75
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
- **Winners / losers:** 7 / 2
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 1.76% / 2.78%
- **Sum % (uncompounded):** 15.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 7 | 77.8% | 2 | 4 | 3 | 1.76% | 15.8% |
| BUY @ 2nd Alert (retest1) | 9 | 7 | 77.8% | 2 | 4 | 3 | 1.76% | 15.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 7 | 77.8% | 2 | 4 | 3 | 1.76% | 15.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 329.75 | 305.79 | 317.41 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=9.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 05:30:00 | 347.85 | 307.89 | 327.75 | T1 booked 50% @ 347.85 |
| Target hit | 2025-07-25 05:30:00 | 332.90 | 312.74 | 339.57 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 05:30:00 | 329.55 | 314.69 | 319.70 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=6.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 05:30:00 | 342.86 | 316.04 | 327.16 | T1 booked 50% @ 342.86 |
| Stop hit — per-position SL triggered | 2025-10-10 05:30:00 | 338.70 | 317.58 | 334.03 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 343.00 | 319.35 | 335.42 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=7.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 05:30:00 | 358.19 | 320.22 | 339.03 | T1 booked 50% @ 358.19 |
| Target hit | 2025-11-24 05:30:00 | 359.65 | 327.34 | 362.07 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 05:30:00 | 384.00 | 335.68 | 366.87 | Stage2 pullback-breakout RSI=68 vol=2.1x ATR=8.14 |
| Stop hit — per-position SL triggered | 2026-01-06 05:30:00 | 371.79 | 337.34 | 370.36 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2026-01-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 05:30:00 | 362.35 | 339.94 | 360.22 | Stage2 pullback-breakout RSI=51 vol=1.8x ATR=9.54 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 348.03 | 340.64 | 361.04 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 373.45 | 341.22 | 362.71 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=11.11 |
| Stop hit — per-position SL triggered | 2026-02-17 05:30:00 | 374.95 | 345.04 | 373.82 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 329.75 | 2025-07-07 05:30:00 | 347.85 | PARTIAL | 0.50 | 5.49% |
| BUY | retest1 | 2025-06-26 05:30:00 | 329.75 | 2025-07-25 05:30:00 | 332.90 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2025-09-19 05:30:00 | 329.55 | 2025-10-01 05:30:00 | 342.86 | PARTIAL | 0.50 | 4.04% |
| BUY | retest1 | 2025-09-19 05:30:00 | 329.55 | 2025-10-10 05:30:00 | 338.70 | STOP_HIT | 0.50 | 2.78% |
| BUY | retest1 | 2025-10-27 05:30:00 | 343.00 | 2025-10-30 05:30:00 | 358.19 | PARTIAL | 0.50 | 4.43% |
| BUY | retest1 | 2025-10-27 05:30:00 | 343.00 | 2025-11-24 05:30:00 | 359.65 | TARGET_HIT | 0.50 | 4.85% |
| BUY | retest1 | 2025-12-31 05:30:00 | 384.00 | 2026-01-06 05:30:00 | 371.79 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest1 | 2026-01-28 05:30:00 | 362.35 | 2026-02-01 05:30:00 | 348.03 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest1 | 2026-02-03 05:30:00 | 373.45 | 2026-02-17 05:30:00 | 374.95 | STOP_HIT | 1.00 | 0.40% |
