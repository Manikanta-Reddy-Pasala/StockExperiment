# Aadhar Housing Finance Ltd. (AADHARHFC)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 501.40
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 0.14% / 1.18%
- **Sum % (uncompounded):** 0.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 4 | 1 | 0.14% | 0.9% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 4 | 1 | 0.14% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 4 | 1 | 0.14% | 0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 454.60 | 428.32 | 443.86 | Stage2 pullback-breakout RSI=58 vol=2.4x ATR=12.90 |
| Stop hit — per-position SL triggered | 2025-07-14 05:30:00 | 459.95 | 431.06 | 451.73 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-07-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 05:30:00 | 477.90 | 431.53 | 454.22 | Stage2 pullback-breakout RSI=68 vol=4.3x ATR=14.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 05:30:00 | 505.99 | 432.26 | 459.02 | T1 booked 50% @ 505.99 |
| Target hit | 2025-08-13 05:30:00 | 496.70 | 445.61 | 498.12 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-09-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 05:30:00 | 537.80 | 458.53 | 511.70 | Stage2 pullback-breakout RSI=66 vol=5.4x ATR=13.88 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 516.98 | 464.18 | 522.14 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2026-01-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 05:30:00 | 500.55 | 479.03 | 486.29 | Stage2 pullback-breakout RSI=59 vol=3.6x ATR=10.94 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 484.15 | 479.69 | 488.25 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2026-01-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 05:30:00 | 491.30 | 479.14 | 479.11 | Stage2 pullback-breakout RSI=59 vol=2.6x ATR=9.75 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 476.67 | 479.16 | 479.29 | SL hit (bars_held=1) |

### Cycle 6 — BUY (started 2026-05-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 05:30:00 | 517.45 | 476.57 | 486.93 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=16.36 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 454.60 | 2025-07-14 05:30:00 | 459.95 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest1 | 2025-07-15 05:30:00 | 477.90 | 2025-07-16 05:30:00 | 505.99 | PARTIAL | 0.50 | 5.88% |
| BUY | retest1 | 2025-07-15 05:30:00 | 477.90 | 2025-08-13 05:30:00 | 496.70 | TARGET_HIT | 0.50 | 3.93% |
| BUY | retest1 | 2025-09-16 05:30:00 | 537.80 | 2025-09-26 05:30:00 | 516.98 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest1 | 2026-01-02 05:30:00 | 500.55 | 2026-01-09 05:30:00 | 484.15 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest1 | 2026-01-30 05:30:00 | 491.30 | 2026-02-01 05:30:00 | 476.67 | STOP_HIT | 1.00 | -2.98% |
