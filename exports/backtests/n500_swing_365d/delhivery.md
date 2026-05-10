# Delhivery Ltd. (DELHIVERY)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 479.10
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
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 3.62% / 6.29%
- **Sum % (uncompounded):** 28.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 2 | 3 | 3 | 3.62% | 28.9% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 2 | 3 | 3 | 3.62% | 28.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 2 | 3 | 3 | 3.62% | 28.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 05:30:00 | 377.05 | 337.70 | 359.24 | Stage2 pullback-breakout RSI=66 vol=2.4x ATR=11.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 05:30:00 | 400.80 | 342.62 | 379.34 | T1 booked 50% @ 400.80 |
| Target hit | 2025-07-30 05:30:00 | 409.15 | 354.65 | 415.73 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 05:30:00 | 462.60 | 394.17 | 457.52 | Stage2 pullback-breakout RSI=52 vol=4.2x ATR=12.92 |
| Stop hit — per-position SL triggered | 2025-10-20 05:30:00 | 474.90 | 401.00 | 462.67 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 05:30:00 | 422.25 | 410.19 | 410.04 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=9.49 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 408.01 | 410.22 | 410.29 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2026-01-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 05:30:00 | 422.95 | 408.82 | 403.40 | Stage2 pullback-breakout RSI=61 vol=2.9x ATR=13.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 05:30:00 | 449.56 | 409.10 | 406.58 | T1 booked 50% @ 449.56 |
| Stop hit — per-position SL triggered | 2026-02-12 05:30:00 | 422.95 | 411.56 | 424.34 | SL hit (bars_held=10) |

### Cycle 5 — BUY (started 2026-04-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 05:30:00 | 442.00 | 415.14 | 424.72 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=15.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 05:30:00 | 472.38 | 416.12 | 432.05 | T1 booked 50% @ 472.38 |
| Target hit | 2026-04-23 05:30:00 | 449.40 | 420.15 | 449.97 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 05:30:00 | 377.05 | 2025-07-08 05:30:00 | 400.80 | PARTIAL | 0.50 | 6.30% |
| BUY | retest1 | 2025-06-24 05:30:00 | 377.05 | 2025-07-30 05:30:00 | 409.15 | TARGET_HIT | 0.50 | 8.51% |
| BUY | retest1 | 2025-10-06 05:30:00 | 462.60 | 2025-10-20 05:30:00 | 474.90 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest1 | 2026-01-07 05:30:00 | 422.25 | 2026-01-09 05:30:00 | 408.01 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest1 | 2026-01-30 05:30:00 | 422.95 | 2026-02-01 05:30:00 | 449.56 | PARTIAL | 0.50 | 6.29% |
| BUY | retest1 | 2026-01-30 05:30:00 | 422.95 | 2026-02-12 05:30:00 | 422.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 05:30:00 | 442.00 | 2026-04-09 05:30:00 | 472.38 | PARTIAL | 0.50 | 6.87% |
| BUY | retest1 | 2026-04-07 05:30:00 | 442.00 | 2026-04-23 05:30:00 | 449.40 | TARGET_HIT | 0.50 | 1.67% |
