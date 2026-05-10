# Gujarat Mineral Development Corporation Ltd. (GMDCLTD)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 684.70
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
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 3.47% / 7.31%
- **Sum % (uncompounded):** 27.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 3.47% | 27.8% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 3.47% | 27.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 1 | 4 | 3 | 3.47% | 27.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 05:30:00 | 435.30 | 343.25 | 394.18 | Stage2 pullback-breakout RSI=69 vol=12.3x ATR=15.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 05:30:00 | 467.14 | 345.30 | 403.80 | T1 booked 50% @ 467.14 |
| Stop hit — per-position SL triggered | 2025-07-25 05:30:00 | 435.30 | 348.34 | 415.12 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2025-09-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 05:30:00 | 459.70 | 364.08 | 423.21 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=19.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 05:30:00 | 499.30 | 366.42 | 434.14 | T1 booked 50% @ 499.30 |
| Target hit | 2025-10-17 05:30:00 | 567.10 | 420.78 | 583.34 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-11-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 05:30:00 | 553.15 | 454.21 | 545.27 | Stage2 pullback-breakout RSI=50 vol=8.3x ATR=28.25 |
| Stop hit — per-position SL triggered | 2025-12-08 05:30:00 | 510.77 | 459.09 | 534.19 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2025-12-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 05:30:00 | 589.00 | 466.64 | 529.61 | Stage2 pullback-breakout RSI=65 vol=4.9x ATR=27.62 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 551.55 | 478.91 | 569.49 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-01-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 05:30:00 | 618.35 | 487.36 | 559.12 | Stage2 pullback-breakout RSI=64 vol=2.8x ATR=29.31 |
| Stop hit — per-position SL triggered | 2026-01-30 05:30:00 | 574.39 | 488.26 | 560.89 | SL hit (bars_held=1) |

### Cycle 6 — BUY (started 2026-03-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 05:30:00 | 575.70 | 510.89 | 553.46 | Stage2 pullback-breakout RSI=55 vol=8.6x ATR=27.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 05:30:00 | 631.06 | 521.24 | 585.55 | T1 booked 50% @ 631.06 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-18 05:30:00 | 435.30 | 2025-07-22 05:30:00 | 467.14 | PARTIAL | 0.50 | 7.31% |
| BUY | retest1 | 2025-07-18 05:30:00 | 435.30 | 2025-07-25 05:30:00 | 435.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 05:30:00 | 459.70 | 2025-09-05 05:30:00 | 499.30 | PARTIAL | 0.50 | 8.61% |
| BUY | retest1 | 2025-09-03 05:30:00 | 459.70 | 2025-10-17 05:30:00 | 567.10 | TARGET_HIT | 0.50 | 23.36% |
| BUY | retest1 | 2025-11-27 05:30:00 | 553.15 | 2025-12-08 05:30:00 | 510.77 | STOP_HIT | 1.00 | -7.66% |
| BUY | retest1 | 2025-12-26 05:30:00 | 589.00 | 2026-01-09 05:30:00 | 551.55 | STOP_HIT | 1.00 | -6.36% |
| BUY | retest1 | 2026-01-29 05:30:00 | 618.35 | 2026-01-30 05:30:00 | 574.39 | STOP_HIT | 1.00 | -7.11% |
| BUY | retest1 | 2026-03-20 05:30:00 | 575.70 | 2026-04-15 05:30:00 | 631.06 | PARTIAL | 0.50 | 9.62% |
