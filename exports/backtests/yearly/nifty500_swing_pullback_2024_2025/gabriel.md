# Gabriel India Ltd. (GABRIEL)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1131.90
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 1.88% / 1.73%
- **Sum % (uncompounded):** 13.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.88% | 13.1% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.88% | 13.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.88% | 13.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 05:30:00 | 499.00 | 388.84 | 479.25 | Stage2 pullback-breakout RSI=65 vol=1.5x ATR=20.44 |
| Stop hit — per-position SL triggered | 2024-08-05 05:30:00 | 468.34 | 395.00 | 485.41 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-08-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 05:30:00 | 511.65 | 401.77 | 491.17 | Stage2 pullback-breakout RSI=61 vol=4.3x ATR=22.06 |
| Stop hit — per-position SL triggered | 2024-08-19 05:30:00 | 478.56 | 403.66 | 492.35 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-08-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 05:30:00 | 548.35 | 414.82 | 516.33 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=23.58 |
| Stop hit — per-position SL triggered | 2024-09-09 05:30:00 | 512.98 | 421.56 | 522.13 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-12-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 05:30:00 | 480.80 | 439.90 | 445.78 | Stage2 pullback-breakout RSI=63 vol=10.9x ATR=17.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 05:30:00 | 515.90 | 441.19 | 456.70 | T1 booked 50% @ 515.90 |
| Stop hit — per-position SL triggered | 2024-12-20 05:30:00 | 489.10 | 446.39 | 484.61 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-03-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 05:30:00 | 495.70 | 454.21 | 471.66 | Stage2 pullback-breakout RSI=57 vol=8.0x ATR=27.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 05:30:00 | 551.07 | 459.39 | 500.57 | T1 booked 50% @ 551.07 |
| Target hit | 2025-04-04 05:30:00 | 555.20 | 473.42 | 557.70 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2025-05-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 05:30:00 | 581.75 | 487.12 | 555.66 | Stage2 pullback-breakout RSI=59 vol=2.2x ATR=27.54 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-26 05:30:00 | 499.00 | 2024-08-05 05:30:00 | 468.34 | STOP_HIT | 1.00 | -6.15% |
| BUY | retest1 | 2024-08-14 05:30:00 | 511.65 | 2024-08-19 05:30:00 | 478.56 | STOP_HIT | 1.00 | -6.47% |
| BUY | retest1 | 2024-08-30 05:30:00 | 548.35 | 2024-09-09 05:30:00 | 512.98 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest1 | 2024-12-06 05:30:00 | 480.80 | 2024-12-10 05:30:00 | 515.90 | PARTIAL | 0.50 | 7.30% |
| BUY | retest1 | 2024-12-06 05:30:00 | 480.80 | 2024-12-20 05:30:00 | 489.10 | STOP_HIT | 0.50 | 1.73% |
| BUY | retest1 | 2025-03-05 05:30:00 | 495.70 | 2025-03-18 05:30:00 | 551.07 | PARTIAL | 0.50 | 11.17% |
| BUY | retest1 | 2025-03-05 05:30:00 | 495.70 | 2025-04-04 05:30:00 | 555.20 | TARGET_HIT | 0.50 | 12.00% |
