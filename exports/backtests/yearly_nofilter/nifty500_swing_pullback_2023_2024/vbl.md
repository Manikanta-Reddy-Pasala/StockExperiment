# Varun Beverages Ltd. (VBL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 508.85
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 8.40% / 5.55%
- **Sum % (uncompounded):** 42.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 8.40% | 42.0% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 8.40% | 42.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 8.40% | 42.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 00:00:00 | 329.04 | 284.01 | 324.64 | Stage2 pullback-breakout RSI=55 vol=2.4x ATR=8.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-14 00:00:00 | 345.35 | 287.34 | 329.44 | T1 booked 50% @ 345.35 |
| Target hit | 2023-10-04 00:00:00 | 368.18 | 310.01 | 369.44 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 00:00:00 | 396.48 | 322.37 | 371.87 | Stage2 pullback-breakout RSI=68 vol=2.1x ATR=11.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 00:00:00 | 418.47 | 328.01 | 389.31 | T1 booked 50% @ 418.47 |
| Target hit | 2024-01-18 00:00:00 | 492.06 | 377.76 | 494.65 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 590.36 | 449.62 | 565.50 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=17.73 |
| Stop hit — per-position SL triggered | 2024-04-12 00:00:00 | 563.76 | 457.20 | 570.83 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-03 00:00:00 | 329.04 | 2023-08-14 00:00:00 | 345.35 | PARTIAL | 0.50 | 4.96% |
| BUY | retest1 | 2023-08-03 00:00:00 | 329.04 | 2023-10-04 00:00:00 | 368.18 | TARGET_HIT | 0.50 | 11.90% |
| BUY | retest1 | 2023-11-07 00:00:00 | 396.48 | 2023-11-16 00:00:00 | 418.47 | PARTIAL | 0.50 | 5.55% |
| BUY | retest1 | 2023-11-07 00:00:00 | 396.48 | 2024-01-18 00:00:00 | 492.06 | TARGET_HIT | 0.50 | 24.11% |
| BUY | retest1 | 2024-04-03 00:00:00 | 590.36 | 2024-04-12 00:00:00 | 563.76 | STOP_HIT | 1.00 | -4.51% |
