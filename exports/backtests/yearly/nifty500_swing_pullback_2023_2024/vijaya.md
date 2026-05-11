# Vijaya Diagnostic Centre Ltd. (VIJAYA)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1278.90
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
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 6.94% / 7.39%
- **Sum % (uncompounded):** 27.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 6.94% | 27.8% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 6.94% | 27.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 6.94% | 27.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 05:30:00 | 489.75 | 437.46 | 477.69 | Stage2 pullback-breakout RSI=56 vol=1.8x ATR=18.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 05:30:00 | 525.95 | 441.77 | 489.61 | T1 booked 50% @ 525.95 |
| Target hit | 2023-11-23 05:30:00 | 584.00 | 480.36 | 593.15 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 05:30:00 | 674.55 | 504.08 | 620.65 | Stage2 pullback-breakout RSI=68 vol=5.1x ATR=28.09 |
| Stop hit — per-position SL triggered | 2023-12-26 05:30:00 | 632.41 | 508.08 | 625.44 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-05-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 05:30:00 | 730.25 | 596.57 | 680.50 | Stage2 pullback-breakout RSI=66 vol=10.8x ATR=26.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 05:30:00 | 784.03 | 598.62 | 692.16 | T1 booked 50% @ 784.03 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-27 05:30:00 | 489.75 | 2023-10-09 05:30:00 | 525.95 | PARTIAL | 0.50 | 7.39% |
| BUY | retest1 | 2023-09-27 05:30:00 | 489.75 | 2023-11-23 05:30:00 | 584.00 | TARGET_HIT | 0.50 | 19.24% |
| BUY | retest1 | 2023-12-20 05:30:00 | 674.55 | 2023-12-26 05:30:00 | 632.41 | STOP_HIT | 1.00 | -6.25% |
| BUY | retest1 | 2024-05-09 05:30:00 | 730.25 | 2024-05-10 05:30:00 | 784.03 | PARTIAL | 0.50 | 7.37% |
