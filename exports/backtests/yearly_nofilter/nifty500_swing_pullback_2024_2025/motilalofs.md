# Motilal Oswal Financial Services Ltd. (MOTILALOFS)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 863.00
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
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 3.87% / 8.29%
- **Sum % (uncompounded):** 19.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.87% | 19.4% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.87% | 19.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.87% | 19.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 00:00:00 | 590.75 | 461.48 | 565.31 | Stage2 pullback-breakout RSI=57 vol=2.4x ATR=25.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 00:00:00 | 641.66 | 464.67 | 575.86 | T1 booked 50% @ 641.66 |
| Stop hit — per-position SL triggered | 2024-08-06 00:00:00 | 590.75 | 472.96 | 597.72 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-08-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 00:00:00 | 677.45 | 485.22 | 610.93 | Stage2 pullback-breakout RSI=66 vol=2.6x ATR=28.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 00:00:00 | 733.64 | 494.18 | 645.57 | T1 booked 50% @ 733.64 |
| Target hit | 2024-09-27 00:00:00 | 750.65 | 550.44 | 751.77 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 00:00:00 | 997.65 | 635.24 | 918.95 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=55.62 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 914.22 | 644.45 | 925.61 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-26 00:00:00 | 590.75 | 2024-07-30 00:00:00 | 641.66 | PARTIAL | 0.50 | 8.62% |
| BUY | retest1 | 2024-07-26 00:00:00 | 590.75 | 2024-08-06 00:00:00 | 590.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 00:00:00 | 677.45 | 2024-08-26 00:00:00 | 733.64 | PARTIAL | 0.50 | 8.29% |
| BUY | retest1 | 2024-08-20 00:00:00 | 677.45 | 2024-09-27 00:00:00 | 750.65 | TARGET_HIT | 0.50 | 10.81% |
| BUY | retest1 | 2024-11-08 00:00:00 | 997.65 | 2024-11-13 00:00:00 | 914.22 | STOP_HIT | 1.00 | -8.36% |
