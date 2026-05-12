# Apar Industries Ltd. (APARINDS)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 12810.00
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
- **Avg / median % per leg:** 3.94% / 7.93%
- **Sum % (uncompounded):** 19.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.94% | 19.7% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.94% | 19.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.94% | 19.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 8653.80 | 6975.42 | 8400.33 | Stage2 pullback-breakout RSI=56 vol=2.3x ATR=375.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 00:00:00 | 9404.72 | 6996.66 | 8467.91 | T1 booked 50% @ 9404.72 |
| Stop hit — per-position SL triggered | 2024-08-02 00:00:00 | 8653.80 | 7052.45 | 8571.24 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 00:00:00 | 10649.70 | 7881.96 | 9685.22 | Stage2 pullback-breakout RSI=69 vol=1.9x ATR=399.14 |
| Stop hit — per-position SL triggered | 2024-10-21 00:00:00 | 10050.99 | 8050.92 | 10012.72 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-11-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 00:00:00 | 9479.30 | 8289.50 | 9341.16 | Stage2 pullback-breakout RSI=51 vol=4.3x ATR=413.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 00:00:00 | 10306.63 | 8420.60 | 9660.29 | T1 booked 50% @ 10306.63 |
| Target hit | 2025-01-13 00:00:00 | 10231.15 | 8878.28 | 10432.43 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-29 00:00:00 | 8653.80 | 2024-07-30 00:00:00 | 9404.72 | PARTIAL | 0.50 | 8.68% |
| BUY | retest1 | 2024-07-29 00:00:00 | 8653.80 | 2024-08-02 00:00:00 | 8653.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 00:00:00 | 10649.70 | 2024-10-21 00:00:00 | 10050.99 | STOP_HIT | 1.00 | -5.62% |
| BUY | retest1 | 2024-11-19 00:00:00 | 9479.30 | 2024-12-03 00:00:00 | 10306.63 | PARTIAL | 0.50 | 8.73% |
| BUY | retest1 | 2024-11-19 00:00:00 | 9479.30 | 2025-01-13 00:00:00 | 10231.15 | TARGET_HIT | 0.50 | 7.93% |
